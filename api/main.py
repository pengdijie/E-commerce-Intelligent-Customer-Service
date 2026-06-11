"""
FastAPI入口 — 提供REST API + SSE流式响应

新增功能：

SSE 流式回复接口 

设计思路
    langGraph 的 astream_events()会在每个节点执行吐出事件，
    我们过滤出关系的两类事件推送给前端：
    1. on_chat_model_stream  -> LLM 正在生成文字，逐token推送
    2. on_chain_start        -> 某个Agent节点开始执行，推送“进度提示”


SSE 数据格式（每条都是完整的JSON）

"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agents.supervisor import create_supervisor_graph
from memory.working_memory import WorkingMemory
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from mcp.mcp_server import MCPToolServer, create_default_tools
from tracing.otel_config import init_tracer, get_metrics
from memory.wiki_knowledge_base import WikiknowledgeBase

load_dotenv()


working_memory = WorkingMemory()
short_term_memory = ShortTermMemory(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
Wiki = WikiknowledgeBase()
mcp_server = create_default_tools(MCPToolServer())
metrics = get_metrics()
graph = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global graph

    init_tracer(
        service_name=os.getenv("OTEL_SERVICE_NAME", "smart-cs-multi-agent"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    graph = create_supervisor_graph(
        working_memory=working_memory,
        short_term_memory=short_term_memory,
        wiki=Wiki,
    )

    yield


app = FastAPI(
    title="智能客服多Agent系统",
    description="基于LangGraph的Supervisor编排多Agent智能客服系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- 数据模型 -----------------

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    compliance_passed: bool


#  ------- Agent 节点名称 -> 用户可读的进度提示 ------------
AGENT_PROGRESS_MESSAGES: dict[str, str] = {
    "supervisor_route":     "🔍 分析您的问题...",
    "knowledge_rag":        "📚 查询知识库...",
    "ticket_handler":       "🎫 处理工单...",
    "compliance_check":     "✅ 合规审查...",
    "synthesize":           "✍️ 生成回复...",
}

def _sse(data: dict) -> str:
    """把 dict 序列化成标准 SSE 格式的字符串，前端 EventSource 会自动解析"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

async def _stream_graph(
        session_id: str,
        initial_state: dict,
        config: dict,
) -> AsyncGenerator[str, None]:
    """
        核心生成器: 订阅 LangGraph astream_events，过滤并转发给前端

        LangGraph v2 事件说明:
        - on_chain_start:        一个节点开始运行
        - on_chat_model_stream:  LLM 正在流式输出 token
        - on_chain_end:          某个 chain 执行完毕
    """
    current_agent = ""
    final_state: dict = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}  # 累计本次对话 token

    try:
        async for event in graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            kind = event["event"]
            name = event.get("name", "")

            # 1) 节点开始 → 推送进度提示
            if kind == "on_chain_start" and name in AGENT_PROGRESS_MESSAGES:
                current_agent = name
                yield _sse({
                    "type": "progress",
                    "agent": name,
                    "message": AGENT_PROGRESS_MESSAGES[name],
                })

            # 2) LLM 流式 token → 只把"最终回答"那次 LLM 调用推给前端
            #    其他中间调用（query 改写、文档重排、合规 JSON 等）通过 tags 区分
            elif kind == "on_chat_model_stream":
                tags = event.get("tags") or []
                if "final_answer" not in tags:
                    continue
                chunk = event["data"].get("chunk")
                if chunk is not None and getattr(chunk, "content", ""):
                    yield _sse({
                        "type": "token",
                        "agent": current_agent,
                        "content": chunk.content,
                    })

            # 3) 任意 chain 结束时记录最后一次 output —— 最后一个会是顶层 graph
            elif kind == "on_chain_end":
                output = event["data"].get("output")
                if isinstance(output, dict) and "final_response" in output:
                    final_state = output

            # 4) 每次 LLM 调用结束 → 累计 token 用量（覆盖所有 Agent 节点）
            elif kind == "on_chat_model_end":
                out = event["data"].get("output")
                meta = getattr(out, "usage_metadata", None)
                if meta:
                    usage["input_tokens"] += meta.get("input_tokens", 0)
                    usage["output_tokens"] += meta.get("output_tokens", 0)
                    usage["total_tokens"] += meta.get("total_tokens", 0)

        final_response = final_state.get("final_response", "")
        if final_response:
            await short_term_memory.add_message(session_id, "assistant", final_response)

        yield _sse({
            "type": "done",
            "session_id": session_id,
            "intent": final_state.get("intent", "unknown"),
            "compliance_passed": final_state.get("compliance_passed", True),
            "response": final_response,
            "usage": usage,
        })

    except Exception as e:
        yield _sse({"type": "error", "message": f"处理失败：{str(e)}"})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """主聊天接口"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())

    await short_term_memory.add_message(session_id, "user", request.message)

    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "final_response": "",
        "current_agent": "",
        "retry_count": 0,
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    final_response = result.get("final_response", "系统处理异常，请稍后重试")

    await short_term_memory.add_message(session_id, "assistant", final_response)

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        intent=result.get("intent", "unknown"),
        compliance_passed=result.get("compliance_passed", True),
    )

@app.post("/api/chat/stream")
async def chat_stream( request: ChatRequest):
    """
        SSE 流式接口 — 实时推送 Agent 进度和 LLM token。

        前端接入示例（原生 fetch）：

            const res = await fetch("/api/chat/stream",{
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({message: "我要退款", user_id: "u001"}),
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while(true){
                const{done, value } = await reader.read();
                if(done)break;
                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split("\\n");
                buffer = lines.pop();          // 保留未完成的行

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const event = JSON.parse(line.slice(6));

                    switch(event_type){
                        case "progress":
                            showStatus(event.message);                  # 查询知识库
                            break;
                        case "token":
                            appendToken(event.content);                 # 打字机效果
                            break;
                        case "done":
                            hideStatus();
                            break;
                        case "error":
                            showError(event.message);
                            break;   
                    }
                }
            }
    注意事项：
      - supervisor.py 里 ChatOpenAI 必须加 streaming=True，否则收不到 token 事件
      - Nginx 反向代理需关闭缓冲：proxy_buffering off
    """
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())
    await short_term_memory.add_message(session_id, "user", request.message)

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "final_response": "",
        "current_agent": "",
        "retry_count": 0,
    }
    config = {"configurable": {"thread_id": session_id}}

    return StreamingResponse(
        _stream_graph(session_id, initial_state, config),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",      # 关闭 Nginx 缓冲
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """获取对话历史"""
    history = await short_term_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/api/tools")
async def list_tools():
    """MCP工具发现接口"""
    return {"tools": mcp_server.list_tools()}


@app.post("/api/tools/call")
async def call_tool(request: dict):
    """MCP工具调用接口"""
    result = await mcp_server.call_tool(
        name=request.get("name", ""),
        arguments=request.get("arguments", {}),
    )
    return {
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.get("/api/metrics")
async def get_metrics():
    """获取系统指标"""
    return {
        "agent_metrics": metrics.get_summary(),
        "tool_call_log": mcp_server.get_call_log(last_n=20),
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


# 静态前端 — 必须放在所有 API 路由之后注册，否则会盖掉 /api/* 路由
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
