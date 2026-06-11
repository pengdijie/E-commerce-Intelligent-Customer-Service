"""
知识查询 Agent — Wiki编译 + RAG检索（分层架构）

架构定位：
  LLM Wiki：离线编译层 — 把原始数据整理成结构化 Markdown 页面（一次性成本）
  FAISS RAG：在线查询层 — 从编译产物中检索相关 chunks 并生成回答

查询流程（极简，仅 1 次 LLM 调用）：
  1. 用户问题 → Embedding → FAISS 向量检索 top-k chunks
  2. Top-k chunks 直接作为 context 注入 LLM → 生成回答

为什么不做 Query 改写？
  Wiki 编译产物已经是结构化、术语统一的知识页面，嵌入质量高，
  用户原始问题的向量就能精准命中，省掉 1 次 LLM 调用。

为什么不做 LLM 重排？
  Wiki 页面结构清晰（标题+正文），chunk 级别粒度小，
  向量相似度排序已经足够好，省掉 1 次 LLM 调用。

对比：
  纯 RAG：query改写(1 LLM) + 检索 + LLM重排(1 LLM) + 生成(1 LLM) = 3次 LLM
  纯 Wiki：embed定位/LLM读index + 整页注入 + 生成(1 LLM)         = 1-2次 LLM, 但 input 长
  Wiki+RAG：embed检索 + chunk注入 + 生成(1 LLM)                  = 1次 LLM, input 短

[接口兼容] process(state) 签名与原版完全一致，对 supervisor.py 透明
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tracing.otel_config import trace_agent_call

from memory.long_term import LongTermMemory


RAG_SYSTEM_PROMPT = """你是一个专业的电商客服知识库问答Agent。

回答规则：
1. 严格基于检索到的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确告知用户并建议转人工
3. 回答要简洁专业，适合客服场景
4. 在回答末尾标注引用的文档来源

回答格式：
- 先直接回答用户问题
- 如有必要补充相关信息
- 如果涉及金融/保修等政策，添加"以上信息仅供参考，具体以实际条款为准"
"""


class KnowledgeRAGAgent:
    """
    知识查询 Agent — Wiki编译+RAG检索 分层架构版本。

    对外接口与原版完全相同，supervisor.py 无需修改。
    知识源来自 LLM Wiki 编译产物（wiki/entities/*.md + wiki/topics/*.md），
    查询时走纯向量检索，仅 1 次 LLM 调用即可生成回答。
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        long_term_memory: LongTermMemory | None = None,
        wiki_pages_dir: str | None = None,
        top_k: int = 5,
        context_k: int = 3,
    ):
        import os
        self.llm = llm or ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            streaming=True,
        )
        self.top_k = top_k
        self.context_k = context_k

        # 知识源：wiki 编译产物
        from pathlib import Path
        if wiki_pages_dir:
            self._wiki_dir = Path(wiki_pages_dir)
        else:
            self._wiki_dir = Path(__file__).resolve().parent.parent / "wiki_knowledge" / "wiki"

        # 构建向量索引（从 wiki 结构化页面）
        self.long_term_memory = long_term_memory or self._build_index()

    def _build_index(self) -> LongTermMemory:
        """从 wiki 编译产物构建 FAISS 向量索引（批量 embedding + 磁盘持久化）"""
        ltm = LongTermMemory(index_path="./vector_store/wiki_rag_index")

        if ltm._documents:
            return ltm

        import faiss
        ltm._index = faiss.IndexFlatIP(ltm.embedding_dim)
        ltm._documents = []

        docs_to_add = []
        for sub in ("entities", "topics"):
            sub_dir = self._wiki_dir / sub
            if not sub_dir.exists():
                continue
            for fp in sorted(sub_dir.glob("*.md")):
                content = fp.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                chunks = LongTermMemory._chunk_text(content)
                for chunk in chunks:
                    docs_to_add.append({
                        "content": chunk,
                        "source": fp.stem,
                        "metadata": {"page": f"{sub}/{fp.name}"},
                    })

        if docs_to_add:
            ltm.add_documents_batch(docs_to_add)
            ltm.save_index()

        print(f"[KnowledgeRAGAgent-Hybrid] 索引构建完成: "
              f"{len(ltm._documents)} chunks from {self._wiki_dir}")
        return ltm

    @trace_agent_call("knowledge_hybrid_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Wiki+RAG 查询流程（作为 LangGraph 节点）。
        state 输入/输出格式与原版完全一致。
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        user_question = messages[-1].content

        # 向量检索（0 次 LLM）
        docs = await self.long_term_memory.search_async(user_question, top_k=self.top_k)

        # 取 top context_k 直接用
        selected = docs[:self.context_k]

        if not selected:
            answer = "抱歉，知识库中暂未找到与您问题相关的信息。建议您联系人工客服获取帮助。"
        else:
            # 1 次 LLM 生成回答
            context = "\n\n---\n\n".join(
                f"来源: {doc.get('source', '未知')}\n内容: {doc.get('content', '')}"
                for doc in selected
            )
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=RAG_SYSTEM_PROMPT),
                    HumanMessage(content=f"用户问题: {user_question}\n\n检索到的参考文档:\n{context}"),
                ],
                config={"tags": ["final_answer"]},
            )
            answer = response.content

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "knowledge_rag": answer,
            },
        }
