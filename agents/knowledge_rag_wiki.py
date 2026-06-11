"""
知识查询 Agent — 基于 LLM Wiki + 混合检索（参考 llm-wiki-compiler pickSearchSlugs 三级降级）

检索流程（三级降级）：
  第1级  chunk 嵌入检索 → 定位最相关页面       ← 0 次 LLM，最省 token
  第2级  page 嵌入检索  → 定位相关页面         ← 0 次 LLM
  第3级  LLM 读 index.md → 选择页面           ← 1 次 LLM（兜底）

定位完成后：读取页面 Markdown → LLM 生成回答（1 次 LLM）

相比纯 LLM 读 index 方案，当嵌入命中时省掉定位阶段的 LLM 调用 + index 注入 token，
定位阶段从 ~1500 token 降至 0。

[接口兼容] process(state) 签名与原版完全一致，对 supervisor.py 透明
"""


from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from tracing.otel_config import trace_agent_call

from memory.wiki_knowledge_base import WikiknowledgeBase
from memory.wiki_hybrid_retriever import WikiHybridRetriever
from memory.long_term import LongTermMemory


class KnowledgeRAGAgent:
    """
    知识查询 Agent — LLM Wiki + 混合检索版本。

    对外接口与原版完全相同，supervisor.py 无需修改。
    内部使用 WikiHybridRetriever 做嵌入定位(chunk→page)，
    命中则 0 次 LLM 调用即可定位页面；未命中则降级到 LLM 读 index 兜底。
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        wiki: WikiknowledgeBase | None = None,
        long_term_memory=None,   # 保留参数名，兼容 supervisor 传参，实际不使用
    ):
        self.llm  = llm or ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

        # 构建混合检索器
        from memory.wiki_knowledge_base import _get_wiki_root
        wiki_dir = _get_wiki_root() / "wiki"
        self.retriever = WikiHybridRetriever(
            wiki_dir=wiki_dir,
            embedder=LongTermMemory(),
            chunk_top_k=5,
            page_top_k=3,
            min_score=0.30,
        )
        # 首次构建索引（增量，已 embed 的页面走缓存不重复调用 API）
        self.retriever.build_index()

        self.wiki = wiki or WikiknowledgeBase(llm=self.llm, retriever=self.retriever)

    @trace_agent_call("knowledge_wiki_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Wiki 查询流程（作为 LangGraph 节点）。
        state 输入/输出格式与原版完全一致。
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        user_question = messages[-1].content

        # Wiki 查询：三级降级定位 + 生成回答
        # save_answer=True：优质问答自动存入 qa_archive，形成知识复利
        answer = await self.wiki.query(user_question, save_answer=True)

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "knowledge_rag": answer,
            },
        }