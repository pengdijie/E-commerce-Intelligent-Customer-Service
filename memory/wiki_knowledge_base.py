"""
WikiKnowledgeBase — 基于 Karpathy LLM Wiki 模式的知识库

核心修复：
  LLM 输出 JSON 时 content 字段含真实换行会导致解析失败。
  解决方案：让 LLM 分两步输出——先输出结构 JSON（不含 content），
  再输出各页面的实际 Markdown 内容（用 ===SPLIT=== 分隔）。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def _get_wiki_root() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / "wiki_knowledge"


class WikiknowledgeBase:
    def __init__(self, llm: ChatOpenAI | None = None, wiki_root: str | None = None,
                 retriever=None):
        self.llm = llm or ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.wiki_root  = Path(wiki_root).resolve() if wiki_root else _get_wiki_root()
        self.raw_dir    = self.wiki_root / "raw"
        self.wiki_dir   = self.wiki_root / "wiki"
        self.index_file = self.wiki_dir / "index.md"
        self.log_file   = self.wiki_dir / "log.md"
        self.schema_file = self.wiki_root / "SCHEMA.md"
        self.state_file  = self.wiki_root / "state.json"  # 增量编译哈希状态
        self.retriever  = retriever  # WikiHybridRetriever | None
        self._ensure_dirs()
        print(f"[WikiknowledgeBase] wiki_root = {self.wiki_root}")

    def _ensure_dirs(self):
        for d in [
            self.wiki_dir / "entities",
            self.wiki_dir / "topics",
            self.wiki_dir / "qa_archive",
            self.raw_dir / "products",
            self.raw_dir / "reviews",
            self.raw_dir / "orders",
        ]:
            d.mkdir(parents=True, exist_ok=True)

        if not self.index_file.exists() or self.index_file.stat().st_size == 0:
            self.index_file.write_text(
                "# 知识库索引\n\n"
                "> 本文件由系统自动维护。\n\n"
                "## 实体页（entities/）\n\n"
                "## 主题页（topics/）\n\n"
                "## 问答存档（qa_archive/）\n",
                encoding="utf-8",
            )
        if not self.log_file.exists() or self.log_file.stat().st_size == 0:
            self.log_file.write_text(
                f"# 摄入日志\n\n## [{datetime.now().date()}] init | 初始化\n",
                encoding="utf-8",
            )

    # ── Ingest ────────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        """读取 state.json — 记录每个源文件的 SHA-256 哈希,用于增量编译门控。"""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def ingest(self, source_path: str, source_type: str = "auto") -> dict:
        path = Path(source_path)
        content     = path.read_text(encoding="utf-8") if path.exists() else source_path
        source_name = path.name if path.exists() else f"inline_{datetime.now().strftime('%H%M%S')}"

        # ── 增量编译: SHA-256 哈希门控 ──────────────────────────────────────
        # 源内容没变(哈希一致)则跳过 LLM 编译,节省 token
        content_hash = self._content_hash(content)
        # state_key: 文件用文件名(路径变内容不变照样跳过); inline 用哈希前缀(相同内容必跳过)
        state_key = path.name if path.exists() else f"inline:{content_hash[:12]}"
        state = self._load_state()
        if state.get(state_key, {}).get("hash") == content_hash:
            print(f"[WikiKB] 增量跳过: {state_key} (SHA-256 未变)")
            return {
                "pages_created": 0, "pages_updated": 0,
                "source": source_name, "skipped": True, "reason": "unchanged",
            }

        index = self.index_file.read_text(encoding="utf-8")

        # ── 第一步：让 LLM 输出"页面计划"（只含路径和索引条目，不含正文）
        plan_prompt = f"""你是客服知识库维护 Agent。分析数据，规划需要创建哪些 Wiki 页面。

现有索引：
{index}

待摄入数据（来源：{source_name}，类型：{source_type}）：
{content[:4000]}

输出合法 JSON（不要代码块，content 字段留空字符串）：
{{
  "pages": [
    {{"path": "entities/xxx.md", "action": "create", "title": "页面标题"}},
    {{"path": "topics/yyy.md",   "action": "create", "title": "页面标题"}}
  ],
  "index_entity_entries": "- [[entities/xxx.md]] — 一句话描述",
  "index_topic_entries":  "- [[topics/yyy.md]] — 一句话描述",
  "log_entry": "## [{datetime.now().date()}] ingest | {source_name}"
}}

规则：
- 必须同时包含实体页（entities/）和主题页（topics/）
- path 只写相对路径如 entities/product_a.md
- 不要在 JSON 里写页面正文内容"""

        plan_resp = await self.llm.ainvoke([HumanMessage(content=plan_prompt)])
        raw = re.sub(r'```(?:json)?|```', '', plan_resp.content).strip()

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [警告] 计划解析失败: {e}\n  原始: {raw[:300]}")
            return {"pages_created": 0, "pages_updated": 0, "source": source_name, "error": str(e)}

        pages = plan.get("pages", [])
        if not pages:
            return {"pages_created": 0, "pages_updated": 0, "source": source_name}

        # ── 第二步：让 LLM 逐页生成正文（每页单独一次调用，彻底避免换行问题）
        created = updated = 0
        schema = self.schema_file.read_text(encoding="utf-8") if self.schema_file.exists() else ""

        for page_info in pages:
            rel_path = page_info.get("path", "")
            if not rel_path or ".." in rel_path:
                continue

            title    = page_info.get("title", rel_path)
            action   = page_info.get("action", "create")
            page_type = "entity" if rel_path.startswith("entities/") else "topic"

            existing_content = ""
            page_path = self.wiki_dir / rel_path
            if page_path.exists():
                existing_content = f"\n\n现有内容（请在此基础上更新）：\n{page_path.read_text(encoding='utf-8')}"

            content_prompt = f"""为客服知识库生成一个 Wiki 页面的完整 Markdown 内容。

页面信息：
- 路径：{rel_path}
- 类型：{page_type}（{"实体页：具体商品/政策信息" if page_type == "entity" else "主题页：归纳某类问题的综合指南"}）
- 标题：{title}
- 操作：{action}
{existing_content}

原始数据（来源：{source_name}）：
{content[:3000]}

直接输出 Markdown 内容，第一行必须是 frontmatter（--- 开头），不要任何前缀解释："""

            content_resp = await self.llm.ainvoke([HumanMessage(content=content_prompt)])
            page_content = content_resp.content.strip()

            page_path.parent.mkdir(parents=True, exist_ok=True)
            existed = page_path.exists()
            page_path.write_text(page_content, encoding="utf-8")

            if existed and action == "update":
                updated += 1
            else:
                created += 1

        # ── 更新 index.md ──────────────────────────────────────────────────
        index_text = self.index_file.read_text(encoding="utf-8")
        entity_entries = plan.get("index_entity_entries", "")
        topic_entries  = plan.get("index_topic_entries", "")

        if entity_entries and entity_entries not in index_text:
            index_text = index_text.replace(
                "## 实体页（entities/）\n",
                f"## 实体页（entities/）\n{entity_entries}\n",
            )
        if topic_entries and topic_entries not in index_text:
            index_text = index_text.replace(
                "## 主题页（topics/）\n",
                f"## 主题页（topics/）\n{topic_entries}\n",
            )
        self.index_file.write_text(index_text, encoding="utf-8")

        # ── 追加 log.md ────────────────────────────────────────────────────
        log_entry = plan.get("log_entry", f"## [{datetime.now().date()}] ingest | {source_name}")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{log_entry}\n")

        # ── 增量编译: 写入新哈希,下次相同内容不再编译 ──────────────────────
        state[state_key] = {
            "hash": content_hash,
            "last_ingest": datetime.now().isoformat(),
            "pages_created": created,
            "pages_updated": updated,
        }
        self._save_state(state)
        print(f"[WikiKB] 编译完成: {source_name} → {created} 新建, {updated} 更新 (哈希已记录)")

        return {"pages_created": created, "pages_updated": updated, "source": source_name}

    # ── Query ─────────────────────────────────────────────────────────────────

    async def _locate_pages(self, user_question: str) -> tuple[list[str], bool]:
        """
        三级降级定位页面。

        返回: (paths: list[str], used_llm: bool)
          - paths: 相对路径如 ["entities/xxx.md", "topics/yyy.md"]
          - used_llm: 是否动用了 LLM 读 index(True=第3级兜底,False=嵌入命中)

        第1/2级: WikiHybridRetriever 嵌入检索 (0 次 LLM)
        第3级:   LLM 读完整 index.md 选页 (1 次 LLM)
        """
        # ── 第1/2级: 嵌入检索 ──────────────────────────────────────────────
        if self.retriever is not None:
            page_keys = await self.retriever.search_async(user_question)
            if page_keys:
                # page_key 格式如 'entities/xxx' → 加 .md 即为相对路径
                paths = [f"{key}.md" for key in page_keys
                         if (self.wiki_dir / f"{key}.md").exists()]
                if paths:
                    return paths, False

        # ── 第3级: LLM 读 index(兜底) ────────────────────────────────────
        index = self.index_file.read_text(encoding="utf-8")
        find_resp = await self.llm.ainvoke([HumanMessage(content=
            f"从 Wiki 索引找出回答这个问题最相关的 1-3 个页面路径。\n\n"
            f"索引：\n{index}\n\n问题：{user_question}\n\n"
            f"只输出 JSON（不要代码块）：{{\"paths\": [\"entities/xxx.md\"]}}\n"
            f"没有相关内容则：{{\"paths\": []}}"
        )])
        raw = re.sub(r'```(?:json)?|```', '', find_resp.content).strip()
        try:
            paths = json.loads(raw).get("paths", [])
        except json.JSONDecodeError:
            paths = []
        return paths, True

    async def query(self, user_question: str, save_answer: bool = True) -> str:
        paths, _used_llm = await self._locate_pages(user_question)

        wiki_content = ""
        valid_paths  = []
        for rel_path in paths:
            fp = self.wiki_dir / rel_path
            if fp.exists():
                wiki_content += f"\n\n---\n### 来源：{rel_path}\n\n{fp.read_text(encoding='utf-8')}"
                valid_paths.append(rel_path)

        if not wiki_content:
            # 知识库未命中：可能是问候/闲聊，也可能是未覆盖的问题。
            # 让 LLM 礼貌回应，而不是直接甩兜底语。
            fallback_resp = await self.llm.ainvoke([
                SystemMessage(content=(
                    "你是友好专业的电商客服助手。当前问题在知识库中没有直接对应的内容。\n"
                    "请根据情况礼貌回应：\n"
                    "- 如果用户在打招呼或闲聊，热情问候，并简要说明你能提供的服务"
                    "（如商品咨询、订单查询、退换货政策、售后保修等）。\n"
                    "- 如果是具体业务问题但知识库暂无信息，先礼貌致歉，"
                    "说明可以为其转接人工客服或建议补充更多细节。\n"
                    "回答简洁自然，不要编造不存在的政策或商品信息。"
                )),
                HumanMessage(content=user_question),
            ])
            return fallback_resp.content

        answer_resp = await self.llm.ainvoke([
            SystemMessage(content="你是专业电商客服。严格基于 Wiki 内容回答，不编造信息，末尾注明引用来源。"),
            HumanMessage(content=f"Wiki内容：\n{wiki_content}\n\n用户问题：{user_question}"),
        ])
        answer = answer_resp.content

        if save_answer and valid_paths:
            await self._archive_answer(user_question, answer, valid_paths)

        return answer

    async def query_with_meta(self, user_question: str) -> dict:
        """
        评测专用：与 query() 同样的两步流程（定位页面 + 生成回答），
        但额外返回命中的页面路径与 token 用量，且不写入 qa_archive（避免污染知识库）。

        返回: {"answer": str, "sources": [stem...], "tokens": {...}, "llm_calls": int,
               "locate_method": "embedding" | "llm_index"}
        """
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        calls = 0

        def _acc(resp):
            nonlocal calls
            calls += 1
            meta = getattr(resp, "usage_metadata", None) or {}
            usage["input_tokens"]  += meta.get("input_tokens", 0)
            usage["output_tokens"] += meta.get("output_tokens", 0)
            usage["total_tokens"]  += meta.get("total_tokens", 0)

        # ── 定位阶段(三级降级) ──
        # 如果有 retriever,先走嵌入(0 LLM 调用);未命中则走 LLM 读 index(1 LLM 调用)
        locate_method = "embedding"
        if self.retriever is not None:
            page_keys = await self.retriever.search_async(user_question)
            if page_keys:
                paths = [f"{key}.md" for key in page_keys
                         if (self.wiki_dir / f"{key}.md").exists()]
            else:
                paths = None
        else:
            paths = None

        if not paths:
            # 第3级: LLM 读 index 兜底
            locate_method = "llm_index"
            index = self.index_file.read_text(encoding="utf-8")
            find_resp = await self.llm.ainvoke([HumanMessage(content=
                f"从 Wiki 索引找出回答这个问题最相关的 1-3 个页面路径。\n\n"
                f"索引：\n{index}\n\n问题：{user_question}\n\n"
                f"只输出 JSON（不要代码块）：{{\"paths\": [\"entities/xxx.md\"]}}\n"
                f"没有相关内容则：{{\"paths\": []}}"
            )])
            _acc(find_resp)
            raw = re.sub(r'```(?:json)?|```', '', find_resp.content).strip()
            try:
                paths = json.loads(raw).get("paths", [])
            except json.JSONDecodeError:
                paths = []

        wiki_content = ""
        valid_paths  = []
        for rel_path in paths:
            fp = self.wiki_dir / rel_path
            if fp.exists():
                wiki_content += f"\n\n---\n### 来源：{rel_path}\n\n{fp.read_text(encoding='utf-8')}"
                valid_paths.append(rel_path)

        if not wiki_content:
            return {"answer": "", "sources": [], "tokens": usage, "llm_calls": calls,
                    "locate_method": locate_method}

        answer_resp = await self.llm.ainvoke([
            SystemMessage(content="你是专业电商客服。严格基于 Wiki 内容回答，不编造信息，末尾注明引用来源。"),
            HumanMessage(content=f"Wiki内容：\n{wiki_content}\n\n用户问题：{user_question}"),
        ])
        _acc(answer_resp)

        stems = [Path(p).stem for p in valid_paths]
        return {"answer": answer_resp.content, "sources": stems, "tokens": usage,
                "llm_calls": calls, "locate_method": locate_method}

    async def _archive_answer(self, question: str, answer: str, sources: list[str]):
        date_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^\w]', '_', question[:30])
        fp = self.wiki_dir / "qa_archive" / f"{date_str}_{safe_name}.md"
        fp.write_text(
            f"---\ntype: qa\nquestion: {question}\ndate: {datetime.now().date()}\n---\n\n"
            f"## 问题\n{question}\n\n## 回答\n{answer}\n\n## 引用来源\n"
            + "\n".join(f"- [[{s}]]" for s in sources) + "\n",
            encoding="utf-8",
        )
        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(f"- [[qa_archive/{fp.name}]] — {question[:50]}\n")

    # ── Lint ──────────────────────────────────────────────────────────────────

    async def lint(self) -> dict:
        all_pages = [p for p in self.wiki_dir.glob("**/*.md")
                     if p.name not in ("index.md", "log.md")]
        random.shuffle(all_pages)
        issues = []
        for page_path in all_pages[:10]:
            resp = await self.llm.ainvoke([HumanMessage(content=
                f"检查 Wiki 页面质量（矛盾/孤立/过时/缺引用）。\n\n"
                f"路径：{page_path.relative_to(self.wiki_root)}\n"
                f"内容：\n{page_path.read_text(encoding='utf-8')[:1500]}\n\n"
                f"输出JSON（不要代码块）：\n"
                f'{{\"page\": \"路径\", \"issues\": [\"问题描述\"], \"severity\": \"high/medium/low\"}}'
            )])
            try:
                raw = re.sub(r'```(?:json)?|```', '', resp.content).strip()
                result = json.loads(raw)
                if result.get("issues"):
                    issues.append(result)
            except json.JSONDecodeError:
                pass
        return {"pages_checked": min(len(all_pages), 10),
                "total_pages": len(all_pages),
                "issues_found": len(issues), "issues": issues}

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_pages":   len(list(self.wiki_dir.glob("**/*.md"))),
            "entity_pages":  len(list((self.wiki_dir / "entities").glob("*.md"))),
            "topic_pages":   len(list((self.wiki_dir / "topics").glob("*.md"))),
            "qa_archived":   len(list((self.wiki_dir / "qa_archive").glob("*.md"))),
            "ingest_count":  self.log_file.read_text(encoding="utf-8").count("## ["),
            "index_size_kb": round(self.index_file.stat().st_size / 1024, 1),
        }