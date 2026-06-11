"""
WikiHybridRetriever — Wiki 定位环节的混合检索（参考 llm-wiki-compiler 的 pickSearchSlugs 三级降级）

原 Wiki 定位：把整个 index.md 喂给 LLM 让它选页 —— 随 index 增长越来越贵，
是 llm-wiki-compiler 三级降级里最末级（最贵）的那一档。

本模块在它之上补齐前两级（嵌入检索，0 次 LLM 调用即可定位）：
  第1级  chunk 级嵌入检索（最高精度，命中页面内最相关分块）
  第2级  page 级嵌入检索（整页摘要召回）
  ——以上任一命中即返回，定位阶段不调用 LLM——
  第3级  交回调用方走 LLM 读 index（由 WikiknowledgeBase 兜底）

工程要点：
  - 复用 LongTermMemory._embed_text 做真实 embedding（text-embedding-3-small）
  - 向量已 L2 归一化，余弦相似度 = 点积，用 numpy 矩阵乘，45 页规模开销可忽略
  - 内容哈希增量：页面内容没变就不重新 embed，向量缓存落盘 .index_cache.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from memory.long_term import LongTermMemory


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class WikiHybridRetriever:
    """Wiki 页面定位的混合检索器。仅负责"定位"，不生成回答。"""

    def __init__(
        self,
        wiki_dir: str | Path,
        embedder: LongTermMemory | None = None,
        cache_path: str | Path | None = None,
        chunk_top_k: int = 5,
        page_top_k: int = 3,
        min_score: float = 0.30,
    ):
        self.wiki_dir = Path(wiki_dir)
        self.embedder = embedder or LongTermMemory()
        self.cache_path = Path(cache_path) if cache_path else self.wiki_dir / ".index_cache.json"
        self.chunk_top_k = chunk_top_k
        self.page_top_k = page_top_k
        self.min_score = min_score
        # 运行态：页向量 / chunk 向量及其归属 slug
        self._page_vecs: dict[str, np.ndarray] = {}      # slug -> vec
        self._chunk_vecs: list[tuple[str, np.ndarray]] = []  # (slug, vec)
        self._cache: dict[str, dict] = {}                # slug -> {hash, page_vec, chunks:[{hash,vec}]}
        self._loaded = False

    # ── 缓存读写 ────────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
        )

    # ── 页面枚举 ────────────────────────────────────────────────────────────

    def _iter_pages(self):
        """遍历 entities/ 与 topics/ 下的 .md（不含 index.md / log.md / qa_archive）。
        返回 (page_key, relative_path, content)。
        page_key = 'entities/xxx' 或 'topics/yyy'(去掉 .md 后缀,保证唯一)。"""
        for sub in ("entities", "topics"):
            d = self.wiki_dir / sub
            if not d.exists():
                continue
            for fp in sorted(d.glob("*.md")):
                try:
                    content = fp.read_text(encoding="utf-8")
                except OSError:
                    continue
                page_key = f"{sub}/{fp.stem}"
                yield page_key, f"{sub}/{fp.name}", content

    # ── 构建索引（增量，哈希门控）──────────────────────────────────────────

    def build_index(self) -> dict:
        """
        扫描 wiki pages,增量 embed 并缓存。返回统计。
        内容哈希没变的页面直接复用旧向量，不调 embedding API。
        """
        self._load_cache()
        stats = {"pages": 0, "chunks": 0, "embedded": 0, "cached": 0}
        new_page_vecs: dict[str, np.ndarray] = {}
        new_chunk_vecs: list[tuple[str, np.ndarray]] = []
        updated_cache: dict[str, dict] = {}

        for slug, rel_path, content in self._iter_pages():
            stats["pages"] += 1
            h = _content_hash(content)
            cached = self._cache.get(slug)

            if cached and cached.get("hash") == h and "page_vec" in cached:
                # 增量命中：复用旧向量
                stats["cached"] += 1
                page_vec = np.array(cached["page_vec"], dtype=np.float32)
                new_page_vecs[slug] = page_vec
                for chunk_entry in cached.get("chunks", []):
                    cv = np.array(chunk_entry["vec"], dtype=np.float32)
                    new_chunk_vecs.append((slug, cv))
                    stats["chunks"] += 1
                updated_cache[slug] = cached
            else:
                # 新页面或内容变更：重新 embed
                stats["embedded"] += 1
                # page 级向量：用前 512 字符的摘要 embed
                summary = content[:512]
                page_vec = self.embedder._embed_text(summary)
                new_page_vecs[slug] = page_vec

                # chunk 级向量
                chunks = LongTermMemory._chunk_text(content, chunk_size=512, overlap=128)
                chunk_entries = []
                for chunk_text in chunks:
                    cv = self.embedder._embed_text(chunk_text)
                    new_chunk_vecs.append((slug, cv))
                    chunk_entries.append({"vec": cv.tolist()})
                    stats["chunks"] += 1

                updated_cache[slug] = {
                    "hash": h,
                    "page_vec": page_vec.tolist(),
                    "chunks": chunk_entries,
                }

        self._page_vecs = new_page_vecs
        self._chunk_vecs = new_chunk_vecs
        self._cache = updated_cache
        self._save_cache()
        self._loaded = True

        print(f"[HybridRetriever] 索引构建完成: {stats['pages']} 页, "
              f"{stats['chunks']} chunks, 新 embed {stats['embedded']}, 缓存命中 {stats['cached']}")
        return stats

    # ── 三级降级检索 ─────────────────────────────────────────────────────────

    def search(self, query: str) -> list[str] | None:
        """
        三级降级检索,返回命中的 slug 列表(有序,最相关在前)。
        返回 None 表示嵌入检索全部未命中——调用方应 fallback 到 LLM 读 index。

        第1级: chunk 嵌入 → 聚合得分取 top page slugs
        第2级: page 嵌入 → cosine top-K
        第3级: 返回 None（由调用方走 LLM 兜底）
        """
        if not self._loaded:
            self.build_index()

        query_vec = self.embedder._embed_text(query)

        # ── 第1级: chunk 级检索 ──
        if self._chunk_vecs:
            chunk_mat = np.array([v for _, v in self._chunk_vecs], dtype=np.float32)
            scores = chunk_mat @ query_vec  # cosine(归一化向量)
            top_indices = np.argsort(scores)[::-1][:self.chunk_top_k]

            # 聚合: 对每个 slug 取其 chunk 最高分
            slug_best: dict[str, float] = {}
            for idx in top_indices:
                s = scores[idx]
                if s < self.min_score:
                    break
                slug = self._chunk_vecs[idx][0]
                if slug not in slug_best or s > slug_best[slug]:
                    slug_best[slug] = float(s)

            if slug_best:
                # 按分数降序,取 page_top_k 个
                ranked = sorted(slug_best.items(), key=lambda x: x[1], reverse=True)
                result = [slug for slug, _ in ranked[:self.page_top_k]]
                print(f"[HybridRetriever] 第1级(chunk)命中: {result} (scores: {[f'{s:.3f}' for _,s in ranked[:self.page_top_k]]})")
                return result

        # ── 第2级: page 级检索 ──
        if self._page_vecs:
            page_slugs = list(self._page_vecs.keys())
            page_mat = np.array([self._page_vecs[s] for s in page_slugs], dtype=np.float32)
            scores = page_mat @ query_vec
            top_indices = np.argsort(scores)[::-1][:self.page_top_k]

            result = []
            for idx in top_indices:
                if scores[idx] >= self.min_score:
                    result.append(page_slugs[idx])

            if result:
                print(f"[HybridRetriever] 第2级(page)命中: {result}")
                return result

        # ── 第3级: 返回 None,调用方走 LLM 兜底 ──
        print("[HybridRetriever] 嵌入检索未命中,降级到 LLM 读 index")
        return None

    async def search_async(self, query: str) -> list[str] | None:
        """异步版本(embedding 调用是同步的,包在 to_thread 里)"""
        import asyncio
        return await asyncio.to_thread(self.search, query)
