"""
长期记忆 — 基于向量数据库的持久化记忆
存储用户画像、历史工单、知识库文档等需要持久化的信息。
支持语义相似度检索，用于RAG知识检索Agent。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LongTermMemory:
    """
    长期记忆：基于FAISS的向量检索。

    特点：
    - 向量化存储，支持语义相似度检索
    - 持久化到磁盘，跨会话保持
    - 支持增量更新和批量导入
    - 生产环境可切换为Milvus/Pinecone

    文档分块策略：
    - 固定长度分块 (512 tokens) + 重叠窗口 (128 tokens)
    - 按段落自然分割优先
    """

    def __init__(
        self,
        index_path: str = "./vector_store/faiss_index",
        embedding_dim: int | None = None,
        embedding_model: str | None = None,
    ):
        self.index_path = Path(index_path)
        # Embedding 配置优先读 EMBEDDING_* 环境变量,兼容旧 OPENAI_* 配置
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
        # 维度:bge-m3=1024, text-embedding-3-small=1536
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        elif "bge" in self.embedding_model:
            self.embedding_dim = 1024
        else:
            self.embedding_dim = 1536
        self._documents: list[dict[str, Any]] = []
        self._index = None
        # Embedding 独立端点(硅基流动等),fallback 到 LLM 端点
        embed_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        embed_base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self._sync_client = (
            OpenAI(
                api_key=embed_api_key,
                base_url=embed_base_url,
            )
            if OpenAI is not None and embed_api_key
            else None
        )
        self._init_index()

    def _init_index(self):
        """初始化FAISS索引"""
        if faiss is None:
            self._index = None
            return

        metadata_path = self.index_path.with_suffix(".meta.json")
        if self.index_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        self._documents = json.load(f)
            except Exception:
                self._index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)

    def _embed_text(self, text: str) -> np.ndarray:
        """调用 Embedding API 生成向量，并做 L2 归一化（配合 IndexFlatIP -> 余弦相似度）"""
        if self._sync_client is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        response = self._sync_client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float",
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-10
        return vec

    def _embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """批量 embedding，减少 API 往返次数"""
        if self._sync_client is None:
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._sync_client.embeddings.create(
                model=self.embedding_model,
                input=batch,
                encoding_format="float",
            )
            for item in response.data:
                vec = np.array(item.embedding, dtype=np.float32)
                vec /= np.linalg.norm(vec) + 1e-10
                all_vecs.append(vec)
        return np.array(all_vecs, dtype=np.float32)

    async def _embed_text_async(self, text: str) -> np.ndarray:
        """异步包装：把同步 embedding 调用丢到线程池，避免阻塞事件循环"""
        return await asyncio.to_thread(self._embed_text, text)

    def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        """添加文档到向量库"""
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

        doc = {
            "id": doc_id,
            "content": content,
            "source": source,
            "metadata": metadata or {},
        }
        self._documents.append(doc)

        if self._index is not None:
            embedding = self._embed_text(content)
            self._index.add(embedding.reshape(1, -1))

        return doc_id

    def add_documents_batch(self, documents: list[dict]) -> list[str]:
        """批量添加文档（使用批量 embedding，比逐条快很多）"""
        if not documents:
            return []
        doc_ids = []
        contents = []
        for doc in documents:
            content = doc.get("content", "")
            doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
            self._documents.append({
                "id": doc_id,
                "content": content,
                "source": doc.get("source", ""),
                "metadata": doc.get("metadata", {}),
            })
            doc_ids.append(doc_id)
            contents.append(content)

        if self._index is not None and contents:
            embeddings = self._embed_batch(contents)
            self._index.add(embeddings)

        return doc_ids

    def save_index(self):
        """持久化 FAISS 索引和文档元数据到磁盘"""
        if self._index is None or not self._documents:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        metadata_path = self.index_path.with_suffix(".meta.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义相似度检索"""
        if self._index is None or not self._documents:
            return self._fallback_search(query, top_k)

        query_vec = self._embed_text(query).reshape(1, -1)
        scores, indices = self._index.search(query_vec, min(top_k, len(self._documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

    async def search_async(self, query: str, top_k: int = 5) -> list[dict]:
        """search 的异步版本：embedding 走线程池，其余逻辑复用同步检索。"""
        if self._index is None or not self._documents:
            return self._fallback_search(query, top_k)

        query_vec = (await self._embed_text_async(query)).reshape(1, -1)
        scores, indices = self._index.search(query_vec, min(top_k, len(self._documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

    def _fallback_search(self, query: str, top_k: int) -> list[dict]:
        """当FAISS不可用时的关键词回退搜索"""
        scored = []
        query_terms = set(query.lower().split())

        for doc in self._documents:
            content_lower = doc["content"].lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def save(self):
        """持久化索引到磁盘"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))

        metadata_path = self.index_path.with_suffix(".meta.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)

    def load_knowledge_base(self, kb_dir: str) -> int:
        """从目录批量加载知识库文档"""
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            return 0

        count = 0
        for file_path in kb_path.glob("**/*.txt"):
            content = file_path.read_text(encoding="utf-8")
            chunks = self._chunk_text(content)
            for chunk in chunks:
                self.add_document(
                    content=chunk,
                    source=str(file_path.name),
                    metadata={"file": str(file_path)},
                )
                count += 1

        return count

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> list[str]:
        """
        文本分块：固定长度 + 重叠窗口。
        优先按段落分割，段落过长则按句子分割。
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    sentences = para.replace("。", "。\n").replace(".", ".\n").split("\n")
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        if len(current_chunk) + len(sentence) <= chunk_size:
                            current_chunk += sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:chunk_size]]
