"""
Wiki+RAG 方案内对比：无 Reranker vs 加 BGE-Reranker-v2-m3

验证轻量 cross-encoder reranker 对答案精度的提升，且不增加 LLM 调用。

Reranker 方案流程：
  1. 用户问题 → Embedding → FAISS top-10（粗检索，召回更多候选）
  2. top-10 chunks → BGE-Reranker-v2-m3 重排（cross-encoder，0 LLM）
  3. 取 rerank 后的 top-3 → 注入 context → 1 次 LLM 生成

对比基线（无 rerank）：
  1. 用户问题 → Embedding → FAISS top-5
  2. 取 top-3 → 注入 context → 1 次 LLM 生成

用法:
  python -m eval.eval_reranker              # 全量 44 题
  python -m eval.eval_reranker --limit 5    # 前 5 题验证
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from memory.long_term import LongTermMemory

ROOT = Path(__file__).resolve().parent.parent
WIKI_PAGES_DIR = ROOT / "wiki_knowledge" / "wiki"
TESTSET = Path(__file__).resolve().parent / "testset.jsonl"
FAISS_INDEX = "/tmp/eval_faiss/reranker_test"

# Reranker 配置
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_URL = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/") + "/rerank"
RERANKER_KEY = os.getenv("EMBEDDING_API_KEY", "")


def load_testset(limit=None):
    rows = []
    with open(TESTSET, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    return rows[:limit] if limit else rows


def build_shared_faiss():
    ltm = LongTermMemory(index_path=FAISS_INDEX)
    import faiss
    ltm._index = faiss.IndexFlatIP(ltm.embedding_dim)
    ltm._documents = []
    page_count = 0
    for sub in ("topics", "entities"):
        for fp in sorted((WIKI_PAGES_DIR / sub).glob("*.md")):
            content = fp.read_text(encoding="utf-8").strip()
            if not content:
                continue
            for chunk in LongTermMemory._chunk_text(content):
                ltm.add_document(content=chunk, source=fp.stem,
                                 metadata={"page": f"{sub}/{fp.name}"})
            page_count += 1
    return ltm, page_count


def _make_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o"), temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def _empty_usage():
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _acc(usage, resp):
    meta = getattr(resp, "usage_metadata", None) or {}
    usage["input_tokens"] += meta.get("input_tokens", 0)
    usage["output_tokens"] += meta.get("output_tokens", 0)
    usage["total_tokens"] += meta.get("total_tokens", 0)


# ── Reranker 调用 ─────────────────────────────────────────────────────────────
async def rerank(query: str, documents: list[str], top_n: int = 3) -> list[tuple[int, float]]:
    """调用 BGE-Reranker-v2-m3，返回 [(原始index, score), ...] 按分数降序。"""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            RERANKER_URL,
            headers={"Authorization": f"Bearer {RERANKER_KEY}"},
            json={
                "model": RERANKER_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    results = sorted(data["results"], key=lambda x: x["relevance_score"], reverse=True)
    return [(r["index"], r["relevance_score"]) for r in results[:top_n]]


# ── 方案1: Wiki+RAG 无 Reranker（基线）──────────────────────────────────────────
HYBRID_SYS = ("你是专业的电商客服。严格基于检索到的文档内容回答，不编造信息，"
              "回答简洁专业，末尾标注引用来源。若无相关信息，告知并建议转人工。")


async def run_baseline(llm, ltm, question, top_k=5, context_k=3):
    """基线：embed检索 top-5 → 取 top-3 → 1次LLM生成"""
    usage = _empty_usage()
    docs = await ltm.search_async(question, top_k=top_k)
    selected = docs[:context_k]
    if not selected:
        return {"answer": "未找到相关信息。", "sources": [], "tokens": usage, "llm_calls": 0}
    ctx = "\n\n---\n\n".join(f"来源: {d.get('source','')}\n内容: {d.get('content','')}" for d in selected)
    ans = await llm.ainvoke([
        SystemMessage(content=HYBRID_SYS),
        HumanMessage(content=f"用户问题: {question}\n\n参考文档:\n{ctx}"),
    ])
    _acc(usage, ans)
    sources = list(dict.fromkeys(d.get("source", "") for d in selected))
    return {"answer": ans.content, "sources": sources, "tokens": usage, "llm_calls": 1}


# ── 方案2: Wiki+RAG + Reranker ────────────────────────────────────────────────
async def run_with_reranker(llm, ltm, question, top_k=10, context_k=3):
    """加 Reranker：embed粗检索 top-10 → reranker精排取 top-3 → 1次LLM生成"""
    usage = _empty_usage()
    # 粗检索：扩大召回范围到 top-10
    docs = await ltm.search_async(question, top_k=top_k)
    if not docs:
        return {"answer": "未找到相关信息。", "sources": [], "tokens": usage,
                "llm_calls": 0, "reranker_tokens": 0}

    # Reranker 精排（cross-encoder，0 LLM调用）
    doc_texts = [d.get("content", "") for d in docs]
    ranked = await rerank(question, doc_texts, top_n=context_k)
    selected = [docs[idx] for idx, _ in ranked]

    if not selected:
        return {"answer": "未找到相关信息。", "sources": [], "tokens": usage,
                "llm_calls": 0, "reranker_tokens": 0}

    # 1次LLM生成
    ctx = "\n\n---\n\n".join(f"来源: {d.get('source','')}\n内容: {d.get('content','')}" for d in selected)
    ans = await llm.ainvoke([
        SystemMessage(content=HYBRID_SYS),
        HumanMessage(content=f"用户问题: {question}\n\n参考文档:\n{ctx}"),
    ])
    _acc(usage, ans)
    sources = list(dict.fromkeys(d.get("source", "") for d in selected))
    return {"answer": ans.content, "sources": sources, "tokens": usage,
            "llm_calls": 1, "reranker_tokens": sum(len(t) for t in doc_texts) // 4}


# ── Judge ─────────────────────────────────────────────────────────────────────
JUDGE = ("你是严格的客服回答质量评审。对照参考答案，从准确性、完整性、是否答到点上，"
         "给候选回答打 1-5 分。只输出JSON：{\"score\": <1-5>, \"reason\": \"<理由>\"}")


async def judge_answer(llm, question, reference, answer, usage):
    if not answer.strip():
        return 0
    r = await llm.ainvoke([
        SystemMessage(content=JUDGE),
        HumanMessage(content=f"问题：{question}\n\n参考答案：{reference}\n\n候选回答：{answer}"),
    ])
    _acc(usage, r)
    raw = re.sub(r'```(?:json)?|```', '', r.content).strip()
    try:
        return int(json.loads(raw).get("score", 0))
    except (json.JSONDecodeError, ValueError, TypeError):
        m = re.search(r'[1-5]', raw)
        return int(m.group()) if m else 0


def hit(predicted, gold):
    return any(p in set(gold) for p in predicted)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    testset = load_testset(args.limit)
    llm = _make_llm()

    print(f"[1/3] 构建 FAISS 索引...")
    ltm, n_pages = build_shared_faiss()
    print(f"      {n_pages} 页面, {len(ltm._documents)} 分块\n")

    base_rows, rerank_rows = [], []
    judge_usage = _empty_usage()

    print(f"[2/3] 逐题评测（{len(testset)} 题 × 2 方案）...\n")
    print(f"  {'#':>3} {'问题':<22} | {'基线tok':>7} {'基线分':>4} {'基线hit':>5} | {'Rerank tok':>9} {'Rerank分':>6} {'Rerank hit':>8}")
    print(f"  {'─'*3} {'─'*22} | {'─'*7} {'─'*4} {'─'*5} | {'─'*9} {'─'*6} {'─'*8}")

    for row in testset:
        q, ref, gold = row["question"], row["reference"], row["sources"]

        base_r, rerank_r = await asyncio.gather(
            run_baseline(llm, ltm, q),
            run_with_reranker(llm, ltm, q),
        )

        bs, rs = await asyncio.gather(
            judge_answer(llm, q, ref, base_r["answer"], judge_usage),
            judge_answer(llm, q, ref, rerank_r["answer"], judge_usage),
        )

        b_hit = hit(base_r["sources"], gold)
        r_hit = hit(rerank_r["sources"], gold)

        base_rows.append({**base_r, "hit": b_hit, "score": bs, "id": row["id"], "q": q})
        rerank_rows.append({**rerank_r, "hit": r_hit, "score": rs, "id": row["id"], "q": q})

        print(f"  Q{row['id']:>2} {q[:20]:<20} | "
              f"{base_r['tokens']['total_tokens']:>5}t  s={bs}  {'✓' if b_hit else '✗'} | "
              f"{rerank_r['tokens']['total_tokens']:>7}t   s={rs}   {'✓' if r_hit else '✗'}")

    print(f"\n[3/3] 生成报告...\n")
    report = render_report(base_rows, rerank_rows, judge_usage, n_pages, len(ltm._documents))
    print(report)

    out = Path(__file__).resolve().parent / "result_reranker.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n报告已保存: {out}")


def render_report(base_rows, rerank_rows, judge_usage, n_pages, n_chunks):
    n = len(base_rows)
    if not n:
        return "无数据"

    def stats(rows, label):
        return {
            "label": label,
            "tot_tok": sum(r["tokens"]["total_tokens"] for r in rows),
            "avg_tok": sum(r["tokens"]["total_tokens"] for r in rows) / n,
            "avg_in": sum(r["tokens"]["input_tokens"] for r in rows) / n,
            "avg_out": sum(r["tokens"]["output_tokens"] for r in rows) / n,
            "hit_rate": sum(1 for r in rows if r["hit"]) / n,
            "avg_score": sum(r["score"] for r in rows) / n,
            "avg_calls": sum(r["llm_calls"] for r in rows) / n,
        }

    base = stats(base_rows, "Wiki+RAG (基线)")
    rr = stats(rerank_rows, "Wiki+RAG+Reranker")

    score_delta = rr["avg_score"] - base["avg_score"]
    hit_delta = (rr["hit_rate"] - base["hit_rate"]) * 100

    lines = [
        "# Wiki+RAG Reranker 对比评测",
        "",
        f"- 测试集：{n} 题",
        f"- 知识库：{n_pages} 页面，{n_chunks} 分块",
        f"- Reranker：BAAI/bge-reranker-v2-m3（cross-encoder，0 LLM调用）",
        "",
        "## 方案差异",
        "",
        "| | Wiki+RAG (基线) | Wiki+RAG + Reranker |",
        "|--|----------------|---------------------|",
        "| 粗检索 | FAISS top-5 | FAISS top-10（扩大召回） |",
        "| 精排 | 无 | BGE-Reranker cross-encoder |",
        "| Context | top-3 by embedding score | top-3 by reranker score |",
        "| LLM调用 | 1次 | 1次（不变） |",
        "| 额外成本 | 无 | reranker API（极低） |",
        "",
        "## 量化结果",
        "",
        "| 指标 | 基线 | +Reranker | 变化 |",
        "|------|------|-----------|------|",
        f"| LLM调用次数 | {base['avg_calls']:.1f} | {rr['avg_calls']:.1f} | 不变 |",
        f"| 平均 Token/题 | {base['avg_tok']:.0f} | {rr['avg_tok']:.0f} | {rr['avg_tok']-base['avg_tok']:+.0f} |",
        f"| ├ input | {base['avg_in']:.0f} | {rr['avg_in']:.0f} | |",
        f"| └ output | {base['avg_out']:.0f} | {rr['avg_out']:.0f} | |",
        f"| 检索命中率 | {base['hit_rate']*100:.1f}% | {rr['hit_rate']*100:.1f}% | {hit_delta:+.1f}pp |",
        f"| **答案精度(1-5)** | **{base['avg_score']:.2f}** | **{rr['avg_score']:.2f}** | **{score_delta:+.2f}** |",
        "",
        f"> Judge 消耗 {judge_usage['total_tokens']} token。Reranker token 成本约为 LLM 的 1/50。",
        "",
        "## 结论",
        "",
    ]

    if score_delta > 0.1:
        lines.append(f"Reranker 将答案精度提升 **{score_delta:+.2f}** 分（{base['avg_score']:.2f}→{rr['avg_score']:.2f}），"
                     f"命中率提升 {hit_delta:+.1f}pp，LLM token 几乎不变。")
        lines.append("cross-encoder 的语义理解优于 bi-encoder 的向量点积，"
                     "能更准确地判断 chunk 与 query 的相关性。")
    elif score_delta > 0:
        lines.append(f"Reranker 小幅提升精度 {score_delta:+.2f} 分，命中率变化 {hit_delta:+.1f}pp。")
        lines.append("提升有限说明 embedding 质量已足够好，reranker 主要在边界 case 上有优势。")
    else:
        lines.append(f"Reranker 对精度无明显提升（{score_delta:+.2f}），说明当前 embedding 已足够。")

    lines += [
        "",
        "## 逐题对比",
        "",
        "| # | 问题 | 基线tok | 基线分 | 基线hit | Rerank tok | Rerank分 | Rerank hit |",
        "|---|------|---------|-------|---------|-----------|---------|-----------|",
    ]
    for b, r in zip(base_rows, rerank_rows):
        delta = "↑" if r["score"] > b["score"] else ("↓" if r["score"] < b["score"] else "=")
        lines.append(
            f"| {b['id']} | {b['q'][:16]} | "
            f"{b['tokens']['total_tokens']} | {b['score']} | {'✓' if b['hit'] else '✗'} | "
            f"{r['tokens']['total_tokens']} | {r['score']}{delta} | {'✓' if r['hit'] else '✗'} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
