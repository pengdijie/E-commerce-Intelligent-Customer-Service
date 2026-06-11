#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 评测脚本 — 一键跑三组对比 + Reranker 测试
# 用法: bash scripts/run_eval.sh [--limit N]
# ─────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

LIMIT_ARG=""
for arg in "$@"; do
    case $arg in
        --limit) shift; LIMIT_ARG="--limit $1"; shift ;;
        --limit=*) LIMIT_ARG="--limit ${arg#*=}" ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║    三组方案对比评测                              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

echo "▶ [1/2] 纯RAG vs 纯LLM Wiki vs Wiki+RAG..."
echo ""
python -m eval.run_eval $LIMIT_ARG
echo ""

echo "▶ [2/2] Wiki+RAG 基线 vs +Reranker..."
echo ""
python -m eval.eval_reranker $LIMIT_ARG
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ 评测完成！报告已保存："
echo "  - eval/result.md          （三组方案对比）"
echo "  - eval/result_reranker.md （Reranker 对比）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
