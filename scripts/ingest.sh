#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 数据摄入脚本 — 导入各种数据集到知识库
# 用法:
#   bash scripts/ingest.sh --demo              # 内置示例数据
#   bash scripts/ingest.sh --all               # 导入所有可用数据集
#   bash scripts/ingest.sh --source FILE       # 指定数据文件
# ─────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
info() { echo -e "${BLUE}[→]${NC} $1"; }

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║    知识库数据摄入                                ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

MODE=""
SOURCE=""

for arg in "$@"; do
    case $arg in
        --demo) MODE="demo" ;;
        --all)  MODE="all" ;;
        --source) shift; SOURCE="$1" ;;
        --source=*) SOURCE="${arg#*=}" ;;
    esac
done

case $MODE in
    demo)
        info "导入内置 Demo 数据..."
        python ingest_pipeline.py --demo
        ;;
    all)
        info "导入所有可用数据集..."

        # E-commerce 评论数据
        if [ -f "online_shopping_10_cats.csv" ]; then
            info "导入 online_shopping_10_cats.csv（商品评论）..."
            python ingest_pipeline.py --source ./online_shopping_10_cats.csv --type review --limit 100
            log "评论数据导入完成"
        fi

        # JDDC 客服对话
        if [ -f "JDDC/chat.txt" ]; then
            info "导入 JDDC/chat.txt（客服对话）..."
            python ingest_pipeline.py --source ./JDDC/chat.txt --type dialogue --limit 50
            log "对话数据导入完成"
        fi

        # 兜底：如果没有找到数据集，跑 demo
        if [ ! -f "online_shopping_10_cats.csv" ] && [ ! -f "JDDC/chat.txt" ]; then
            info "未找到外部数据集，使用 Demo 数据..."
            python ingest_pipeline.py --demo
        fi
        ;;
    *)
        if [ -n "$SOURCE" ]; then
            info "导入: $SOURCE"
            python ingest_pipeline.py --source "$SOURCE"
        else
            echo "用法:"
            echo "  bash scripts/ingest.sh --demo          # 内置示例"
            echo "  bash scripts/ingest.sh --all           # 导入所有数据集"
            echo "  bash scripts/ingest.sh --source FILE   # 指定文件"
            exit 1
        fi
        ;;
esac

echo ""
info "知识库统计："
python -c "
from memory.wiki_knowledge_base import WikiknowledgeBase
wiki = WikiknowledgeBase()
stats = wiki.get_stats()
for k, v in stats.items():
    print(f'  {k}: {v}')
"
echo ""
log "数据摄入完成！"
