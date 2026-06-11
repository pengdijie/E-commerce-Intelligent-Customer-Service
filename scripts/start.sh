#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 智能客服多Agent系统 — 一键启动脚本
# 用法: bash scripts/start.sh [--skip-redis] [--skip-ingest] [--dev]
# ─────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }
info() { echo -e "${BLUE}[→]${NC} $1"; }

SKIP_REDIS=false
SKIP_INGEST=false
DEV_MODE=false

for arg in "$@"; do
    case $arg in
        --skip-redis)  SKIP_REDIS=true ;;
        --skip-ingest) SKIP_INGEST=true ;;
        --dev)         DEV_MODE=true ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║    智能客服多Agent系统 — 启动中...               ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. 检查 .env ─────────────────────────────────────────
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        warn ".env 不存在，从 .env.example 复制..."
        cp .env.example .env
        warn "请编辑 .env 填入你的 API Key，然后重新运行此脚本"
        exit 1
    else
        err "未找到 .env 或 .env.example"
        exit 1
    fi
fi
log ".env 配置文件已就绪"

# ── 2. 检查 Python 依赖 ──────────────────────────────────
info "检查 Python 依赖..."
if ! python -c "import langchain, fastapi, faiss" 2>/dev/null; then
    info "安装依赖..."
    pip install -r requirements.txt -q
fi
log "Python 依赖已安装"

# ── 3. Redis（可选）──────────────────────────────────────
if [ "$SKIP_REDIS" = false ]; then
    if command -v redis-cli &>/dev/null && redis-cli ping &>/dev/null 2>&1; then
        log "Redis 已运行"
    else
        warn "Redis 未运行，短期记忆将使用内存存储（重启丢失对话历史）"
        warn "安装 Redis: sudo apt install redis-server 或 docker run -d -p 6379:6379 redis:7-alpine"
    fi
else
    info "跳过 Redis 检查（--skip-redis）"
fi

# ── 4. 知识库初始化 ──────────────────────────────────────
if [ "$SKIP_INGEST" = false ]; then
    WIKI_DIR="wiki_knowledge/wiki"
    ENTITY_COUNT=$(find "$WIKI_DIR/entities" -name "*.md" 2>/dev/null | wc -l)
    TOPIC_COUNT=$(find "$WIKI_DIR/topics" -name "*.md" 2>/dev/null | wc -l)
    TOTAL=$((ENTITY_COUNT + TOPIC_COUNT))

    if [ "$TOTAL" -gt 5 ]; then
        log "知识库已就绪（${TOTAL} 个页面：${ENTITY_COUNT} 实体 + ${TOPIC_COUNT} 主题）"
    else
        info "知识库为空或数据不足，运行 Demo 数据初始化..."
        python ingest_pipeline.py --demo
        log "知识库初始化完成"
    fi
else
    info "跳过知识库初始化（--skip-ingest）"
fi

# ── 5. 向量索引 ─────────────────────────────────────────
info "构建/验证向量索引..."
python -c "
from memory.wiki_knowledge_base import WikiknowledgeBase
from memory.long_term import LongTermMemory
wiki = WikiknowledgeBase()
stats = wiki.get_stats()
print(f'  知识库: {stats[\"total_pages\"]} 页面, {stats[\"entity_pages\"]} 实体, {stats[\"topic_pages\"]} 主题')
" 2>/dev/null || warn "向量索引构建跳过（首次查询时自动构建）"

# ── 6. 启动服务 ─────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "系统启动中..."
echo ""
info "聊天界面:  http://localhost:${PORT:-8000}"
info "API文档:   http://localhost:${PORT:-8000}/docs"
info "健康检查:  http://localhost:${PORT:-8000}/health"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$DEV_MODE" = true ]; then
    info "开发模式启动（热重载）"
    uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
else
    python -m api.main
fi
