# 智能客服多Agent系统 — 运行指南

## 快速启动（一键脚本）

```bash
# 在 python-impl/ 目录下执行
bash scripts/start.sh
```

如果你想手动一步步来，往下看。

---

## 目录结构

```
python-impl/
├── agents/              # 5个Agent（Supervisor/意图/知识/工单/合规）
├── memory/              # 三层记忆（工作/短期Redis/长期FAISS）
├── api/main.py          # FastAPI 服务入口
├── static/index.html    # 前端聊天UI
├── mcp/                 # MCP工具服务（订单/退款/工单/风控）
├── eval/                # 三组评测脚本 + 结果
├── wiki_knowledge/      # LLM Wiki 编译产物
├── tracing/             # OpenTelemetry 链路追踪
├── ingest_pipeline.py   # 数据摄入管线
├── Dockerfile           # 容器化部署
└── scripts/             # 启动/停止/数据初始化脚本
```

---

## 环境准备

### 1. Python 依赖

```bash
# 推荐 Python 3.11+
pip install -r requirements.txt
```

### 2. 环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

必需的配置项：

| 变量 | 说明 | 示例 |
|------|------|------|
| `OPENAI_API_KEY` | LLM API Key | `sk-xxx` |
| `OPENAI_BASE_URL` | LLM API 端点 | `https://api.deepseek.com` |
| `MODEL_NAME` | 模型名称 | `deepseek-chat` |
| `EMBEDDING_API_KEY` | Embedding API Key | `sk-xxx` |
| `EMBEDDING_BASE_URL` | Embedding 端点 | `https://api.siliconflow.cn/v1` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-m3` |

可选配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `REDIS_URL` | Redis 连接地址 | `redis://localhost:6379/0` |
| `FAISS_INDEX_PATH` | 向量索引存储路径 | `./vector_store/faiss_index` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 链路追踪收集器 | `http://localhost:4317` |
| `HOST` | 服务监听地址 | `0.0.0.0` |
| `PORT` | 服务监听端口 | `8000` |

### 3. Redis（可选）

Redis 用于短期记忆（多轮对话上下文缓存）。**不装也能跑**，系统会自动降级为内存存储。

```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis

# macOS
brew install redis && brew services start redis

# Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

验证：
```bash
redis-cli ping
# 应返回 PONG
```

### 4. OpenTelemetry（可选）

用于全链路追踪可视化。不装也能跑，只是没有追踪数据。

```bash
# 用 Jaeger 做可视化
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

追踪UI访问：`http://localhost:16686`

---

## 知识库初始化

### 方式一：使用内置 Demo 数据（推荐首次体验）

```bash
python ingest_pipeline.py --demo
```

这会用内置的电商示例数据编译出 Wiki 知识页面。

### 方式二：导入 E-commerce 数据集

```bash
# 导入商品评论（CSV）
python ingest_pipeline.py --source ./online_shopping_10_cats.csv --type review --limit 100

# 导入 JDDC 客服对话
python ingest_pipeline.py --source ./JDDC/chat.txt --type dialogue --limit 50
```

### 方式三：导入自定义数据

```bash
# 任意 CSV（含 review/text/comment 列）
python ingest_pipeline.py --source your_data.csv --type review

# 纯文本
python ingest_pipeline.py --source knowledge.txt --type raw
```

### 验证知识库状态

```bash
python -c "
from memory.wiki_knowledge_base import WikiknowledgeBase
wiki = WikiknowledgeBase()
stats = wiki.get_stats()
for k, v in stats.items():
    print(f'  {k}: {v}')
"
```

期望输出：
```
  total_pages: 75
  entity_pages: 37
  topic_pages: 36
  qa_archived: 2
  ingest_count: 5
  index_size_kb: 3.2
```

---

## 启动服务

### 开发模式（带热重载）

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 生产模式

```bash
python -m api.main
```

### Docker 部署

```bash
docker build -t smart-cs-agent .
docker run -d \
  --name smart-cs \
  -p 8000:8000 \
  --env-file .env \
  smart-cs-agent
```

---

## 访问系统

启动后打开浏览器：

- **聊天UI**：`http://localhost:8000`
- **健康检查**：`http://localhost:8000/health`
- **API文档**：`http://localhost:8000/docs`（Swagger UI）
- **指标**：`http://localhost:8000/api/metrics`

### 试用示例问题

在聊天界面输入：
- "耳机右耳没声音了怎么办？"
- "便携充电宝保修多长时间？"
- "我想退货，运费谁承担？"
- "帮我查一下订单状态"

---

## 运行评测

### 三组方案对比（纯RAG vs 纯Wiki vs Wiki+RAG）

```bash
python -m eval.run_eval
# 或限制题数快速验证
python -m eval.run_eval --limit 10
```

测试集共 100 题（覆盖 8 个分类：售后政策、产品说明书、物流配送、订单管理、操作手册、故障排查、促销活动、复杂故障排查），结果保存在 `eval/result.md`。

### Reranker 对比测试

```bash
python -m eval.eval_reranker
# 结果保存在 eval/result_reranker.md
```

---

## 架构说明

```
用户请求
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI + SSE 流式响应 (api/main.py)                    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Supervisor Agent — LangGraph StateGraph 编排             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ 意图路由  │  │  知识RAG查询  │  │   工单处理      │    │
│  └──────────┘  └──────────────┘  └────────────────┘    │
│                       │               ┌────────────────┐│
│                       │               │   合规审查      ││
│                       │               └────────────────┘│
└───────────────────────┼─────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
  ┌──────────────┐ ┌────────┐ ┌───────────┐
  │ FAISS 向量索引│ │  Redis │ │ MCP Tools │
  │ (长期记忆)    │ │(短期)  │ │(订单/退款) │
  └──────────────┘ └────────┘ └───────────┘
          ▲
          │ 离线编译
  ┌──────────────────┐
  │  LLM Wiki 编译层  │
  │ (ingest_pipeline) │
  └──────────────────┘
```

### 知识分层

| 层级 | 组件 | 职责 | 时机 |
|------|------|------|------|
| 编译层 | LLM Wiki | 原始数据→结构化Markdown | 离线批处理 |
| 检索层 | FAISS + BGE-M3 | 向量检索 top-k chunks | 在线实时 |
| 生成层 | DeepSeek LLM | 基于 context 生成回答 | 在线实时 |
| 工具层 | MCP Tools | 查订单/退款等实时数据 | 按需调用 |
| 兜底层 | 人工客服 | 高风险/合规不通过 | 升级转接 |

---

## 常见问题

### Q: Redis 没装能跑吗？
可以。短期记忆会自动降级为进程内字典，功能不受影响，只是重启后会丢失对话历史。

### Q: OpenTelemetry 没配会报错吗？
不会。追踪模块检测不到 OTLP collector 会静默降级为 NoOp，不影响主流程。

### Q: 知识库为空怎么办？
先跑 `python ingest_pipeline.py --demo` 初始化。或者如果 `wiki_knowledge/wiki/` 下已有 .md 文件，系统启动时会自动加载。

### Q: API Key 用什么服务？
项目默认配置使用 DeepSeek（LLM）+ SiliconFlow（Embedding），都是国内可直连的服务商，注册即有免费额度。

### Q: 如何切换 LLM 模型？
修改 `.env` 中的 `MODEL_NAME`、`OPENAI_BASE_URL`、`OPENAI_API_KEY` 即可。兼容所有 OpenAI API 格式的服务（DeepSeek、通义千问、Moonshot 等）。
