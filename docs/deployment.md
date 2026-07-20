# 部署说明

本文说明如何使用 Docker Desktop 或本地 Python/Node 环境运行 Adaptive RAG 系统。

## API Key 准备

部署者需要自己准备并填写 `.env`。仓库只提交 `.env.example`，不提交真实密钥。

| 配置项 | 何时需要 | 用途 |
| --- | --- | --- |
| `LLM_API_KEY` | 必需 | 调用聊天模型 |
| `LLM_BASE_URL` | 必需 | OpenAI 兼容聊天补全接口地址 |
| `EMBEDDING_API_KEY` 或 `DASHSCOPE_API_KEY` | 使用云端向量模型时必需 | 文档向量化 |
| `RERANK_API_KEY` 或 `DASHSCOPE_API_KEY` | 启用重排时必需 | 检索结果精排 |
| `RAGAS_EVAL_API_KEY` | 启用 RAGAS 在线评估时必需 | 调用评估模型 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 启用追踪时必需 | Langfuse 可观测性 |

## Docker Desktop 部署

```bash
copy .env.example .env
```

编辑 `.env`，填入你自己的服务凭据。然后运行：

```bash
docker compose up --build
```

`docker-compose.yml` 会先读取 `.env.example`，再在 `.env` 存在时读取 `.env`。这样全新克隆后可以运行 `docker compose config`，本地真实密钥也能覆盖模板占位值。

访问地址：

- 前端：<http://localhost:3001>
- 后端健康检查：<http://localhost:8000/health>
- 后端接口文档：<http://localhost:8000/docs>

停止服务：

```bash
docker compose down
```

如果需要清理本地向量库和运行时数据：

```bash
docker compose down -v
```

## 本地开发部署

后端：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python main.py serve
```

前端：

```bash
cd frontend
npm ci
npm run dev
```

## 常见问题

- 没有填写 API key 时，健康检查可以通过，但真实问答会在调用模型时失败。
- `LANGFUSE_ENABLED=false` 时不会写入 Langfuse 追踪，这适合最小演示。
- `RAGAS_ONLINE_ENABLED=false` 时评估页面仍可跑对比，但不会强依赖在线 RAGAS 评分。
- 如果端口 `3001` 或 `8000` 被占用，修改 compose 端口映射或本地启动命令。
