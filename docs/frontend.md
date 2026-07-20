# Next.js 前端重构说明

本项目已新增独立前端 `frontend/`，主技术栈为 Next.js + TypeScript + Tailwind CSS。前端只通过 FastAPI 调用后端，不直接访问 Python RAG 内部模块。

## 端口约定

你的 Langfuse 使用容器化部署并已占用 `localhost:3000`，因此前端默认运行在 `3001`。

- FastAPI: <http://localhost:8000>
- Next.js 前端: <http://localhost:3001>
- Langfuse: <http://localhost:3000> 或你的实际 Langfuse 地址

## 后端启动

```bash
python main.py serve
```

或：

```bash
uvicorn api.routes:app --reload --host 0.0.0.0 --port 8000
```

## 前端启动

```bash
cd frontend
npm install
npm run dev
```

访问：

```text
http://localhost:3001
```

前端环境变量示例见 `frontend/.env.local.example`：

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

不要在前端 `.env.local` 中放置 OpenAI、Langfuse secret 或任何后端密钥。

## Langfuse 配置

后端 `.env` 中配置：

```env
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_BASE_URL=http://localhost:3000
```

Langfuse 追踪 由 FastAPI 后端完成。前端只生成并传递：

- `session_id`
- `request_id`
- 可选 `user_id`

关键 trace name：

- `adaptive-rag.api.ask`
- `adaptive-rag.api.chat_stream`
- `adaptive-rag.api.documents_upload`
- `adaptive-rag.api.eval`

## 新增/增强 API

- `POST /ask`：非流式问答，保留并增加 `request_id` / `user_id`。
- `POST /chat/stream`：POST + `text/event-stream`，前端使用 `fetch + ReadableStream` 解析 SSE，不使用原生 EventSource。
- `POST /documents/upload`：浏览器 multipart 文件上传。
- `GET /sources`：知识库来源列表。
- `POST /eval`：三路对比评估。
- `GET /diagnostics`：Langfuse、性能和 token 状态摘要，不返回任何密钥。

## 流式说明

第一版 `/chat/stream` 是 LangGraph 工作流更新 流，不承诺 逐个 token。事件类型包括：

- `metadata`
- `state_update`
- `answer`
- `sources`
- `done`
- `error`

## 验证建议

```bash
python -m compileall api src
python main.py serve
```

然后访问：

```text
http://localhost:3001/chat
```
