# Adaptive RAG 文档问答系统

Adaptive RAG 是一个面向文档问答场景的自适应检索增强生成系统。项目用 LangGraph 编排完整问答链路，根据问题复杂度自动选择不检索、单步 RAG 或多步 RAG，并在答案生成后接入质量审核、RAGAS 评估、安全护栏、HITL 人工审核和语义缓存。

这个仓库适合作为 RAG 工程作品集展示，重点不是包装一个简单 Demo，而是展示从文档摄入、混合检索、动态路由、复杂问题召回、评估闭环到 Streamlit / FastAPI / CLI 多入口的完整工程实现。

## 当前能力

- **自适应路由**：先进行问题复杂度分类，再将问题路由到 `no_retrieval`、`single_step` 或 `multi_step`。
- **LangGraph 工作流**：使用 `StateGraph` 串联 `classify -> cache -> route -> retrieve -> generate -> review -> ragas -> guard -> hitl -> cache_store`。
- **混合检索**：单步 RAG 组合 BM25、向量检索、RRF 融合和 Cross-encoder 重排。
- **多步检索**：复杂问题支持 multi-query rewrite、step-back query、迭代检索、HyDE 兜底和证据覆盖保护。
- **范围隔离**：支持按 session / document 元数据过滤，避免跨文档误召回。
- **表格问答增强**：对预算、支出、剩余、合计等数值问题提供确定性表格聚合辅助，降低纯 LLM 归纳错误。
- **质量与安全闭环**：答案生成后经过 reviewer、RAGAS 在线评估、内容安全检查和 HITL 门禁。
- **语义缓存**：支持 exact cache 和 embedding cache，命中后可跳过后续检索与生成。
- **多入口运行**：提供 Streamlit UI、FastAPI 服务、CLI 单次问答、终端对话和三路评估命令。

## 系统流程

```text
用户问题
  -> 复杂度分类
  -> 语义缓存检查
  -> 自适应路由
      -> simple  -> no_retrieval
      -> medium  -> single_step RAG
      -> complex -> multi_step RAG
  -> LLM 生成答案
  -> 质量审核
  -> RAGAS 在线评估
  -> 内容安全检查
  -> HITL 人工审核门禁
  -> 缓存可复用答案
  -> 返回最终回答
```

单步 RAG：

```text
Query -> BM25 + Dense Search -> RRF Fusion -> Cross-encoder Rerank -> Context -> LLM
```

多步 RAG：

```text
Query
  -> Multi-query Rewrite / Step-back Query
  -> First-hop Retrieval
  -> Original-query Rerank
  -> HyDE Fallback
  -> Iterative Retrieve and Evaluate
  -> Evidence Coverage Guard
  -> Context -> LLM
```

## 目录结构

```text
adaptive-rag-system/
├── api/                 # FastAPI 路由
├── cli/                 # HITL 队列等 CLI 辅助脚本
├── config/              # pydantic-settings 配置和 YAML prompt
├── data/                # 小型样例数据；运行时数据库不提交
├── docs/                # 项目说明、评估报告和维护记录
├── src/
│   ├── agents/          # generator / reviewer
│   ├── cache/           # 语义缓存
│   ├── evaluation/      # RAGAS 与三路对比评估
│   ├── graph/           # LangGraph 工作流、状态和路由
│   ├── ingestion/       # 文档加载、分块和索引
│   ├── memory/          # 短期、中期、长期记忆
│   ├── models/          # LLM 和 Embedding 封装
│   ├── retrieval/       # no/single/multi/adaptive 检索策略
│   ├── safety/          # 内容安全与熔断器
│   └── utils/           # 日志、prompt、token、JSON 工具
├── test_data/           # StarVault 基准文档、问题和结果
├── tests/               # 路由、检索、RAGAS、HITL、安全等测试
├── ui/                  # Streamlit 应用
├── main.py              # 统一入口
└── requirements.txt
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
```

编辑 `.env`，至少配置：

```env
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_DEFAULT_MODEL=gpt-4o-mini
LLM_COMPLEX_MODEL=gpt-4o

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANK_ENABLED=true
```

启动 Streamlit：

```bash
python main.py ui
```

命令行问答：

```bash
python main.py ask "天枢项目的预算是多少？"
```

摄入文档：

```bash
python main.py ingest test_data/starvault_report.md
```

三路对比评估：

```bash
python main.py eval "公司总预算和总支出分别是多少？"
```

运行测试：

```bash
python -m pytest tests/ -v
```

## 配置说明

项目使用 OpenAI 兼容 Chat Completion API，可以接入 OpenAI、DeepSeek、Ollama、vLLM、Azure OpenAI 或其他兼容服务。

常用配置项：

| 配置 | 作用 |
| --- | --- |
| `LLM_API_KEY` / `LLM_BASE_URL` | 聊天模型凭证和地址 |
| `LLM_SIMPLE_MODEL` / `LLM_MEDIUM_MODEL` / `LLM_COMPLEX_MODEL` | 不同复杂度问题使用的模型 |
| `EMBEDDING_PROVIDER` | `local` 或 `openai` |
| `RERANK_ENABLED` | 是否启用 Cross-encoder 重排 |
| `CHROMA_PERSIST_DIR` | Chroma 向量库目录 |
| `RAGAS_ONLINE_ENABLED` | 是否启用在线 RAGAS 评估 |
| `HITL_ENABLED` | 是否启用人工审核门禁 |

不要提交 `.env`、本地向量库、SQLite 数据库、模型缓存、运行日志或人工审核队列。

## 基准测试

仓库内置 StarVault 小型基准，用一份内部项目报告和 8 个问题覆盖不同问答能力：

| ID | 能力类型 | 关注点 |
| --- | --- | --- |
| Q1 / Q2 | 单点事实 | 预算、技术栈等直接事实 |
| Q3 | 列表聚合 | 列出项目和部门 |
| Q4 | 数值聚合 | 总预算、总支出、剩余预算 |
| Q5 | 隐含推断 | 根据技能、项目状态和时间线推荐候选人 |
| Q6 | 战略推断 | 关联收入来源和战略约束 |
| Q7 | 矛盾识别 | 识别时间线和状态不一致 |
| Q8 | 不可回答边界 | 不编造融资和估值信息 |

当前 `test_data/benchmark_summary_multiquery.csv` 记录了 multi-query 版本的对比结果。结果显示：

- Q1-Q4 路由到 `medium / single_step`。
- Q5-Q8 路由到 `complex / multi_step`。
- Q2、Q7、Q8 的 Adaptive RAG 指标相对 Standard RAG 有明显改善。
- Q5/Q6 这类隐含推断问题不能只看默认 RAGAS 分数，需要结合证据覆盖和人工 rubric。

更详细的评估解释见 [docs/evaluation_report.md](docs/evaluation_report.md)。

## 为什么这个项目有展示价值

这个项目能体现几个真实 RAG 工程问题：

- 简单事实题不应无条件走昂贵的多步检索。
- 表格数值题不能只依赖 LLM 读上下文后自由归纳。
- 隐含推断题需要召回人员表、项目状态、时间线等多类证据。
- 不可回答问题需要明确拒答，而不是为了相关性编造答案。
- RAGAS 是有用信号，但不能替代任务型校验器和人工评估。
- RAG 系统需要工程化护栏，包括缓存、熔断、HITL、日志和测试。

## 已知限制

- 首次运行本地 embedding / reranker 可能需要下载 Hugging Face 模型。
- 完整 benchmark 会调用 LLM 和 RAGAS，耗时和费用取决于模型服务。
- RAGAS 对隐含推断和不可回答问题存在局限，需结合任务型校验。
- 当前项目适合学习、作品集和单机演示，不是生产级多租户知识库。

## 更多文档

- [docs/project-overview.md](docs/project-overview.md)：当前项目工程说明
- [docs/evaluation_report.md](docs/evaluation_report.md)：StarVault 基准评估说明
- [docs/performance-optimization-report.md](docs/performance-optimization-report.md)：性能优化记录
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md)：已知维护项
