# Adaptive RAG 文档问答系统

这是一个基于 LangGraph 的自适应 RAG 文档问答系统。项目重点展示从文档解析、混合检索、复杂度路由、多步检索、生成评估到 Streamlit 交互界面的完整 RAG 工程能力。

## 项目亮点

- **自适应路由**：根据问题复杂度自动选择 `no_retrieval`、`single_step` 或 `multi_step`
- **混合检索**：BM25 + 向量检索 + RRF 融合 + Cross-encoder 重排序
- **多步检索**：HyDE 改写、迭代检索、隐含推断问题的证据覆盖保护
- **文档范围隔离**：支持按 session / document 过滤，只检索当前上传文档
- **表格问答增强**：对预算、支出、剩余等表格数值题使用确定性计算兜底
- **评估体系**：支持 Direct / Standard RAG / Adaptive RAG 三路对比，并集成 RAGAS 与任务型校验
- **交互界面**：Streamlit 支持文档上传、处理、问答和流式输出
- **测试覆盖**：当前全量测试通过，`86 passed`

## 系统架构

```text
用户问题
  -> 输入安全检测
  -> 问题复杂度分类
      -> simple  -> 不检索，直接生成
      -> medium  -> Single-step RAG
      -> complex -> Multi-step RAG
  -> 答案审查 / 输出安全检测
  -> 可选 RAGAS 评估
  -> 返回答案
```

Single-step RAG：

```text
Query -> BM25 + Dense -> RRF 融合 -> Cross-encoder 重排 -> Context -> LLM
```

Multi-step RAG：

```text
Query -> HyDE -> 迭代检索 -> 证据覆盖保护 -> Context -> LLM
```

## 目录结构

```text
adaptive-rag-system/
├── api/                 # API 入口
├── cli/                 # CLI 辅助脚本
├── config/              # 配置和 Prompt 模板
├── docs/                # 工程说明和评估报告
├── src/
│   ├── evaluation/      # RAGAS 与三路对比评估
│   ├── graph/           # LangGraph 工作流与路由
│   ├── ingestion/       # 文档切分与索引
│   ├── models/          # LLM 与 Embedding 封装
│   ├── retrieval/       # No/Single/Multi/Adaptive 检索策略
│   └── safety/          # 输入/输出安全检测
├── test_data/           # StarVault 基准测试文档与问题
├── tests/               # 单元测试与工作流测试
└── ui/                  # Streamlit 应用
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
# 编辑 .env，填写 LLM_API_KEY / LLM_BASE_URL / 模型名称

streamlit run ui/app.py
```

CLI 示例：

```bash
python main.py ask "文档中天枢项目的预算是多少？"
python main.py eval "公司总预算和总支出分别是多少？"
python test_data/run_full_benchmark.py --output test_data/benchmark_result_after.json --csv-output test_data/benchmark_summary_after.csv
```

运行测试：

```bash
python -m pytest tests/ -v
```

## 环境配置

项目使用 OpenAI 兼容的 Chat Completion API，可接入 OpenAI、DeepSeek、Ollama、vLLM、Azure OpenAI 或其他兼容服务。

最小 `.env` 示例：

```env
LLM_API_KEY=sk-your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_DEFAULT_MODEL=gpt-4o-mini
LLM_SIMPLE_MODEL=gpt-4o-mini
LLM_MEDIUM_MODEL=gpt-4o-mini
LLM_COMPLEX_MODEL=gpt-4o

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANK_ENABLED=true
```

不要提交 `.env`、本地向量数据库、模型缓存、运行日志或测试生成数据。

## 基准测试

项目内置 StarVault 基准测试，使用一份内部项目汇报文档和 8 个问题覆盖不同能力：

- Q1/Q2：单点事实检索
- Q3/Q4：列表聚合与数值计算
- Q5/Q6：隐含推断
- Q7：文档内部矛盾识别
- Q8：不可回答问题的诚实边界

当前本地回归结论：

- Q1/Q2/Q3/Q4 路由到 `medium / single_step`
- Q5/Q6/Q7/Q8 路由到 `complex / multi_step`
- Q5 能召回团队成员表，并基于技能、项目状态和时间线给出候选人推断
- Q8 在文档缺少融资 / 估值信息时能够明确说明未找到相关信息

RAGAS 是有价值的自动评估信号，但不能单独代表全部答案质量。Q5 和 Q8 的评估局限已记录在 [docs/evaluation_report.md](docs/evaluation_report.md)。

## 作品集说明

这个仓库适合作为 RAG 工程作品集展示。建议重点介绍：

- 为什么 Adaptive RAG 能在事实题上避免不必要的多步检索成本
- 为什么表格问答不能只依赖 LLM 归纳，需要确定性数值校验
- 为什么隐含推断问题需要证据覆盖保护，避免关键表格被上下文截断
- 为什么 RAGAS 需要结合任务型校验器一起使用
- 如何用工作流测试保护路由、检索范围、安全检测和评估逻辑

## 已知限制

- 完整基准测试包含生成和 RAGAS 调用，耗时较高
- 部分 OpenAI 兼容评估模型下，RAGAS 指标可能不稳定
- 本地 Embedding / Reranker 模型首次运行时可能需要下载
- 项目主要面向工程展示和学习，不是生产级多租户系统
