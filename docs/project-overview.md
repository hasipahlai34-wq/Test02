# 当前项目工程说明

本文说明 `adaptive-rag-system` 当前版本的工程形态，方便 GitHub 访问者快速理解项目不是单一脚本，而是一套完整的自适应 RAG 系统。

## 项目定位

`adaptive-rag-system` 是从完整实验项目中抽离出来的可展示版本，目标是演示：

- 如何用 LangGraph 编排 RAG 问答工作流。
- 如何根据问题复杂度动态选择检索策略。
- 如何组合关键词检索、向量检索、RRF 融合和重排。
- 如何处理表格数值、隐含推断和不可回答问题。
- 如何在生成后加入质量评估、安全检查、HITL 和缓存。

## 核心工作流

主流程在 `src/graph/workflow.py` 中定义：

```text
classify_query
  -> cache_lookup
  -> route_by_complexity
  -> no_retrieval | single_step | multi_step
  -> generate
  -> review
  -> ragas_evaluate
  -> guard
  -> hitl_gate
  -> cache_store
  -> END
```

关键点：

- `classify_query` 判断 simple / medium / complex。
- `cache_lookup` 优先复用已有答案，降低重复调用成本。
- `route_by_complexity` 是 Adaptive RAG 的核心条件路由。
- `review` 和 `guard` 在答案返回前做质量与安全校验。
- `hitl_gate` 在低质量、高风险或低评估分时触发人工审核。

## 检索策略

### No Retrieval

用于简单问题或不需要知识库上下文的问题，跳过检索，直接生成答案。

### Single-step RAG

用于中等复杂度问题，主要流程：

```text
Query
  -> BM25 Search
  -> Dense Search
  -> RRF Fusion
  -> Cross-encoder Rerank
  -> Context Assembly
```

该策略适合事实查找、列表聚合和可直接从文档定位的信息。

### Multi-step RAG

用于复杂问题，当前实现包括：

- multi-query rewrite：生成多个保守改写查询。
- step-back query：为分析类问题补充背景证据查询。
- HyDE fallback：在保守检索召回不足时作为兜底。
- iterative retrieval：最多多轮检索并评估证据是否充分。
- evidence coverage guard：对隐含推断问题补齐人员、项目状态、时间线等证据。

该策略适合隐含推断、矛盾识别、战略分析和不可回答边界判断。

## 表格与数值问题

项目对 Markdown 表格做了额外处理。对于预算、支出、剩余、合计等问题，系统不只依赖 LLM 从上下文中总结，而是尝试进行确定性数值聚合，降低下面几类错误：

- 合计漏项。
- 把预算和支出列混淆。
- 对不可回答问题编造数字。
- 检索到表格但上下文截断导致答案缺证据。

## 评估体系

项目内置三路对比：

- Direct Answer：不检索，直接回答。
- Standard RAG：固定使用标准检索。
- Adaptive RAG：按问题复杂度动态路由。

评估指标包括：

- 响应时间。
- 召回文档数量。
- RAGAS faithfulness。
- RAGAS answer relevancy。
- RAGAS context precision。

注意：RAGAS 对隐含推断和不可回答问题并不总是可靠。因此项目保留任务型校验思路，例如：

- 数值题校验计算结果。
- 不可回答题校验是否拒绝编造。
- 隐含推断题校验证据覆盖，而不是只看一句结论。

## 运行入口

统一入口是 `main.py`：

```bash
python main.py ui              # Streamlit UI
python main.py serve           # FastAPI 服务
python main.py ask <query>     # 单次问答
python main.py chat            # 终端对话
python main.py eval <query>    # 三路对比评估
python main.py ingest <path>   # 摄入文档
```

## 不提交的本地文件

这个仓库准备上传 GitHub 时应排除：

- `.env`
- `data/chroma/`
- `data/sqlite/`
- `data/hitl_queue/`
- `data/hitl_results/`
- `data/ltm_chroma/`
- `*.log`
- `*.pyc`
- 本地模型缓存和临时 benchmark 日志

这些都是运行时产物，不是项目源码。

## 当前展示重点

如果用于简历或面试，建议重点讲：

- 为什么要做复杂度路由，而不是所有问题都走同一条 RAG pipeline。
- 为什么混合检索需要 RRF 和 rerank，而不是只用向量相似度。
- 为什么复杂推断题需要 multi-query 和证据覆盖。
- 为什么评估不能只看 RAGAS，需要任务型校验和人工 rubric。
- 如何用测试保护路由、检索范围、安全和评估逻辑。
