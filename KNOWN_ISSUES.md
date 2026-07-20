# 已知遗留问题

> 记录于 2026-07-04，Adaptive-RAG 全项目终审修复完成后。
> 以下是 6 个 🟢 轻微问题，不影响功能，留待日常维护逐步修复。

---

## M1: SemanticCache 相似度阈值硬编码

- **文件**: `src/cache/semantic_cache.py:52`
- **问题**: `similarity_threshold: float = 0.92` 硬编码在构造函数默认值中
- **修复**: 从 `config.settings` 读取，或作为可配置的环境变量暴露
- **影响**: 调整阈值需改代码重新部署，不够灵活

## M2: MultiStep 最大迭代次数硬编码

- **文件**: `src/retrieval/multi_step.py:44`
- **问题**: `MAX_ITERATIONS = 3` 模块级常量，不应硬编码
- **修复**: 移至 `config.settings` 或作为 `MultiStepStrategy.__init__()` 参数
- **影响**: 复杂查询可能因迭代次数固定而检索不充分

## M3: ShortTermMemory 最大窗口大小硬编码

- **文件**: `src/memory/short_term.py:24`
- **问题**: `DEFAULT_MAX_ROUNDS = 10` 独立于 `settings.memory_short_term_max_rounds`
- **修复**: 统一从 `get_settings().memory_short_term_max_rounds` 读取
- **影响**: 修改 settings 中的配置不会生效，两处定义不一致

## M4: PROJECT_ROOT 模块级求值

- **文件**: `config/settings.py:26`
- **问题**: `PROJECT_ROOT = Path(__file__).resolve().parent.parent` 在 `import` 时求值
- **修复**: 使用 `@property` 延迟求值或 `pathlib.Path.cwd()` 替代
- **影响**: PyInstaller/py2app 打包后 `__file__` 路径可能不正确

## M5: MODEL_PRICING 价格数据过时

- **文件**: `src/utils/token_manager.py:31-36`
- **问题**: `MODEL_PRICING` 字典硬编码 2024 年模型价格（如 `gpt-4o: 5.00/15.00`），模型 ID 和价格已过时
- **修复**: 更新为当前官方定价，或从 `config/settings` 读取自定义价格
- **影响**: Token 成本统计不准确

## M6: 部分内部函数缺少 docstring

- **文件**: 多个文件
- **问题**: 以下函数缺少文档字符串：
  - `src/graph/workflow.py:_format_search_summary()` — 已有 docstring，但与 `retriever.py:RetrieverAgent._format_summary()` 功能重复
  - `src/retrieval/single_step.py:_tokenize()` — 缺少 docstring
  - `src/memory/long_term.py:_tokenize()` — 缺少 docstring
- **修复**: 补充 Google-style docstring
- **影响**: IDE 智能提示信息不完整，新开发者上手需要阅读源码

---

## 修复优先级建议

| 编号 | 估算工时 | 优先级 | 建议时间 |
|------|:---:|:---:|------|
| M3 | 0.25h | 先修 | 下次改 short_term 时顺手改 |
| M1 | 0.25h | 先修 | 下次改 cache 配置时顺手改 |
| M2 | 0.25h | 先修 | 下次调检索参数时顺手改 |
| M5 | 0.5h | 中等 | 季度价格更新时修复 |
| M4 | 1h | 中等 | 打包部署前必须修复 |
| M6 | 0.5h | 低 | 新成员入职时补充 |

**总计**: ~2.75 工时，建议在日常迭代中逐步消化，无需专项时间。
