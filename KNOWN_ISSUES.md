# 已知遗留问题

> 记录于项目整理阶段。以下问题不影响当前核心功能和测试通过，但建议在后续迭代中逐步处理。

## M1：SemanticCache 相似度阈值仍为硬编码

- **文件**：`src/cache/semantic_cache.py`
- **问题**：`similarity_threshold` 默认值写在构造函数中，调整阈值需要改代码
- **建议**：移动到 `config.settings` 或环境变量
- **影响**：语义缓存命中策略不够灵活

## M2：MultiStep 最大迭代次数仍为模块常量

- **文件**：`src/retrieval/multi_step.py`
- **问题**：最大迭代次数由模块常量控制
- **建议**：移动到 `config.settings` 或 `MultiStepStrategy.__init__()` 参数
- **影响**：不同复杂度任务无法灵活调整检索轮数

## M3：ShortTermMemory 默认窗口大小存在重复定义

- **文件**：`src/memory/short_term.py`
- **问题**：短期记忆最大轮数与 settings 中的配置存在重复定义
- **建议**：统一从 `get_settings().memory_short_term_max_rounds` 读取
- **影响**：修改配置时可能不会完全生效

## M4：PROJECT_ROOT 在模块导入时求值

- **文件**：`config/settings.py`
- **问题**：`PROJECT_ROOT = Path(__file__).resolve().parent.parent` 在 import 时固定
- **建议**：如需打包部署，可改为延迟求值或显式配置项目根目录
- **影响**：PyInstaller/py2app 等打包场景下路径可能不符合预期

## M5：模型价格表需要定期更新

- **文件**：`src/utils/token_manager.py`
- **问题**：`MODEL_PRICING` 中的价格数据可能随模型供应商调整而过期
- **建议**：将价格配置外置，或在版本发布时同步更新
- **影响**：Token 成本统计可能不准确

## M6：部分内部函数缺少 docstring

- **文件**：多个源码文件
- **问题**：少量内部工具函数缺少说明
- **建议**：后续维护时补充简短 docstring
- **影响**：不影响运行，但会增加新开发者阅读成本

## 处理优先级建议

| 编号 | 优先级 | 预估工时 | 建议处理时机 |
| --- | --- | --- | --- |
| M3 | P1 | 0.25h | 下次调整 memory 配置时处理 |
| M1 | P1 | 0.25h | 下次调整 cache 策略时处理 |
| M2 | P1 | 0.25h | 下次调整 multi_step 参数时处理 |
| M5 | P2 | 0.5h | 季度模型价格更新时处理 |
| M4 | P2 | 1h | 打包部署前处理 |
| M6 | P3 | 0.5h | 日常维护时补充 |

这些问题均不影响当前作品集展示目标。当前版本重点展示 Adaptive RAG 的路由、检索、评估和 UI 闭环能力。