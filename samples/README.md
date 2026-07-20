# 示例文档目录

此目录存放用于演示 Adaptive RAG 系统的示例文档。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `demo_report.md` | 合成的企业内部项目报告，用于演示文档上传、检索、表格问答和不可回答边界 |
| `demo_questions.md` | 与示例报告配套的问题清单 |

## 推荐演示方式

1. 启动后端和前端。
2. 打开 `http://localhost:3001/documents`。
3. 上传 `samples/demo_report.md`。
4. 在 `/chat` 中按 `samples/demo_questions.md` 提问。
5. 在 `/evaluation` 中运行同样问题，对比直接回答、标准 RAG 和自适应 RAG。

这些样例是合成数据，可以公开提交到 GitHub。
