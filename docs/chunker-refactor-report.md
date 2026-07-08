# 文档切分系统重构报告

## 概述

将 `auto_chunk()` 从"按扩展名二选一"升级为**企业级结构感知 + 语义感知**分块系统，为 5 种文件类型各设计专属策略。

## 修改文件清单

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `config/settings.py` | **扩展** | 新增 `chunk_min_size`、`chunk_structure_aware`、`chunk_heading_font_delta`；`chunk_size` 500→800，`chunk_overlap` 50→100 |
| 2 | `src/ingestion/chunker.py` | **重构** | 新增 5 个专属策略函数 + 后处理器 + 新 `auto_chunk(file_path)`；旧版重命名为 `auto_chunk_legacy` |
| 3 | `ui/app.py` | **适配** | 调用改为 `auto_chunk(str(filepath))` |
| 4 | `main.py` | **适配** | 同上 |
| 5 | `api/routes.py` | **适配** | `strategy=auto` 或不指定时使用新 `auto_chunk`；显式指定策略时继续用 `chunk_documents` |
| 6 | `src/retrieval/chunker.py` | **适配** | 同步导出 `auto_chunk_legacy` |
| 7 | `requirements.txt` | **新增** | `pandas>=2.0.0` |

## 策略对比表

| 文件类型 | 旧策略 | 新策略 | 核心技术 |
|---------|--------|--------|---------|
| **PDF** | 递归盲切 | **结构感知** | pymupdf 提取字体大小 → 众数字号=正文 → 字号差≥2pt 识别标题 → 按标题边界切分 |
| **Word** | 递归盲切 | **样式感知** | python-docx 读取 Heading 1/2/3 样式 → 段落级章节切分 → 表格完整保留 |
| **Markdown** | 标题切分 | **增强标题切分** | 代码块/表格/列表 placeholder 保护 → 切分后还原，保证完整性 |
| **TXT** | 递归盲切 | **段落感知** | 按 `\n\n` 段落边界切分 → 中文句号/问号优先 → 过长段递归细切 |
| **CSV** | 递归盲切 | **行完整+表头携带** | 每 N 行一组 → 每组携带表头行 → 列名=值 格式 |

## 配置变更

```python
# 旧
chunk_size = 500
chunk_overlap = 50

# 新
chunk_size = 800              # 扩大窗口，减少碎片
chunk_overlap = 100           # 增加重叠，保持上下文连续性
chunk_min_size = 100          # ★ 新增：过小 chunk 自动合并
chunk_structure_aware = True  # ★ 新增：结构感知开关
chunk_heading_font_delta = 2  # ★ 新增：标题字体检测阈值(pt)
```

## 架构变更

```
旧流程: load_document(file) → auto_chunk(docs, ".pdf") → 递归盲切
        结构信息在此丢失 ↑

新流程: auto_chunk(file_path) → 内部打开文件 → 提取结构 → 结构感知切分
        完整保留字体/样式/标题 ↑

兼容: auto_chunk_legacy(docs, suffix) 保留向后兼容
```

## 新 auto_chunk 分发逻辑

```python
def auto_chunk(file_path: str) -> list[Document]:
    .pdf / .docx / .doc  → _chunk_pdf_structured / _chunk_docx_structured
    .md / .markdown      → _chunk_markdown_enhanced
    .txt                 → _chunk_txt_paragraph
    .csv                 → _chunk_csv_rows
    其他                  → _chunk_txt_paragraph (降级)

    策略失败 → 自动降级为 _chunk_txt_paragraph
    → _post_process_chunks() 合并过小 chunk (< chunk_min_size)
    → 统一注入 chunk_index / chunk_id / source_file 元数据
```

## Metadata 增强

每种策略注入差异化元数据：

| 策略 | 元数据字段 |
|------|-----------|
| PDF | `heading_path`, `heading_level`, `page_number`, `chunk_type` |
| DOCX | `heading_path`, `heading_level`, `chunk_type` |
| Markdown | `heading_path`, `heading_level`, `has_code`, `has_table`, `chunk_type` |
| TXT | `paragraph_index`, `chunk_type` |
| CSV | `columns`, `row_range`, `total_rows`, `chunk_type` |
| 通用 | `source_file`, `chunk_index`, `chunk_id`, `chunk_strategy` |

## 降级保护链

```
PDF: pymupdf 失败 → TXT 段落感知
DOCX: python-docx 失败 → TXT 段落感知
Markdown: MarkdownHeaderTextSplitter 失败 → 纯文本降级
任意策略异常 → _chunk_txt_paragraph 兜底
```

## 验证结果

| 验证项 | 结果 |
|--------|------|
| 65 个单元测试 | ✅ 65 passed, 1 warning (pre-existing ragas) |
| Markdown 增强分块 | ✅ 3 chunks，heading_path 层级完整 |
| TXT 段落分块 | ✅ 2 chunks，按段落边界，paragraph_index 正确 |
| CSV 行完整分块 | ✅ 1 chunk，表头携带，8 行数据完整，row_range 正确 |
| 分发器正确性 | ✅ .md/.txt/.csv/.xyz 均命中正确策略 |
| 降级兜底 | ✅ .xyz → TXT 段落感知 |
| 向后兼容 | ✅ `auto_chunk_legacy` + `chunk_documents` 保留 |
| 导入完整性 | ✅ `src/ingestion.chunker` 和 `src/retrieval/chunker` 全部导出正常 |

## 版本记录

| 日期 | 变更 |
|------|------|
| 2026-07-04 | 初始重构：5 种专属策略 + 统一分发 + 后处理 + 完整降级链 |
