# 任务：Multi-Step 检索结果增加"证据覆盖保护"重新排序

## 背景

Adaptive-RAG 项目在处理隐含推断类查询（如"如果天枢项目需要紧急加人，谁最可能被抽调？"）时，MultiStepStrategy 已做了专项优化：对隐含推断查询自动生成 3 个子查询分别检索「团队成员技能」「项目进度人力」「技术栈匹配」。8 个 chunk 全部入库，团队表（chunk=5）也被检索到了。

**真正的问题是**：multi_step 最终按 `score` 降序排列（`multi_step.py` 第 264 行），团队表得分低于项目进展描述段落，被挤到第 6 位。而生成阶段 `_build_context()`（`generator.py` 第 68-84 行）按顺序遍历文档直到 token 预算耗尽（medium 复杂度默认 3000 字符），前 5 个项目描述 chunk 就会耗尽预算，团队表根本进不了上下文窗口。

**结论**：不改路由、不改 prompt、不提高 top_k。只在 multi_step 最终排序阶段加一道「证据覆盖保护」：识别隐含推断查询需要的证据类型，确保每种类型至少保留一个 chunk 在 top 位置。

---

## 修改范围

### 只改一个文件

**文件路径**：`src/retrieval/multi_step.py`

### 改动位置

在 `retrieve()` 方法的 **Step 5 排序之后、return SearchResult 之前**，即当前第 264 行之后：

```python
# 当前代码 (第 263-264 行)
# Step 5: 排序 — 按分数降序
all_documents.sort(key=lambda d: d.score, reverse=True)
```

在这之后插入证据覆盖保护逻辑。

### 不要改动的部分
- 第 48-58 行的 `IMPLICIT_INFERENCE_PATTERNS` 和 `is_implicit_inference_query()` — 保留不动
- 第 194-216 行的隐含推断子查询分发逻辑 — 保留不动
- 第 218-261 行的 HyDE + 迭代循环 — 保留不动
- `generator.py`、`workflow.py`、prompt 模板、`settings.py` — 全部不动

---

## 需求详述

### 功能：`_ensure_evidence_coverage(documents, query)`

新增一个模块级函数，对隐含推断查询执行证据覆盖重排。

#### 输入
- `documents: list[Document]` — 已按 score 降序排列的文档列表
- `query: str` — 用户原始查询

#### 输出
- `list[Document]` — 重排后的文档列表（保持原有文档对象不变，只调整顺序）

#### 核心逻辑

**1. 判断是否需要覆盖保护**

调用已有的 `is_implicit_inference_query(query)` 判断。如果不是隐含推断查询，直接返回原列表不做任何处理。

**2. 定义证据类型及检测规则**

对隐含推断类查询，定义 3 类需要覆盖的证据：

| 证据类型 | 说明 | 检测方式（满足任一即命中） |
|---------|------|------------------------|
| `personnel` | 人员/技能信息 | content 含关键词 `姓名` 或 `职位` 或 `技能`，或 metadata.source 含 `团队` |
| `project_status` | 项目进度/需求 | content 含关键词 `项目` 或 `进度` 或 `预计` 或 `发布` 或 `滞后` |
| `timeline_resource` | 时间线/预算/资源 | content 含关键词 `预算` 或 `时间线` 或 `Q1` 或 `Q2` 或 `支出` 或 `剩余` 或 `测试中` |

> 关键词需同时匹配大小写不敏感（英文关键词小写化后匹配）。

**3. 覆盖检查与提升**

```
CONST top_window = 6  # 只保护 top-6 窗口

对于每个证据类型 type：
  # 检查 type 是否已有文档在 top_window 内
  has_cover = any(documents[i] 命中 type 规则 for i in range(0, min(top_window, len(documents))))

  if has_cover:
    continue  # 该类型已覆盖，跳过

  # 该类型缺失：从 top_window 之后找该类型的最高分文档
  best_doc, best_idx = None, -1
  for i in range(top_window, len(documents)):
    if documents[i] 命中 type 规则:
      best_doc, best_idx = documents[i], i
      break  # 已按 score 排序，第一个就是最高分

  if best_doc:
    # 将该文档提升到 top_window 的最后一位
    # 方式：删除原位置，插入到 top_window 位置
    del documents[best_idx]
    documents.insert(top_window, best_doc)
    top_window += 1  # 后续类型插入位置后移，避免互相覆盖
```

**4. 去重保护**

如果某个文档已被提升过一次（比如它同时命中 `personnel` 和 `project_status`），不应重复提升。用 `promoted_ids = set()` 跟踪已被提升的文档（用 `id(doc)` 或 `(doc.content[:80], doc.metadata.get("chunk_index"))` 作为唯一标识）。

**5. 日志**

每次提升操作输出 info 日志：
```python
logger.info("证据覆盖保护: 类型=%s 文档(chunk=%s) 从位置%d提升到%d, score=%.3f",
    evidence_type,
    doc.metadata.get("chunk_index", "?"),
    original_position,
    new_position,
    doc.score,
)
```

如所有类型都已覆盖，输出：
```python
logger.info("证据覆盖保护: 所有证据类型已覆盖 (top-6), 无需调整")
```

#### 伪代码总览

```python
def _ensure_evidence_coverage(
    documents: list[Document],
    query: str,
) -> list[Document]:
    """对隐含推断查询确保证据覆盖：人员表/项目进度/时间线各至少一个在 top 窗口。"""
    if not is_implicit_inference_query(query):
        return documents

    if len(documents) < 4:
        return documents  # 文档太少，不需要保护

    EVIDENCE_PATTERNS = {
        "personnel": [
            r"姓名", r"职位", r"技能", r"所属部门",
        ],
        "project_status": [
            r"项目", r"进度", r"预计", r"发布", r"滞后",
        ],
        "timeline_resource": [
            r"预算", r"时间线", r"Q1", r"Q2", r"支出", r"剩余", r"测试中",
        ],
    }

    def _match_type(doc: Document, evidence_type: str) -> bool:
        content_lower = doc.content.lower()
        source = doc.metadata.get("source", "").lower()
        patterns = EVIDENCE_PATTERNS[evidence_type]
        for pat in patterns:
            if pat.lower() in content_lower or pat.lower() in source:
                return True
        return False

    top_window = min(6, len(documents))
    promoted_ids: set[int] = set()
    insert_pos = top_window  # 提升文档的插入位置

    for ev_type in EVIDENCE_PATTERNS:
        # 检查 top_window 内是否已有覆盖
        covered = any(
            _match_type(documents[i], ev_type)
            for i in range(insert_pos)
            if id(documents[i]) not in promoted_ids
        )
        if covered:
            continue

        # 从 insert_pos 之后找该类型的最高分文档
        best_idx = -1
        for i in range(insert_pos, len(documents)):
            if _match_type(documents[i], ev_type) and id(documents[i]) not in promoted_ids:
                best_idx = i
                break  # 已按 score 降序排列

        if best_idx < 0:
            continue

        doc = documents[best_idx]
        promoted_ids.add(id(doc))
        original_pos = best_idx
        del documents[best_idx]
        documents.insert(insert_pos, doc)
        insert_pos += 1

        logger.info(
            "证据覆盖保护: 类型=%s chunk=%s 从位置%d提升到%d, score=%.3f",
            ev_type,
            doc.metadata.get("chunk_index", "?"),
            original_pos,
            insert_pos - 1,
            doc.score,
        )

    if not promoted_ids:
        logger.info("证据覆盖保护: 所有证据类型已覆盖 (top-%d), 无需调整", top_window)

    return documents
```

### 调用方式

在 `retrieve()` 方法的 Step 5 之后调用：

```python
# Step 5: 排序 — 按分数降序
all_documents.sort(key=lambda d: d.score, reverse=True)

# Step 5.5: 证据覆盖保护 — 隐含推断查询确保关键证据类型不被挤出 top 窗口
all_documents = _ensure_evidence_coverage(all_documents, query)
```

> 注意：`query` 参数用 **用户原始查询**（函数开头的 `query` 变量），不是 HyDE 改写后的 `search_query`。

---

## 设计要求

1. **最小侵入**：只在 multi_step 排序后插入一个重排步骤，不动任何其他模块
2. **只提升不降级**：被提升的文档插入到 top_window 末尾，不替代原 top_window 中的文档；原 top_window 中的文档顺序和位置都不变（只是被"挤"后移一位）
3. **容错**：如果某个证据类型在整个文档列表中完全找不到匹配，跳过即可，不报错
4. **兼容**：重排后文档对象不变（同一个 Document 实例），只是列表顺序变化；不影响后续 generator 的 token 预算控制
5. **性能**：O(n) 复杂度，n = 文档数量（通常 8-20 个），无需担心性能

---

## 验证

修改完成后，用以下方式验证：

```bash
# 1. 编译检查
python -m compileall src/retrieval/multi_step.py

# 2. 单元测试
pytest tests/test_retrieval.py -v

# 3. 完整测试
pytest tests/ -v

# 4. 手动验证：索引 starvault_report.md，提问 Q5
# "如果天枢项目10月发布前需要紧急加人，谁最有可能被抽调过去帮忙？为什么？"
# 预期：团队表（chunk=5）应出现在 retrieved_docs[:6] 中
```

---

## 严禁行为

- ❌ 修改 generator.py、workflow.py、single_step.py、prompts 模板、settings.py
- ❌ 修改 `is_implicit_inference_query()` 函数签名或行为
- ❌ 修改子查询分发逻辑（第 194-216 行）
- ❌ 修改 `_build_context()` 的 token 预算或截断逻辑
- ❌ 提高 `RETRIEVAL_TOP_K` 或任何 settings 默认值
- ❌ 删除或降级任何已有文档（只能提升，不能删除）
