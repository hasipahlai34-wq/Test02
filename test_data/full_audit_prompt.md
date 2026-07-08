# 任务：全量审计 evidence_coverage 改动 + 全量测试修复

## 背景

我对 `src/retrieval/multi_step.py` 做了一次修改：在最终排序后增加证据覆盖保护函数 `_ensure_evidence_coverage()`，确保隐含推断查询（如"谁最可能被抽调"）的关键证据类型（人员表/项目进度/时间线）至少各有一个 chunk 在 top 窗口内。

**修改后 85 个 pytest 全部通过，但实际三路对比评估（`/eval` 端点）对 Q5 问题不产生任何结果**——页面不显示答案、不报错、也不出现超时提示，就像什么都没发生。

---

## 你的任务

1. **全量审计这次改动** — 找出导致三路对比静默失败的根因
2. **全量测试** — 运行所有测试 + 模拟对比路径，确认修复后一切正常

---

## 改动范围（只改了一个文件）

**文件**：`src/retrieval/multi_step.py`

**改动内容**：
1. 第 72-95 行：新增 `EVIDENCE_PATTERNS` 模块级常量
2. 第 98-163 行：新增 `_ensure_evidence_coverage(documents, query)` 函数
3. 第 361 行：在 `retrieve()` 的 Step 5 排序后调用 `all_documents = _ensure_evidence_coverage(all_documents, query)`

**未改动的文件**：`generator.py`、`workflow.py`、`single_step.py`、prompt 模板、`settings.py`、`compare.py`、`ragas_eval.py`

---

## 三路对比的调用链（关键路径）

```
用户点击「三路对比」
  → api/routes.py  /eval 端点
    → run_comparison(query) [src/evaluation/compare.py:144]
      → asyncio.gather(
          _run_direct_answer,      # path 1
          _run_standard_rag,       # path 2
          _run_adaptive_rag,       # path 3 ← 出问题的路径
        )
      → _run_adaptive_rag [compare.py:98]
        → create_adaptive_chain() [adaptive.py:259]
          → 内部 import MultiStepStrategy [multi_step.py]
        → adaptive.retrieve(query, state)
          → 路由到 MultiStepStrategy.retrieve() [multi_step.py:251]
            → ...各种检索...
            → all_documents.sort(key=lambda d: d.score, reverse=True)  [line 358]
            → all_documents = _ensure_evidence_coverage(all_documents, query)  [line 361] ← 新代码
            → return SearchResult(documents=all_documents, ...)  [line 370]
        → docs = adaptive_result.documents  [compare.py:115]
        → contexts = [doc.content for doc in docs[:5]]  [compare.py:116]
        → answer = await llm_client.generate(...)  [compare.py:118]
```

**关键点**：`asyncio.gather` 默认行为是任何一个协程抛异常，gather 自己也抛异常，导致三个路径全部失败、没有任何输出。

---

## 审计清单

### 检查点 1：`_ensure_evidence_coverage` 自身是否会抛异常

逐行审查以下高风险位置：

```python
# 第 109 行：list(documents) — documents 可能是 SearchResult 或其他类型？
reordered = list(documents)

# 第 111-113 行：doc.content 是否可能为 None 或非字符串？
content_lower = (doc.content or "").lower()
metadata = doc.metadata or {}

# 第 128-129 行：range 参数是否可能越界？
for i in range(min(insert_pos, len(reordered)))
    if id(reordered[i]) not in promoted_ids

# 第 145-150 行：del + insert 是否会导致索引错误？
doc = reordered[best_idx]
del reordered[best_idx]
reordered.insert(insert_pos, doc)

# 第 153 行：metadata 可能为 None？
(doc.metadata or {}).get("chunk_index", "?")
```

**重点**：如果 `all_documents` 是空列表（`len < 4`），函数在第 106-107 行会 `return documents`（原样返回），这没问题。但如果 len >= 4 但某些 doc 的 metadata 或 content 为 None，`_match_type` 会怎么处理？

### 检查点 2：返回类型兼容性

```python
# multi_step.py line 361
all_documents = _ensure_evidence_coverage(all_documents, query)
# all_documents 原本是 list[Document]
# _ensure_evidence_coverage 返回 reordered（也是 list，但是重新拼接的）
# 下游代码假设 all_documents 是 list[Document]，需确认类型不变
```

### 检查点 3：与 compare.py 的交互

`compare.py` 第 115-116 行：
```python
docs = adaptive_result.documents        # ← adaptive_result 来自 adaptive.retrieve()
contexts = [doc.content for doc in docs[:5]]
```

`MultiStepStrategy.retrieve()` 第 370-376 行返回：
```python
return SearchResult(
    query=query,
    documents=all_documents,   # ← 这是 _ensure_evidence_coverage 的返回值
    ...
)
```

**检查**：`SearchResult` 的 `documents` 字段预期类型是什么？`_ensure_evidence_coverage` 返回的是否兼容？

### 检查点 4：模块加载时是否已有异常

`EVIDENCE_PATTERNS` 是模块级常量（第 72 行），在 `import multi_step` 时就会执行。如果这里有问题，会在导入时就报错（但 pytest 通过了所以应该没问题）。

### 检查点 5：Q5 的覆盖率逻辑本身是否有问题

Q5 原文："如果天枢项目10月发布前需要紧急加人，谁最有可能被抽调过去帮忙？为什么？"

问题命中 `is_implicit_inference_query` → 子查询分发 → HyDE → 迭代 → 排序 → 证据覆盖保护。确认每个步骤都有 try/except 保护（第 298-300 行子查询有保护，但整个 retrieve 方法是否有可能在某处无保护地抛异常？）。

---

## 全量测试命令

```bash
# 1. 语法 / 导入检查
python -c "from src.retrieval.multi_step import MultiStepStrategy, _ensure_evidence_coverage, EVIDENCE_PATTERNS, is_implicit_inference_query; print('Import OK')"

# 2. 单元测试：_ensure_evidence_coverage 各种边界条件
python -c "
from src.retrieval.multi_step import _ensure_evidence_coverage, is_implicit_inference_query
from langchain_core.documents import Document

# 测试1：非隐含推断查询 → 原样返回
docs = [Document(page_content='测试', metadata={'chunk_index': 1})]
result = _ensure_evidence_coverage(docs, '你好')
assert result == docs, '非推断查询应原样返回'
print('PASS: 非推断查询原样返回')

# 测试2：文档太少 → 原样返回
docs2 = [Document(page_content='测试', metadata={'chunk_index': i}) for i in range(3)]
result2 = _ensure_evidence_coverage(docs2, '如果天枢项目需要加人，谁最可能被抽调？')
assert len(result2) == 3, '文档太少应原样返回'
print('PASS: 文档太少原样返回')

# 测试3：所有证据类型已覆盖 → 无改动但有日志
import logging
logging.basicConfig(level=logging.INFO)
docs3 = [
    Document(page_content='天枢项目：智能客服 项目进度正常', metadata={'chunk_index': 0, 'source': 'report.md'}),
    Document(page_content='团队表：姓名 职位 技能 所属部门', metadata={'chunk_index': 1, 'source': 'report.md'}),
    Document(page_content='时间线：预算 320万 Q1支出 85万', metadata={'chunk_index': 2, 'source': 'report.md'}),
    Document(page_content='玉衡项目：自动化运维 进度滞后', metadata={'chunk_index': 3, 'source': 'report.md'}),
    Document(page_content='公司概况', metadata={'chunk_index': 4, 'source': 'report.md'}),
    Document(page_content='其他内容', metadata={'chunk_index': 5, 'source': 'report.md'}),
]
result3 = _ensure_evidence_coverage(docs3, '如果天枢项目需要紧急加人，谁最可能被抽调？')
assert len(result3) == len(docs3), '文档数量不应改变'
# 团队表(chunk=1)应该在top窗口
chunk_ids = [doc.metadata.get('chunk_index') for doc in result3]
print(f'重排后chunk顺序: {chunk_ids}')
assert 1 in chunk_ids[:6], '团队表(chunk=1)应该在top-6中'
print('PASS: 所有证据已覆盖，团队表在top窗口')

# 测试4：空文档列表
result4 = _ensure_evidence_coverage([], '如果天枢项目需要加人')
assert result4 == [], '空列表应原样返回'
print('PASS: 空列表原样返回')

print()
print('所有单元测试通过！')
"

# 3. pytest 全量
pytest tests/ -v

# 4. 模拟 compare.py 路径（不实际调 LLM，只测检索是否成功返回）
python -c "
import asyncio
from src.retrieval.adaptive import create_adaptive_chain
from src.types import AgentState

async def test():
    adaptive, _ = create_adaptive_chain()
    state = AgentState(query='如果天枢项目10月发布前需要紧急加人，谁最有可能被抽调过去帮忙？为什么？')
    try:
        result = await adaptive.retrieve(query=state.query, state=state)
        print(f'检索成功: {len(result.documents)} 个文档')
        print(f'文档chunk索引顺序: {[doc.metadata.get(\"chunk_index\", \"?\") for doc in result.documents]}')
    except Exception as e:
        print(f'检索失败: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test())
"

# 5. 回归：三路对比完整路径（需要索引测试文档 starvault_report.md）
#    在 Streamlit 中上传 starvault_report.md 后：
#    curl -X POST http://localhost:8000/eval -H 'Content-Type: application/json' -d '{"query":"如果天枢项目10月发布前需要紧急加人，谁最有可能被抽调过去帮忙？为什么？"}'
```

---

## 输出要求

1. **定位根因**：指出是哪个检查点出的问题，具体到文件和行号
2. **修复方案**：精确到改哪几行、怎么改
3. **验证**：修复后运行上面全部 5 组测试命令，确认全部通过

---

## 约束

- 只修改 `src/retrieval/multi_step.py`
- 不改 `compare.py`、`generator.py`、`adaptive.py`、`workflow.py`、`single_step.py`、`settings.py`、prompt 模板
- 可以调整 `_ensure_evidence_coverage` 的内部逻辑，但不能删除这个函数（保留证据覆盖保护的意图）
- 如果根因在 compare.py 的 `asyncio.gather` 异常传播，也先告诉我，不要直接改 compare.py
