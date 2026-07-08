# 任务：Adaptive-RAG 三路对比全量基准测试

## 背景

我有一个 Adaptive-RAG 文档问答系统，需要你用一份测试文档 + 8 个覆盖 L1~L4+Boundary 级别的问题，跑一次全量三路对比测试（Direct / Standard RAG / Adaptive RAG），并给出结构化的数据结果和答案对比。

---

## 你需要做什么

### 第一步：保存并运行基准测试脚本

将下面附带的完整 Python 脚本保存为 `test_data/run_full_benchmark.py`，然后在项目根目录执行：

```bash
python test_data/run_full_benchmark.py
```

**前置条件**：
1. 项目已启动（`streamlit run` 或 API 服务在运行）
2. `starvault_report.md` 已通过 `/ingest` 接口索引到向量数据库
3. `.env` 中 LLM 配置正确（中转站 gpt-5.4 用于日常问答）

如果 API 服务已在 `localhost:8000` 运行，脚本会自动调用 `/eval` 端点。否则脚本会降级为直接调用 `compare.py` 的 `run_comparison()` 函数。

### 第二步：将脚本输出的完整 JSON 粘贴回来

脚本会输出一个 JSON 对象，包含 8 个问题的三路对比结果。把完整 JSON 粘贴给我，我会逐题分析。

---

## 输出要求（脚本已内置，此处说明含义）

对每个问题，输出以下字段：

| 字段 | 说明 |
|------|------|
| `query` | 问题原文 |
| `level` | L1/L2/L3/L4/Boundary |
| `direct.answer` | 直接 LLM 回答（不检索） |
| `direct.time_ms` | 直接回答耗时 |
| `standard_rag.answer` | 标准 RAG 回答（单步 BM25+Dense+Rerank） |
| `standard_rag.time_ms` | 标准 RAG 耗时 |
| `standard_rag.docs_count` | 标准 RAG 检索到的文档数 |
| `standard_rag.scores` | RAGAS 评分 {faithfulness, answer_relevancy, context_precision?, context_recall?} |
| `standard_rag.retrieved_sources` | 检索到的来源列表 |
| `standard_rag.retrieved_chunk_ids` | 检索到的 chunk 索引列表（前 5） |
| `adaptive_rag.answer` | 自适应 RAG 回答 |
| `adaptive_rag.time_ms` | 自适应 RAG 耗时 |
| `adaptive_rag.complexity` | 自适应分类结果 (simple/medium/complex) |
| `adaptive_rag.strategy` | 使用的检索策略 (no_retrieval/single_step/multi_step) |
| `adaptive_rag.docs_count` | 自适应 RAG 检索到的文档数 |
| `adaptive_rag.scores` | RAGAS 评分 |
| `adaptive_rag.retrieved_sources` | 检索来源列表 |
| `adaptive_rag.retrieved_chunk_ids` | chunk 索引列表（前 6，含证据覆盖重排结果） |
| `adaptive_rag.eval_error` | 如 RAGAS 评估失败，此处记录错误原因 |
| `conclusion` | 三路对比结论（各项指标变化百分比） |

### 最终汇总表

脚本还会在最后输出一个 CSV 格式的汇总表，包含所有 8 题的三路对比关键指标，方便直接复制到 Excel。

---

## 附带：完整基准测试脚本

```python
#!/usr/bin/env python3
"""
Adaptive-RAG 全量三路对比基准测试
用 starvault_report.md + 8 个问题跑 Direct / Standard-RAG / Adaptive-RAG，
输出结构化 JSON + CSV 汇总表。

用法:
    python test_data/run_full_benchmark.py              # 直接调用 compare.py
    python test_data/run_full_benchmark.py --api        # 通过 localhost:8000/eval 调用
    python test_data/run_full_benchmark.py --output result.json  # 保存到文件
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ── 问题定义 ──────────────────────────────────────────────
QUESTIONS = [
    {
        "id": "Q1",
        "level": "L1",
        "question": "天枢项目的预算是多少？截至6月底支出了多少？",
        "ground_truth": "天枢项目预算320万元，截至6月底实际支出约210万元（Q1:85万 + Q2:125万）。",
        "expected_keywords": ["320万元", "320万", "210万元", "210万"],
    },
    {
        "id": "Q2",
        "level": "L1",
        "question": "玉衡项目的技术栈是什么？为什么选择了这个技术栈？",
        "ground_truth": "玉衡项目后端使用Go语言，选择原因是在并发场景下的性能优势通过了技术评审。",
        "expected_keywords": ["Go", "并发", "性能"],
    },
    {
        "id": "Q3",
        "level": "L2",
        "question": "星穹科技目前一共有多少个正在进行的项目？它们分别属于哪些部门？",
        "ground_truth": "共3个项目：天枢（智能平台部）、开阳（数据服务部）、玉衡（产品创新部）。",
        "expected_keywords": ["3个", "三个", "天枢", "开阳", "玉衡", "智能平台部", "数据服务部", "产品创新部"],
    },
    {
        "id": "Q4",
        "level": "L2",
        "question": "公司总预算和总支出分别是多少？哪个项目剩余预算最多？",
        "ground_truth": "总预算750万元，总支出465万元（Q1:195+Q2:270），剩余预算最多是开阳项目（120万）。",
        "expected_keywords": ["750万", "465万", "开阳", "120万"],
    },
    {
        "id": "Q5",
        "level": "L3",
        "question": "如果天枢项目10月发布前需要紧急加人，谁最有可能被抽调过去帮忙？为什么？",
        "ground_truth": (
            "孙伟（数据工程师，Python兼容，开阳ETL已完成70%有余力）或王海东（已在智能平台部，技术栈完全匹配）；"
            "玉衡团队（Go技术栈）不太适合。合理推断需从技术栈匹配度和项目紧迫度两个维度分析。"
        ),
        "expected_keywords": ["王海东", "孙伟", "马晓军", "Python", "后端", "玉衡"],
    },
    {
        "id": "Q6",
        "level": "L3",
        "question": "公司目前有哪些收入来源？新战略方向是要控制什么？",
        "ground_truth": (
            "收入来源为智能客服系统企业授权和数据分析平台企业授权；"
            "战略要控制定制化项目不超过总收入的35%，重点投入SaaS产品线。"
        ),
        "expected_keywords": ["智能客服", "数据分析平台", "定制化", "35%", "SaaS"],
    },
    {
        "id": "Q7",
        "level": "L4",
        "question": "玉衡项目目前到底是什么状态？有没有什么异常？",
        "ground_truth": (
            "核心矛盾：时间线表写'预计6月完成'，但当前状态写'7月仍在内部测试中'；"
            "项目描述也说'预计6月完成MVP并启动内测'、'正在进行内部压力测试'。"
            "结论：玉衡MVP已延期至少1个月，'启动内测'未兑现。"
        ),
        "expected_keywords": ["延期", "6月", "7月", "测试", "矛盾", "不一致", "MVP"],
    },
    {
        "id": "Q8",
        "level": "Boundary",
        "question": "星穹科技下一轮融资计划是什么？预计估值多少？",
        "ground_truth": "",
        "expected_keywords": ["未提及", "没有", "融资计划", "A轮"],
    },
]


@dataclass
class BenchmarkResult:
    question_id: str
    level: str
    query: str
    direct_answer: dict = field(default_factory=dict)
    standard_rag: dict = field(default_factory=dict)
    adaptive_rag: dict = field(default_factory=dict)
    conclusion: str = ""
    error: Optional[str] = None
    duration_sec: float = 0.0


async def run_all_benchmarks(use_api: bool = False) -> list[BenchmarkResult]:
    """Run all 8 questions through the three-way comparison pipeline."""
    results: list[BenchmarkResult] = []

    if use_api:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            for q in QUESTIONS:
                t0 = time.perf_counter()
                result = BenchmarkResult(
                    question_id=q["id"],
                    level=q["level"],
                    query=q["question"],
                )
                try:
                    async with session.post(
                        "http://localhost:8000/eval",
                        json={"query": q["question"]},
                        timeout=aiohttp.ClientTimeout(total=180),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            result.direct_answer = data.get("direct_answer", {})
                            result.standard_rag = data.get("standard_rag", {})
                            result.adaptive_rag = data.get("adaptive_rag", {})
                            result.conclusion = data.get("conclusion", "")
                        else:
                            error_text = await resp.text()
                            result.error = f"HTTP {resp.status}: {error_text[:300]}"
                except asyncio.TimeoutError:
                    result.error = "Timeout (>180s)"
                except Exception as e:
                    result.error = f"{type(e).__name__}: {e}"
                result.duration_sec = time.perf_counter() - t0
                results.append(result)
                print(f"  [{q['id']}] {q['level']} — {result.duration_sec:.1f}s", flush=True)
    else:
        # 直接调用 compare.py 的 run_comparison
        from src.evaluation.compare import run_comparison

        for q in QUESTIONS:
            t0 = time.perf_counter()
            result = BenchmarkResult(
                question_id=q["id"],
                level=q["level"],
                query=q["question"],
            )
            print(f"\n{'='*60}", flush=True)
            print(f"[{q['id']}] {q['level']} — {q['question'][:60]}...", flush=True)
            print(f"{'='*60}", flush=True)

            try:
                gt = q.get("ground_truth") or None
                comparison = await run_comparison(
                    query=q["question"],
                    ground_truth=gt,
                )
                result.direct_answer = comparison.direct_answer
                result.standard_rag = comparison.standard_rag
                result.adaptive_rag = comparison.adaptive_rag
                result.conclusion = comparison.conclusion
            except Exception as e:
                result.error = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                print(f"  ❌ ERROR: {result.error}", flush=True)
                print(f"  {tb[:500]}", flush=True)

            result.duration_sec = time.perf_counter() - t0
            results.append(result)
            print(f"  ⏱ {result.duration_sec:.1f}s", flush=True)

    return results


def safe_get(obj: dict, *keys, default=""):
    """Safely navigate nested dicts."""
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, {})
        else:
            return default
    return obj if obj != {} else default


def format_results_json(results: list[BenchmarkResult]) -> list[dict]:
    """Convert results to a clean JSON-serializable structure."""
    output = []
    for r in results:
        entry = {
            "id": r.question_id,
            "level": r.level,
            "query": r.query,
            "error": r.error,
            "duration_sec": round(r.duration_sec, 1),
            "direct": {
                "answer": safe_get(r.direct_answer, "answer")[:500],
                "time_ms": round(safe_get(r.direct_answer, "time_ms", default=0), 0),
                "model": safe_get(r.direct_answer, "model"),
            },
            "standard_rag": {
                "answer": safe_get(r.standard_rag, "answer")[:800],
                "time_ms": round(safe_get(r.standard_rag, "time_ms", default=0), 0),
                "docs_count": safe_get(r.standard_rag, "docs_count", default=0),
                "model": safe_get(r.standard_rag, "model"),
                "scores": safe_get(r.standard_rag, "scores"),
                "eval_error": safe_get(r.standard_rag, "eval_error"),
                "retrieved_sources": safe_get(r.standard_rag, "retrieved_sources"),
                "retrieved_chunk_ids": _extract_chunk_ids(r.standard_rag),
            },
            "adaptive_rag": {
                "answer": safe_get(r.adaptive_rag, "answer")[:800],
                "time_ms": round(safe_get(r.adaptive_rag, "time_ms", default=0), 0),
                "complexity": safe_get(r.adaptive_rag, "complexity"),
                "strategy": safe_get(r.adaptive_rag, "strategy"),
                "docs_count": safe_get(r.adaptive_rag, "docs_count", default=0),
                "model": safe_get(r.adaptive_rag, "model"),
                "scores": safe_get(r.adaptive_rag, "scores"),
                "eval_error": safe_get(r.adaptive_rag, "eval_error"),
                "retrieved_sources": safe_get(r.adaptive_rag, "retrieved_sources"),
                "retrieved_chunk_ids": _extract_chunk_ids(r.adaptive_rag),
            },
            "conclusion": r.conclusion[:500],
        }
        output.append(entry)
    return output


def _extract_chunk_ids(rag_result: dict) -> list:
    """Extract chunk_index from contexts_preview or retrieved sources."""
    contexts = rag_result.get("contexts_preview", [])
    if not contexts and rag_result.get("retrieved_document_ids"):
        return rag_result.get("retrieved_document_ids", [])[:6]
    ids = []
    for ctx in contexts:
        # Try to extract chunk_index from metadata patterns in the content
        import re
        m = re.search(r"chunk[_\s]*(\d+)", str(ctx), re.IGNORECASE)
        if m:
            ids.append(int(m.group(1)))
    return ids[:6]


def print_csv_summary(results: list[BenchmarkResult]):
    """Print a CSV-formatted summary table for easy Excel import."""
    print("\n" + "=" * 100)
    print("CSV 汇总表 (可直接复制到 Excel)")
    print("=" * 100)
    header = (
        "ID,Level,"
        "Direct_Time_ms,"
        "StdRAG_Time_ms,StdRAG_Docs,StdRAG_Faithfulness,StdRAG_Relevancy,StdRAG_Precision,"
        "AdpRAG_Time_ms,AdpRAG_Complexity,AdpRAG_Strategy,AdpRAG_Docs,"
        "AdpRAG_Faithfulness,AdpRAG_Relevancy,AdpRAG_Precision,"
        "Faithfulness_Delta%,Relevancy_Delta%,Precision_Delta%,"
        "Error"
    )
    print(header)

    for r in results:
        std = r.standard_rag
        adp = r.adaptive_rag

        std_scores = std.get("scores") or {}
        adp_scores = adp.get("scores") or {}

        std_f = std_scores.get("faithfulness", 0)
        std_r = std_scores.get("answer_relevancy", 0)
        std_p = std_scores.get("context_precision", 0)
        adp_f = adp_scores.get("faithfulness", 0)
        adp_r = adp_scores.get("answer_relevancy", 0)
        adp_p = adp_scores.get("context_precision", 0)

        def delta_pct(a, b):
            if b == 0:
                return "N/A"
            return f"{(a-b)/b*100:+.0f}%"

        error = (r.error or "").replace(",", ";").replace('"', "'")
        row = (
            f"{r.question_id},{r.level},"
            f"{r.direct_answer.get('time_ms', 0):.0f},"
            f"{std.get('time_ms', 0):.0f},{std.get('docs_count', 0)},{std_f:.3f},{std_r:.3f},{std_p:.3f},"
            f"{adp.get('time_ms', 0):.0f},{adp.get('complexity', '?')},{adp.get('strategy', '?')},{adp.get('docs_count', 0)},"
            f"{adp_f:.3f},{adp_r:.3f},{adp_p:.3f},"
            f"{delta_pct(adp_f, std_f)},{delta_pct(adp_r, std_r)},{delta_pct(adp_p, std_p)},"
            f'"{error}"'
        )
        print(row)


def print_human_readable(results: list[BenchmarkResult]):
    """Print human-readable results for each question."""
    print("\n" + "█" * 80)
    print("  Adaptive-RAG 三路对比基准测试 — 详细结果")
    print("█" * 80)
    print(f"  时间: {datetime.now().isoformat()}")
    print(f"  问题数: {len(results)}")
    success = sum(1 for r in results if not r.error)
    failed = sum(1 for r in results if r.error)
    print(f"  成功: {success} / 失败: {failed}")
    print("█" * 80)

    for r in results:
        print(f"\n{'─'*80}")
        print(f"  [{r.question_id}] {r.level}  —  {r.query}")
        print(f"{'─'*80}")

        if r.error:
            print(f"  ❌ 错误: {r.error}")
            continue

        # Direct
        d = r.direct_answer
        print(f"\n  📍 路径1: 直接回答 (无检索)")
        print(f"     耗时: {d.get('time_ms', 0):.0f}ms")
        print(f"     模型: {d.get('model', '?')}")
        print(f"     回答: {d.get('answer', '')[:300]}...")

        # Standard RAG
        s = r.standard_rag
        print(f"\n  📍 路径2: 标准 RAG (单步检索)")
        print(f"     耗时: {s.get('time_ms', 0):.0f}ms | 检索数: {s.get('docs_count', 0)}")
        print(f"     模型: {s.get('model', '?')}")
        scores = s.get("scores") or {}
        print(f"     RAGAS: faithfulness={scores.get('faithfulness', '-'):.3f}  "
              f"relevancy={scores.get('answer_relevancy', '-'):.3f}  "
              f"precision={scores.get('context_precision', '-'):.3f}")
        if s.get("eval_error"):
            print(f"     ⚠ RAGAS 评估异常: {s['eval_error']}")
        print(f"     Chunk IDs: {_extract_chunk_ids(s)}")
        print(f"     回答: {s.get('answer', '')[:300]}...")

        # Adaptive RAG
        a = r.adaptive_rag
        print(f"\n  📍 路径3: 自适应 RAG")
        print(f"     复杂度: {a.get('complexity', '?')} | 策略: {a.get('strategy', '?')}")
        print(f"     耗时: {a.get('time_ms', 0):.0f}ms | 检索数: {a.get('docs_count', 0)}")
        print(f"     模型: {a.get('model', '?')}")
        scores = a.get("scores") or {}
        print(f"     RAGAS: faithfulness={scores.get('faithfulness', '-'):.3f}  "
              f"relevancy={scores.get('answer_relevancy', '-'):.3f}  "
              f"precision={scores.get('context_precision', '-'):.3f}")
        if a.get("eval_error"):
            print(f"     ⚠ RAGAS 评估异常: {a['eval_error']}")
        print(f"     Chunk IDs: {_extract_chunk_ids(a)}")
        print(f"     回答: {a.get('answer', '')[:300]}...")

        # Conclusion
        print(f"\n  📊 对比结论: {r.conclusion[:200]}...")


async def main():
    use_api = "--api" in sys.argv
    output_file = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]

    print("=" * 60)
    print("Adaptive-RAG 三路对比全量基准测试")
    print(f"模式: {'API (localhost:8000)' if use_api else '直接调用 compare.py'}")
    print(f"问题数: {len(QUESTIONS)}")
    print("=" * 60)

    results = await run_all_benchmarks(use_api=use_api)

    # ── 输出 ──
    print_human_readable(results)
    print_csv_summary(results)

    # JSON 输出
    json_data = format_results_json(results)
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)

    if output_file:
        from pathlib import Path
        Path(output_file).write_text(json_str, encoding="utf-8")
        print(f"\n✅ 结果已保存到: {output_file}")

    print(f"\n{'='*60}")
    print("完整 JSON 结果:")
    print(json_str)
    print(f"{'='*60}")

    # 统计
    success = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    print(f"\n📊 汇总: {len(success)}/{len(results)} 成功, {len(failed)} 失败")
    if failed:
        print("失败列表:")
        for r in failed:
            print(f"  - [{r.question_id}] {r.level}: {r.error}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

---

## 如何使用这个脚本

### 方法 A：直接调用 compare.py（推荐，无需启动 API）

```bash
cd d:\GitHub-Agent\adaptive-rag
python test_data/run_full_benchmark.py
```

### 方法 B：通过 API 调用

```bash
# 先启动 API 服务
cd d:\GitHub-Agent\adaptive-rag
python -m api.main &

# 再跑基准测试
python test_data/run_full_benchmark.py --api
```

### 保存结果到文件

```bash
python test_data/run_full_benchmark.py --output benchmark_result.json
```

---

## 把以下内容粘贴回来给我分析

1. **完整的 JSON 输出**（脚本最后打印的那个大 JSON）
2. **CSV 汇总表**（脚本最后打印的 CSV）
3. **控制台完整输出**（包括所有 INFO 日志，特别是 `证据覆盖保护` 相关的日志行）

拿到这些数据后，我会逐题分析：
- 哪些问题 Adaptive RAG 明显优于 Standard RAG
- 哪些问题两者持平或 Adaptive RAG 反而更差
- Q5（隐含推断）的证据覆盖保护是否生效
- Q7（矛盾识别）自适应策略是否识别出矛盾
- Q8（边界试探）是否能诚实说"不知道"
- 整体结论：当前的 Adaptive-RAG 是否值得
