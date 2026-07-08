#!/usr/bin/env python3
"""Run the StarVault three-way Adaptive-RAG benchmark.

This script indexes test_data/starvault_report.md into an isolated temporary
Chroma collection, then runs Direct / Standard RAG / Adaptive RAG for the 8
benchmark questions.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


QUESTIONS = [
    {
        "id": "Q1",
        "level": "L1",
        "question": "天枢项目的预算是多少？截至6月底支出了多少？",
        "ground_truth": "天枢项目预算320万元，截至6月底实际支出约210万元。",
    },
    {
        "id": "Q2",
        "level": "L1",
        "question": "玉衡项目的技术栈是什么？为什么选择了这个技术栈？",
        "ground_truth": "玉衡项目后端使用Go语言，原因是Go在并发场景下的性能优势通过了技术评审。",
    },
    {
        "id": "Q3",
        "level": "L2",
        "question": "星穹科技目前一共有多少个正在进行的项目？它们分别属于哪些部门？",
        "ground_truth": "共有3个项目：天枢属于智能平台部，开阳属于数据服务部，玉衡属于产品创新部。",
    },
    {
        "id": "Q4",
        "level": "L2",
        "question": "公司总预算和总支出分别是多少？哪个项目剩余预算最多？",
        "ground_truth": "总预算750万元，总支出465万元，剩余预算最多的是开阳项目，剩余120万元。",
    },
    {
        "id": "Q5",
        "level": "L3",
        "question": "如果天枢项目10月发布前需要紧急加人，谁最有可能被抽调过去帮忙？为什么？",
        "ground_truth": (
            "合理推断应基于技术栈匹配和项目紧迫度。候选包括王海东、孙伟或马晓军等，"
            "需要说明这是推断而非文档直接信息。"
        ),
    },
    {
        "id": "Q6",
        "level": "L3",
        "question": "公司目前有哪些收入来源？新战略方向是要控制什么？",
        "ground_truth": (
            "收入来源是智能客服系统和数据分析平台的企业授权；新战略是重点投入SaaS产品线，"
            "并控制定制化项目比例不超过总收入的35%。"
        ),
    },
    {
        "id": "Q7",
        "level": "L4",
        "question": "玉衡项目目前到底是什么状态？有没有什么异常？",
        "ground_truth": (
            "玉衡MVP预计6月完成，但7月仍在内部测试中，存在时间线或状态描述不一致，"
            "说明项目可能延期。"
        ),
    },
    {
        "id": "Q8",
        "level": "Boundary",
        "question": "星穹科技下一轮融资计划是什么？预计估值多少？",
        "ground_truth": "文档未提及下一轮融资计划或估值，只提到公司处于A轮融资后的扩张期。",
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
    traceback: str = ""
    duration_sec: float = 0.0


def safe_get(obj: dict, *keys, default=""):
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, {})
        else:
            return default
    return obj if obj != {} else default


def _chunk_ids_from_docs(result: dict, limit: int = 6) -> list:
    ids = result.get("retrieved_chunk_ids")
    if ids:
        return ids[:limit]
    ids = result.get("retrieved_document_ids")
    if ids:
        return ids[:limit]
    return []


def _trim(text: object, limit: int) -> str:
    return str(text or "")[:limit]


def format_results_json(results: list[BenchmarkResult]) -> list[dict]:
    output = []
    for r in results:
        output.append({
            "id": r.question_id,
            "level": r.level,
            "query": r.query,
            "error": r.error,
            "duration_sec": round(r.duration_sec, 1),
            "direct": {
                "answer": _trim(safe_get(r.direct_answer, "answer"), 1200),
                "time_ms": round(safe_get(r.direct_answer, "time_ms", default=0), 0),
                "model": safe_get(r.direct_answer, "model"),
            },
            "standard_rag": {
                "answer": _trim(safe_get(r.standard_rag, "answer"), 1600),
                "time_ms": round(safe_get(r.standard_rag, "time_ms", default=0), 0),
                "docs_count": safe_get(r.standard_rag, "docs_count", default=0),
                "model": safe_get(r.standard_rag, "model"),
                "scores": safe_get(r.standard_rag, "scores"),
                "eval_error": safe_get(r.standard_rag, "eval_error"),
                "retrieved_sources": safe_get(r.standard_rag, "retrieved_sources"),
                "retrieved_chunk_ids": _chunk_ids_from_docs(r.standard_rag, 5),
            },
            "adaptive_rag": {
                "answer": _trim(safe_get(r.adaptive_rag, "answer"), 1600),
                "time_ms": round(safe_get(r.adaptive_rag, "time_ms", default=0), 0),
                "complexity": safe_get(r.adaptive_rag, "complexity"),
                "strategy": safe_get(r.adaptive_rag, "strategy"),
                "docs_count": safe_get(r.adaptive_rag, "docs_count", default=0),
                "model": safe_get(r.adaptive_rag, "model"),
                "scores": safe_get(r.adaptive_rag, "scores"),
                "eval_error": safe_get(r.adaptive_rag, "eval_error"),
                "retrieved_sources": safe_get(r.adaptive_rag, "retrieved_sources"),
                "retrieved_chunk_ids": _chunk_ids_from_docs(r.adaptive_rag, 6),
            },
            "conclusion": _trim(r.conclusion, 800),
        })
    return output


def csv_summary(results: list[BenchmarkResult]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([
        "ID",
        "Level",
        "Direct_Time_ms",
        "StdRAG_Time_ms",
        "StdRAG_Docs",
        "StdRAG_Faithfulness",
        "StdRAG_Relevancy",
        "StdRAG_Precision",
        "AdpRAG_Time_ms",
        "AdpRAG_Complexity",
        "AdpRAG_Strategy",
        "AdpRAG_Docs",
        "AdpRAG_Faithfulness",
        "AdpRAG_Relevancy",
        "AdpRAG_Precision",
        "Faithfulness_Delta%",
        "Relevancy_Delta%",
        "Precision_Delta%",
        "Error",
    ])
    for r in results:
        std = r.standard_rag or {}
        adp = r.adaptive_rag or {}
        std_scores = std.get("scores") or {}
        adp_scores = adp.get("scores") or {}
        std_f = std_scores.get("faithfulness", 0) or 0
        std_r = std_scores.get("answer_relevancy", 0) or 0
        std_p = std_scores.get("context_precision", 0) or 0
        adp_f = adp_scores.get("faithfulness", 0) or 0
        adp_r = adp_scores.get("answer_relevancy", 0) or 0
        adp_p = adp_scores.get("context_precision", 0) or 0

        def delta(a: float, b: float) -> str:
            return "N/A" if not b else f"{(a - b) / b * 100:+.0f}%"

        writer.writerow([
            r.question_id,
            r.level,
            round((r.direct_answer or {}).get("time_ms", 0), 0),
            round(std.get("time_ms", 0), 0),
            std.get("docs_count", 0),
            f"{std_f:.3f}",
            f"{std_r:.3f}",
            f"{std_p:.3f}",
            round(adp.get("time_ms", 0), 0),
            adp.get("complexity", "?"),
            adp.get("strategy", "?"),
            adp.get("docs_count", 0),
            f"{adp_f:.3f}",
            f"{adp_r:.3f}",
            f"{adp_p:.3f}",
            delta(adp_f, std_f),
            delta(adp_r, std_r),
            delta(adp_p, std_p),
            r.error or "",
        ])
    return out.getvalue()


async def _prepare_scoped_index():
    from src.ingestion.chunker import auto_chunk
    from src.ingestion.indexer import DocumentIndexer

    report_path = PROJECT_ROOT / "test_data" / "starvault_report.md"
    session_id = "benchmark-" + uuid.uuid4().hex[:8]
    document_id = "starvault-" + uuid.uuid4().hex[:8]
    persist_dir = Path(tempfile.gettempdir()) / ("adaptive-rag-benchmark-" + uuid.uuid4().hex[:8])
    collection = "benchmark_" + uuid.uuid4().hex[:8]

    chunks = auto_chunk(str(report_path))
    for i, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata.update({
            "session_id": session_id,
            "document_id": document_id,
            "source_name": "starvault_report.md",
            "chunk_index": i,
        })

    indexer = DocumentIndexer(collection_name=collection, persist_dir=persist_dir)
    indexed = await indexer.index_documents(chunks)
    retrieval_filter = {"session_id": session_id, "document_ids": [document_id]}
    return indexer, retrieval_filter, indexed, len(chunks)


def _patch_compare_for_index(indexer):
    from src.evaluation import compare
    from src.retrieval.adaptive import create_adaptive_chain as real_create_adaptive_chain
    from src.retrieval.single_step import SingleStepStrategy
    from src.types import AgentState, QueryComplexity
    from src.utils.token_manager import TokenBudget

    original_single = compare.SingleStepStrategy
    original_create = compare.create_adaptive_chain
    original_standard = compare._run_standard_rag
    original_adaptive = compare._run_adaptive_rag

    def scoped_single_step(*args, **kwargs):
        return SingleStepStrategy(indexer=indexer)

    def scoped_adaptive_chain(llm_client=None):
        adaptive, registry = real_create_adaptive_chain(llm_client=llm_client)
        registry.get("single_step")._indexer = indexer
        registry.get("multi_step")._single_step._indexer = indexer
        return adaptive, registry

    compare.SingleStepStrategy = scoped_single_step
    compare.create_adaptive_chain = scoped_adaptive_chain

    def chunk_ids(docs: list, limit: int) -> list:
        ids = []
        for doc in docs[:limit]:
            metadata = getattr(doc, "metadata", None) or {}
            ids.append(metadata.get("chunk_index", "?"))
        return ids

    async def scoped_run_standard(query, ground_truth, llm_client, retrieval_filter):
        t0 = time.time()
        strategy = scoped_single_step()
        agent_state = AgentState(query=query, complexity=QueryComplexity.MEDIUM)
        search_result = await strategy.retrieve(
            query,
            agent_state,
            retrieval_filter=retrieval_filter,
        )
        docs = search_result.documents
        contexts = [doc.content for doc in docs[:5]]
        answer = await llm_client.generate(
            messages=[{"role": "user", "content": query}],
            system_prompt=compare._build_rag_prompt(contexts),
            model_name=compare.get_settings().llm_simple_model,
        )
        scores, eval_error = await compare._safe_evaluate_ragas(
            query, answer, contexts, ground_truth,
        )
        return {
            "answer": answer,
            "time_ms": (time.time() - t0) * 1000,
            "model": compare.get_settings().llm_simple_model,
            "docs_count": len(docs),
            "tokens_est": len(answer) * 2 + sum(len(c) for c in contexts),
            "scores": scores,
            "eval_error": eval_error,
            "retrieved_sources": compare._sources(docs),
            "retrieved_document_ids": compare._document_ids(docs),
            "retrieved_chunk_ids": chunk_ids(docs, 5),
            "contexts_preview": contexts[:3],
        }

    async def scoped_run_adaptive(query, ground_truth, llm_client, retrieval_filter):
        t0 = time.time()
        adaptive, _ = scoped_adaptive_chain(llm_client=llm_client)
        adaptive_state = AgentState(query=query)
        adaptive_result = await adaptive.retrieve(
            query,
            adaptive_state,
            retrieval_filter=retrieval_filter,
        )
        docs = adaptive_result.documents
        contexts = [doc.content for doc in docs[:5]]
        model = TokenBudget.model_for_complexity(adaptive_state.complexity.value)
        answer = await llm_client.generate(
            messages=[{"role": "user", "content": query}],
            system_prompt=compare._build_rag_prompt(contexts),
            model_name=model,
        )
        scores, eval_error = await compare._safe_evaluate_ragas(
            query, answer, contexts, ground_truth,
        )
        return {
            "answer": answer,
            "time_ms": (time.time() - t0) * 1000,
            "model": model,
            "strategy": adaptive_state.selected_strategy.value,
            "complexity": adaptive_state.complexity.value,
            "docs_count": len(docs),
            "tokens_est": len(answer) * 2 + sum(len(c) for c in contexts),
            "scores": scores,
            "eval_error": eval_error,
            "retrieved_sources": compare._sources(docs),
            "retrieved_document_ids": compare._document_ids(docs),
            "retrieved_chunk_ids": chunk_ids(docs, 6),
            "contexts_preview": contexts[:5],
        }

    compare._run_standard_rag = scoped_run_standard
    compare._run_adaptive_rag = scoped_run_adaptive
    return compare, original_single, original_create, original_standard, original_adaptive


async def run_all_benchmarks() -> tuple[list[BenchmarkResult], dict]:
    from src.evaluation.compare import run_comparison

    indexer, retrieval_filter, indexed, total = await _prepare_scoped_index()
    metadata = {
        "indexed_chunks": indexed,
        "total_chunks": total,
        "retrieval_filter": retrieval_filter,
    }
    (
        compare_module,
        original_single,
        original_create,
        original_standard,
        original_adaptive,
    ) = _patch_compare_for_index(indexer)

    results: list[BenchmarkResult] = []
    try:
        for q in QUESTIONS:
            t0 = time.perf_counter()
            result = BenchmarkResult(q["id"], q["level"], q["question"])
            print(f"\n{'=' * 80}", flush=True)
            print(f"[{q['id']}] {q['level']} - {q['question']}", flush=True)
            print(f"{'=' * 80}", flush=True)
            try:
                comparison = await run_comparison(
                    query=q["question"],
                    ground_truth=q.get("ground_truth") or None,
                    retrieval_filter=retrieval_filter,
                )
                result.direct_answer = comparison.direct_answer
                result.standard_rag = comparison.standard_rag
                result.adaptive_rag = comparison.adaptive_rag
                result.conclusion = comparison.conclusion
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                result.traceback = traceback.format_exc()
                print(result.traceback, flush=True)
            result.duration_sec = time.perf_counter() - t0
            results.append(result)
            print(f"[{q['id']}] done in {result.duration_sec:.1f}s error={result.error}", flush=True)
    finally:
        compare_module.SingleStepStrategy = original_single
        compare_module.create_adaptive_chain = original_create
        compare_module._run_standard_rag = original_standard
        compare_module._run_adaptive_rag = original_adaptive

    return results, metadata


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test_data/benchmark_result.json")
    parser.add_argument("--csv-output", default="test_data/benchmark_summary.csv")
    parser.add_argument("--log-output", default="test_data/benchmark_console.log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.log_output, encoding="utf-8"),
        ],
    )

    print("=" * 80)
    print("Adaptive-RAG StarVault full benchmark")
    print(f"started_at={datetime.now().isoformat()}")
    print("=" * 80)

    results, metadata = await run_all_benchmarks()
    json_data = {
        "metadata": metadata,
        "results": format_results_json(results),
    }
    json_text = json.dumps(json_data, ensure_ascii=False, indent=2)
    csv_text = csv_summary(results)

    Path(args.output).write_text(json_text, encoding="utf-8")
    Path(args.csv_output).write_text(csv_text, encoding="utf-8")

    print("\n" + "=" * 80)
    print("CSV SUMMARY")
    print("=" * 80)
    print(csv_text)
    print("\n" + "=" * 80)
    print("FULL JSON")
    print("=" * 80)
    print(json_text)
    print("=" * 80)
    print(f"json_saved={args.output}")
    print(f"csv_saved={args.csv_output}")
    print(f"log_saved={args.log_output}")

    failed = [r for r in results if r.error]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
