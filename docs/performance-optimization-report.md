# Performance Optimization Report

Date: 2026-07-05

## Summary

Implemented the high-impact workflow optimizations that can be verified without live LLM benchmarking:

- Quick rule-based query classification before LLM classification.
- Exact cache hit shortcut in the graph workflow before retrieval/generation.
- Simple-query review shortcut.
- RAGAS online evaluation moved off the user-facing critical path.
- Comparison mode paths now run concurrently with `asyncio.gather`.

Full regression result:

```text
python -m pytest tests/ -v
68 passed, 1 warning in 5.99s
```

The warning is a pytest cache write warning for `.pytest_cache`, not a test failure.

## Optimization Table

| Optimization | Before | After | LLM calls reduced / latency impact |
|---|---:|---:|---|
| Simple query classification | Always calls classifier LLM | Regex rules return `simple` / `complex` for obvious queries | `-1` classifier call on rule hits |
| Cache hit shortcut | Repeated queries still run graph path | Exact cache hit returns cached answer before retrieval/generation | Skips retrieval, generation, review, RAGAS, guard on exact hit |
| RAGAS async | Workflow waits for RAGAS | Workflow returns with `ragas_eval_error="pending_async"` and logs later | RAGAS no longer blocks answer |
| Safety parallelization | Output guard remains after generation | Not changed in this pass | Not implemented; adding input guard would add a new LLM call to this workflow |
| Review conditionalization | Review node reachable for all paths | `complexity == "simple"` returns default pass without reviewer LLM | Up to `-1` reviewer call for simple queries with docs |
| Streaming output | Backend stream helper exists | UI still uses graph `ainvoke` final answer | Not completed; needs UI path refactor |
| Comparison parallelization | Direct, standard RAG, adaptive RAG were serial | Three paths run via `asyncio.gather` | Total wall time approaches slowest path instead of sum |

## Notes

- Cache integration uses exact hits in the request path. Semantic embedding lookup was intentionally not initialized on misses because it blocked first-query tests and would hurt the target latency. The existing `SemanticCache.lookup()` semantic API remains available outside the critical path.
- Moving RAGAS async means RAGAS scores no longer synchronously influence the same request's HITL gate. Quality review and safety guard still do.
- `src/evaluation/compare.py` was rewritten to remove corrupted encoded strings and keep the same public `run_comparison()` return shape.

## Follow-Up Completion

Date: 2026-07-07

Completed the remaining items:

- Semantic cache fallback now calls `SemanticCache.lookup()` after exact miss, only when semantic entries exist, with `asyncio.wait_for(..., timeout=2.0)`.
- Input safety now runs concurrently with `single_step` / `multi_step` retrieval. If unsafe returns before retrieval finishes, retrieval is cancelled and the graph returns a blocked answer.
- Streamlit Q&A now attempts `run_adaptive_rag_stream()` through `st.write_stream()` and falls back to the previous `ainvoke` path on error.

Timing verification used a repeatable local end-to-end workflow harness with fixed mock delays: 120ms per LLM call and 180ms retrieval. This avoids live API variance and makes call-count effects visible.

| Scenario | Before perceived time | After perceived time | LLM calls |
|---|:---:|:---:|:---:|
| Simple greeting | ~0.40s modeled | 0.279s measured | 2 |
| Cache hit | ~0.70s first-run equivalent | 0.126s measured | 1 |
| Document Q&A | ~0.82s modeled serial safety | 0.699s measured | 5 |

The cache-hit path still performs classification before cache lookup in the current graph, then skips retrieval/generation/review/guard on hit.
