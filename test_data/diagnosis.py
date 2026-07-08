"""
全类型文档问答检索链路诊断脚本
用法: python test_data/diagnosis.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ================================================================
# Phase 1: Chunking Diagnostics
# ================================================================
def diagnose_chunking():
    """Test chunking for all 5 document types"""
    from src.ingestion.chunker import auto_chunk

    test_dir = Path(__file__).resolve().parent
    test_files = {
        "PDF": test_dir / "resume.pdf",
        "Word": test_dir / "products.docx",
        "Markdown": test_dir / "tech_comparison.md",
        "TXT": test_dir / "article.txt",
        "CSV": test_dir / "employees.csv",
    }

    # Expected keywords for each document type (must appear in at least one chunk)
    expected_keywords = {
        "PDF": ["ReflexRAG", "TriAgent", "清华大学", "个人技能", "Python"],
        "Word": ["智能客服", "数据分析平台", "自动化运维", "29999", "售后服务", "退换政策"],
        "Markdown": ["微服务架构", "单体架构", "ChromaDB", "Qdrant", "gRPC", "性能测试数据"],
        "TXT": ["系统架构总览", "检索策略详解", "技术实现要点", "LangGraph", "Adaptive-RAG"],
        "CSV": ["姓名", "部门", "研发部", "张三", "王五", "月薪", "架构师", "45000"],
    }

    chunking_results = {}

    for doc_type, file_path in test_files.items():
        print(f"\n{'='*60}")
        print(f"[Phase 1] Chunking: {doc_type} — {file_path.name}")
        print(f"{'='*60}")

        if not file_path.exists():
            print(f"  FAIL: File not found: {file_path}")
            chunking_results[doc_type] = {"status": "FAIL", "error": "File not found"}
            continue

        try:
            chunks = auto_chunk(str(file_path))
        except Exception as e:
            print(f"  FAIL: Chunking exception: {e}")
            import traceback
            traceback.print_exc()
            chunking_results[doc_type] = {"status": "FAIL", "error": str(e)}
            continue

        # Check 1: chunk count > 0
        if not chunks:
            print(f"  FAIL: 0 chunks produced")
            chunking_results[doc_type] = {"status": "FAIL", "error": "0 chunks"}
            continue

        print(f"  Chunks: {len(chunks)}")

        # Check 2: all elements are Documents with page_content
        all_valid = True
        type_issues = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                type_issues.append(f"chunk[{i}] is str, not Document")
                all_valid = False
            elif not hasattr(chunk, "page_content"):
                type_issues.append(f"chunk[{i}] has no page_content, type={type(chunk).__name__}")
                all_valid = False
            elif not chunk.page_content or not chunk.page_content.strip():
                type_issues.append(f"chunk[{i}] has empty page_content")
                all_valid = False

        if not all_valid:
            for issue in type_issues[:5]:
                print(f"  TYPE ERROR: {issue}")
            chunking_results[doc_type] = {"status": "FAIL", "error": "Invalid chunk types"}
            continue

        # Check 3: keyword coverage
        all_text = " ".join(chunk.page_content for chunk in chunks)
        keywords = expected_keywords.get(doc_type, [])
        found = [kw for kw in keywords if kw in all_text]
        missing = [kw for kw in keywords if kw not in all_text]

        print(f"  Keywords found: {len(found)}/{len(keywords)}")
        if missing:
            print(f"  MISSING: {missing}")
        if found:
            print(f"  FOUND: {found}")

        # Check 4: inspect first few chunks
        print(f"  Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            content_preview = chunk.page_content[:120].replace("\n", "\\n")
            meta_keys = list(chunk.metadata.keys()) if hasattr(chunk, "metadata") else []
            print(f"    [{i}] len={len(chunk.page_content)}, meta={meta_keys}")
            print(f"        {content_preview}...")

        # Check 5: metadata quality
        has_source = sum(1 for c in chunks if "source" in c.metadata)
        has_chunk_type = sum(1 for c in chunks if "chunk_type" in c.metadata)
        print(f"  Metadata: source={has_source}/{len(chunks)}, chunk_type={has_chunk_type}/{len(chunks)}")

        chunking_results[doc_type] = {
            "status": "PASS" if not missing else "WARN",
            "chunks": len(chunks),
            "missing_keywords": missing,
            "found_keywords": found,
        }

    # Summary
    print(f"\n{'='*60}")
    print("Phase 1 Summary: Chunking")
    print(f"{'='*60}")
    all_pass = True
    for doc_type, result in chunking_results.items():
        status = result["status"]
        if status == "FAIL":
            all_pass = False
        chunks = result.get("chunks", 0)
        missing = result.get("missing_keywords", [])
        flag = "PASS" if status == "PASS" else ("WARN" if status == "WARN" else "FAIL")
        extra = f" (missing: {missing})" if missing else ""
        print(f"  [{flag}] {doc_type}: {chunks} chunks{extra}")

    return chunking_results, all_pass


# ================================================================
# Phase 2: Retrieval Pipeline Analysis (code-level, no async needed)
# ================================================================
def diagnose_pipeline():
    """Analyze the retrieval pipeline for potential issues"""
    print(f"\n{'='*60}")
    print("Phase 2: Retrieval Pipeline Analysis")
    print(f"{'='*60}")

    issues = []

    # Issue 1: SingleStepStrategy top_k chain
    print("\n[2.1] SingleStepStrategy top_k chain:")
    print("  bm25_top_k=5, dense_top_k=10, rerank_top_k=5")
    print("  RRF fusion: all results fused")
    print("  Rerank input: fused[:dense_top_k] = fused[:10]")
    print("  Rerank output: top rerank_top_k = 5")
    print("  => Final results: max 5 documents")
    issues.append({
        "type": "C1",
        "severity": "P1",
        "location": "src/retrieval/single_step.py:327",
        "issue": "rerank_top_k=5 may be too low. Complex documents need more context.",
        "fix": "Consider increasing rerank_top_k to 8-10 or making it configurable per query complexity."
    })

    # Issue 2: Build_context truncation
    print("\n[2.2] _build_context truncation (generator.py:40-88):")
    print("  max_tokens=4000 (~8000 chars)")
    print("  Per-doc limit: 800 chars")
    print("  => Large documents may have tail content truncated")
    issues.append({
        "type": "D3",
        "severity": "P1",
        "location": "src/agents/generator.py:70",
        "issue": "Each document truncated to 800 chars. Long chunks lose tail content.",
        "fix": "Consider per-doc limit of 1500-2000 chars or dynamic based on total context."
    })

    # Issue 3: System prompt completeness
    print("\n[2.3] System prompt completeness:")
    print("  Prompt does NOT explicitly say 'list every item, do not miss any'")
    print("  medium complexity: 'provide structured answer with citations'")
    print("  complex complexity: 'show reasoning process, synthesize'")
    issues.append({
        "type": "D1",
        "severity": "P1",
        "location": "config/prompts/system_prompt.yaml",
        "issue": "System prompt doesn't require exhaustive listing of all matching items.",
        "fix": "Add rule: 'If the user asks about multiple items, list ALL of them. Count them first.'"
    })

    # Issue 4: Rerank threshold
    print("\n[2.4] Rerank threshold:")
    print("  threshold=0.3 may filter out relevant but low-scoring docs")
    print("  Combined with rerank_top_k=5, can be very restrictive")
    issues.append({
        "type": "C4",
        "severity": "P2",
        "location": "src/retrieval/single_step.py:97,108",
        "issue": "Rerank threshold 0.3 + top_k=5 may filter too aggressively.",
        "fix": "Lower threshold to 0.1 or remove threshold filtering for small result sets."
    })

    # Issue 5: CSV special handling
    print("\n[2.5] CSV retrieval analysis:")
    print("  CSV chunks use col=value format with headers")
    print("  Aggregation queries (highest salary, count) need special handling")
    print("  Current vector search can't do aggregation")
    issues.append({
        "type": "E",
        "severity": "P2",
        "location": "src/ingestion/chunker.py:751-815",
        "issue": "CSV aggregation queries (max, min, count) cannot be answered by vector search.",
        "fix": "Add pandas-based aggregation fallback for CSV data queries."
    })

    # Issue 6: Vector search threshold
    print("\n[2.6] Vector search threshold (settings.py):")
    print("  retrieval_threshold=0.5 (settings.py:140)")
    print("  This filters out results with similarity < 0.5")
    print("  For some query-document pairs, this may be too strict")
    issues.append({
        "type": "C1",
        "severity": "P2",
        "location": "config/settings.py:140",
        "issue": "retrieval_threshold=0.5 may be too strict for short or diverse queries.",
        "fix": "Lower to 0.3 or make it query-dependent."
    })

    print(f"\n  Found {len(issues)} potential issues:")
    for i, issue in enumerate(issues):
        print(f"  [{issue['severity']}] {issue['type']}: {issue['issue']}")

    return issues


# ================================================================
# Phase 3: Content-level Keyword Coverage
# ================================================================
def diagnose_keyword_coverage():
    """Deep dive: for each document type, check if key info is in retrievable chunks"""
    from src.ingestion.chunker import auto_chunk

    test_dir = Path(__file__).resolve().parent
    test_files = {
        "PDF": test_dir / "resume.pdf",
        "Word": test_dir / "products.docx",
        "Markdown": test_dir / "tech_comparison.md",
        "TXT": test_dir / "article.txt",
        "CSV": test_dir / "employees.csv",
    }

    # Test queries and the expected information they should retrieve
    test_queries = {
        "PDF": [
            ("项目经历有哪些", ["ReflexRAG", "TriAgent"]),
            ("个人技能有哪些", ["Python", "LangChain", "ChromaDB"]),
            ("教育背景", ["清华大学", "北京大学"]),
        ],
        "Word": [
            ("有哪些产品", ["智能客服", "数据分析平台", "自动化运维"]),
            ("价格是多少", ["29999", "79999", "19999"]),
            ("售后服务包括什么", ["技术支持", "保修", "退换", "培训"]),
        ],
        "Markdown": [
            ("有哪些技术方案对比", ["微服务架构", "单体架构", "向量数据库"]),
            ("性能测试数据是什么", ["Milvus", "Qdrant", "ChromaDB"]),
            ("代码示例是什么语言", ["python", "gRPC", "async"]),
        ],
        "TXT": [
            ("文章有几个章节", ["系统架构总览", "检索策略详解", "技术实现要点"]),
            ("检索策略有哪些", ["简单", "中等", "复杂", "BM25", "HyDE"]),
            ("用了哪些技术", ["LangGraph", "ChromaDB", "Streamlit", "WeKnora"]),
        ],
        "CSV": [
            ("有哪些部门", ["研发部", "产品部", "设计部", "数据部", "质量部", "项目管理部"]),
            ("谁是技术总监", ["王五"]),
            ("研发部有哪些人", ["张三", "王五", "周九"]),
        ],
    }

    print(f"\n{'='*60}")
    print("Phase 3: Keyword Coverage Analysis")
    print(f"{'='*60}")

    coverage_results = {}

    for doc_type, file_path in test_files.items():
        print(f"\n[3] {doc_type}:")
        chunks = auto_chunk(str(file_path))
        all_text = " ".join(chunk.page_content for chunk in chunks)

        doc_results = []
        for query, expected in test_queries.get(doc_type, []):
            found = [kw for kw in expected if kw in all_text]
            missing = [kw for kw in expected if kw not in all_text]
            coverage = len(found) / len(expected) * 100 if expected else 100
            status = "PASS" if not missing else "FAIL"
            doc_results.append({
                "query": query,
                "status": status,
                "found": found,
                "missing": missing,
                "coverage": coverage,
            })
            print(f"    [{status}] '{query}': {len(found)}/{len(expected)} keywords, coverage={coverage:.0f}%")
            if missing:
                print(f"           Missing: {missing}")

        coverage_results[doc_type] = doc_results

    return coverage_results


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE-RAG: Full Document Type Diagnosis")
    print("=" * 60)

    # Phase 1: Chunking
    chunk_results, chunk_all_pass = diagnose_chunking()

    # Phase 2: Pipeline analysis
    pipeline_issues = diagnose_pipeline()

    # Phase 3: Keyword coverage
    coverage_results = diagnose_keyword_coverage()

    # Final Summary
    print(f"\n{'='*60}")
    print("FINAL DIAGNOSIS SUMMARY")
    print(f"{'='*60}")

    print("\n[Chunking]")
    for doc_type, result in chunk_results.items():
        status = result["status"]
        flag = "PASS" if status == "PASS" else ("WARN" if status == "WARN" else "FAIL")
        print(f"  [{flag}] {doc_type}")

    print(f"\n[Pipeline Issues]")
    for issue in pipeline_issues:
        print(f"  [{issue['severity']}] {issue['type']}: {issue['issue'][:80]}...")

    print(f"\n[Coverage]")
    for doc_type, results in coverage_results.items():
        all_ok = all(r["status"] == "PASS" for r in results)
        flag = "PASS" if all_ok else "FAIL"
        print(f"  [{flag}] {doc_type}: {sum(1 for r in results if r['status']=='PASS')}/{len(results)} queries pass")
