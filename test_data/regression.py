"""
全类型回归验证脚本
验证所有文档类型的 chunking → keyword coverage → index readiness
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import auto_chunk
from langchain_core.documents import Document

test_dir = Path(__file__).resolve().parent

TEST_CASES = [
    {
        "type": "PDF",
        "file": "resume.pdf",
        "checks": {
            "min_chunks": 2,
            "keywords": ["ReflexRAG", "TriAgent", "Python", "LangChain", "ChromaDB"],
        },
        "qa_keywords": {
            "有几个项目经历": ["ReflexRAG", "TriAgent"],
            "个人技能有哪些": ["Python", "LangChain", "ChromaDB"],
            "教育背景": ["清华大学", "北京大学"],
        },
    },
    {
        "type": "Word",
        "file": "products.docx",
        "checks": {
            "min_chunks": 3,
            "keywords": ["智能客服", "数据分析平台", "自动化运维", "29999", "售后服务"],
        },
        "qa_keywords": {
            "有哪些产品": ["智能客服", "数据分析平台", "自动化运维"],
            "价格是多少": ["29999", "79999"],
            "售后服务包括什么": ["技术支持", "保修", "退换", "培训"],
        },
    },
    {
        "type": "Markdown",
        "file": "tech_comparison.md",
        "checks": {
            "min_chunks": 3,
            "keywords": ["微服务架构", "单体架构", "ChromaDB", "Qdrant", "gRPC"],
        },
        "qa_keywords": {
            "有哪些技术方案": ["微服务架构", "单体架构", "向量数据库"],
            "性能测试数据": ["Milvus", "ChromaDB", "Qdrant"],
            "代码示例": ["python", "grpc"],
        },
    },
    {
        "type": "TXT",
        "file": "article.txt",
        "checks": {
            "min_chunks": 1,
            "keywords": ["系统架构总览", "检索策略详解", "LangGraph", "Adaptive-RAG"],
        },
        "qa_keywords": {
            "文章有几个章节": ["系统架构总览", "检索策略详解", "技术实现要点"],
            "检索策略有哪些": ["简单", "BM25", "HyDE"],
            "用了哪些技术": ["LangGraph", "ChromaDB", "Streamlit"],
        },
    },
    {
        "type": "CSV",
        "file": "employees.csv",
        "checks": {
            "min_chunks": 1,
            "keywords": ["姓名", "部门", "研发部", "月薪", "45000"],
        },
        "qa_keywords": {
            "有哪些部门": ["研发部", "产品部", "设计部", "数据部", "质量部"],
            "谁是技术总监": ["王五"],
            "研发部有哪些人": ["张三", "王五", "周九"],
        },
    },
]


def run_regression():
    """Run all regression tests"""
    print("=" * 60)
    print("FULL REGRESSION TEST — All 5 Document Types")
    print("=" * 60)

    results = {}
    all_pass = True
    issues_found = []

    for case in TEST_CASES:
        doc_type = case["type"]
        file_path = test_dir / case["file"]
        print(f"\n{'='*60}")
        print(f"[{doc_type}] {file_path.name}")
        print(f"{'='*60}")

        if not file_path.exists():
            print(f"  FAIL: File not found")
            results[doc_type] = "FAIL"
            all_pass = False
            continue

        # Test 1: Chunking
        try:
            chunks = auto_chunk(str(file_path))
        except Exception as e:
            print(f"  FAIL: auto_chunk() raised {e}")
            results[doc_type] = "FAIL"
            all_pass = False
            issues_found.append(f"{doc_type}: chunking exception: {e}")
            continue

        # Test 2: Min chunk count
        min_chunks = case["checks"]["min_chunks"]
        if len(chunks) < min_chunks:
            print(f"  FAIL: {len(chunks)} chunks < min {min_chunks}")
            results[doc_type] = "FAIL"
            all_pass = False
            issues_found.append(f"{doc_type}: too few chunks ({len(chunks)} < {min_chunks})")
            continue
        print(f"  [PASS] Chunks: {len(chunks)} (min: {min_chunks})")

        # Test 3: All chunks are Document
        invalid = []
        for i, c in enumerate(chunks):
            if isinstance(c, str):
                invalid.append(f"chunk[{i}] is str")
            elif not hasattr(c, "page_content"):
                invalid.append(f"chunk[{i}] no page_content")
            elif not c.page_content.strip():
                invalid.append(f"chunk[{i}] empty content")
        if invalid:
            print(f"  FAIL: Invalid chunks: {invalid[:3]}")
            results[doc_type] = "FAIL"
            all_pass = False
            issues_found.extend(f"{doc_type}: {x}" for x in invalid[:3])
            continue
        print(f"  [PASS] All chunks are valid Documents")

        # Test 4: Keyword coverage
        all_text = " ".join(c.page_content for c in chunks)
        keywords = case["checks"]["keywords"]
        found_kw = [kw for kw in keywords if kw in all_text]
        missing_kw = [kw for kw in keywords if kw not in all_text]
        if missing_kw:
            print(f"  FAIL: Missing keywords: {missing_kw}")
            results[doc_type] = "FAIL"
            all_pass = False
            issues_found.append(f"{doc_type}: missing keywords: {missing_kw}")
            continue
        print(f"  [PASS] Keywords: {len(found_kw)}/{len(keywords)} all found")

        # Test 5: Metadata quality
        has_source = sum(1 for c in chunks if "source" in c.metadata)
        has_index = sum(1 for c in chunks if "chunk_index" in c.metadata)
        meta_ok = has_source == len(chunks) and has_index == len(chunks)
        if not meta_ok:
            print(f"  WARN: Metadata incomplete: source={has_source}/{len(chunks)}, chunk_index={has_index}/{len(chunks)}")
            issues_found.append(f"{doc_type}: metadata incomplete")
        else:
            print(f"  [PASS] Metadata complete: source + chunk_index on all chunks")

        # Test 6: QA keyword coverage (content is in chunks, retrievability depends on embedding)
        qa_results = []
        all_text = " ".join(c.page_content for c in chunks)
        for query, expected in case["qa_keywords"].items():
            found = [kw for kw in expected if kw in all_text]
            missing = [kw for kw in expected if kw not in all_text]
            status = "PASS" if not missing else "FAIL"
            qa_results.append((query, status, len(found), len(expected), missing))
            if missing:
                issues_found.append(f"{doc_type}: query '{query}' missing {missing} in chunks")

        qa_pass = sum(1 for _, s, _, _, _ in qa_results if s == "PASS")
        qa_total = len(qa_results)
        print(f"  [{'PASS' if qa_pass == qa_total else 'FAIL'}] QA coverage: {qa_pass}/{qa_total} queries have all keywords")
        for query, status, found, total, missing in qa_results:
            if missing:
                print(f"        '{query}': missing {missing}")

        results[doc_type] = "PASS" if qa_pass == qa_total else "WARN"

    # Final summary
    print(f"\n{'='*60}")
    print("REGRESSION SUMMARY")
    print(f"{'='*60}")
    pass_count = sum(1 for v in results.values() if v == "PASS")
    warn_count = sum(1 for v in results.values() if v == "WARN")
    fail_count = sum(1 for v in results.values() if v == "FAIL")

    for doc_type, status in results.items():
        flag = "PASS" if status == "PASS" else ("WARN" if status == "WARN" else "FAIL")
        print(f"  [{flag}] {doc_type}")

    print(f"\n  Total: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")

    if issues_found:
        print(f"\n  Issues ({len(issues_found)}):")
        for issue in issues_found:
            print(f"    - {issue}")

    return all_pass and fail_count == 0


if __name__ == "__main__":
    ok = run_regression()
    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL PASS' if ok else 'SOME FAILURES'}")
    print(f"{'='*60}")
    sys.exit(0 if ok else 1)
