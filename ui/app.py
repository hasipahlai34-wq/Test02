"""Streamlit UI for the Adaptive RAG application."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running via `streamlit run ui/app.py`
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import asyncio
import tempfile
import uuid
from typing import Any

import streamlit as st


COMPLEXITY_LABELS = {
    "simple": "简单问题",
    "medium": "标准 RAG",
    "complex": "复杂推理",
}

STRATEGY_LABELS = {
    "no_retrieval": "无需检索",
    "single_step": "单步检索",
    "multi_step": "多步检索",
}


st.set_page_config(
    page_title="Adaptive RAG 文档问答系统",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
.main .block-container {
    max-width: 1280px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.35rem;
}
.ar-hero {
    border: 1px solid #d8dee9;
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    background: #ffffff;
    margin-bottom: 1rem;
}
.ar-hero h1 {
    font-size: 1.75rem;
    margin: 0 0 .35rem 0;
    letter-spacing: 0;
}
.ar-hero p {
    margin: 0;
    color: #4b5563;
    line-height: 1.55;
}
.ar-panel {
    border: 1px solid #d8dee9;
    border-radius: 8px;
    padding: 1rem;
    background: #ffffff;
}
.ar-muted {
    color: #6b7280;
    font-size: .9rem;
}
.ar-pill {
    display: inline-block;
    border: 1px solid #c9d3df;
    border-radius: 999px;
    padding: .12rem .55rem;
    margin: .1rem .25rem .1rem 0;
    color: #334155;
    background: #f8fafc;
    font-size: .82rem;
}
.ar-source {
    border-left: 3px solid #64748b;
    padding: .45rem .7rem;
    margin: .45rem 0;
    background: #f8fafc;
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def run_coro(coro):
    """Run an async coroutine from Streamlit's synchronous script context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@st.cache_resource(show_spinner=False)
def get_indexer():
    from src.ingestion.indexer import DocumentIndexer

    return DocumentIndexer()


def get_session_id() -> str:
    if "rag_session_id" not in st.session_state:
        st.session_state.rag_session_id = f"streamlit-{uuid.uuid4()}"
    return st.session_state.rag_session_id


def reset_chat() -> None:
    st.session_state.rag_session_id = f"streamlit-{uuid.uuid4()}"
    st.session_state.chat_messages = []
    st.session_state.last_state = None


def value_or_dash(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def bool_label(value: Any) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "-"


def get_doc_content(doc: Any) -> str:
    if hasattr(doc, "page_content"):
        return str(doc.page_content)
    if hasattr(doc, "content"):
        return str(doc.content)
    if isinstance(doc, dict):
        return str(doc.get("page_content") or doc.get("content") or "")
    return str(doc)


def get_doc_metadata(doc: Any) -> dict[str, Any]:
    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        return doc.metadata
    if isinstance(doc, dict) and isinstance(doc.get("metadata"), dict):
        return doc["metadata"]
    return {}


def source_name(doc: Any) -> str:
    metadata = get_doc_metadata(doc)
    return str(
        metadata.get("source_name")
        or metadata.get("source_file")
        or metadata.get("source")
        or getattr(doc, "source", "")
        or "未知来源"
    )


def truncate(text: str, limit: int = 600) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def refresh_index_stats() -> None:
    try:
        st.session_state.indexed_chunks = run_coro(get_indexer().count())
        st.session_state.sources = run_coro(get_indexer().get_sources())
    except Exception as exc:
        st.session_state.index_stats_error = str(exc)


def render_sidebar() -> None:
    from config.settings import get_settings

    settings = get_settings()
    st.sidebar.title("Adaptive RAG")
    st.sidebar.caption("文档问答 · 自适应路由 · LangGraph")

    st.sidebar.markdown("### 会话")
    st.sidebar.code(get_session_id(), language=None)
    if st.sidebar.button("新建会话", use_container_width=True):
        reset_chat()
        st.rerun()
    if st.sidebar.button("清空对话", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.last_state = None
        st.rerun()

    st.sidebar.markdown("### 知识库")
    if st.sidebar.button("刷新索引状态", use_container_width=True):
        refresh_index_stats()
    chunks = st.session_state.get("indexed_chunks")
    sources = st.session_state.get("sources")
    st.sidebar.metric("已索引 chunks", value_or_dash(chunks))
    st.sidebar.metric("来源数量", value_or_dash(len(sources) if isinstance(sources, list) else None))

    st.sidebar.markdown("### Langfuse")
    has_keys = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
    st.sidebar.write("状态：" + ("已启用" if settings.langfuse_enabled and has_keys else "未启用"))
    st.sidebar.write("地址：" + settings.langfuse_base_url)
    st.sidebar.caption("密钥只用于后端上报，不在页面展示。")


async def ingest_file(uploaded_file, strategy: str, chunk_size: int) -> dict[str, int | str]:
    from src.ingestion.chunker import ChunkingStrategy, auto_chunk, chunk_documents
    from src.ingestion.loader import load_document

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    raw_docs = await load_document(tmp_path)
    if strategy == "auto":
        chunks = auto_chunk(tmp_path)
    else:
        chunks = chunk_documents(raw_docs, strategy=ChunkingStrategy(strategy), chunk_size=chunk_size)

    count = await get_indexer().index_documents(chunks)
    return {
        "文件": uploaded_file.name,
        "原始片段": len(raw_docs),
        "分块数量": len(chunks),
        "写入数量": count,
    }


def render_header() -> None:
    st.markdown(
        """
        <div class="ar-hero">
          <h1>Adaptive RAG 文档问答系统</h1>
          <p>根据问题复杂度自动选择无需检索、单步 RAG 或多步 RAG，并在生成后执行质量审核、RAGAS 评估和安全检查。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    last_state = st.session_state.get("last_state") or {}
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("问题复杂度", COMPLEXITY_LABELS.get(last_state.get("complexity"), value_or_dash(last_state.get("complexity"))))
    col_b.metric("检索策略", STRATEGY_LABELS.get(last_state.get("selected_strategy"), value_or_dash(last_state.get("selected_strategy"))))
    col_c.metric("命中文档", value_or_dash(last_state.get("search_count")))
    col_d.metric("质量分", value_or_dash(last_state.get("quality_score")))


def render_trace_summary(state: dict[str, Any]) -> None:
    st.markdown("#### 本次链路")
    if not state:
        st.info("提交一个问题后，这里会展示路由、检索、审核和安全状态。")
        return

    rows = [
        {"项目": "复杂度", "值": COMPLEXITY_LABELS.get(state.get("complexity"), value_or_dash(state.get("complexity")))},
        {"项目": "分类置信度", "值": value_or_dash(state.get("complexity_confidence"))},
        {"项目": "检索策略", "值": STRATEGY_LABELS.get(state.get("selected_strategy"), value_or_dash(state.get("selected_strategy")))},
        {"项目": "检索命中", "值": value_or_dash(state.get("search_count"))},
        {"项目": "缓存命中", "值": bool_label(state.get("cache_hit") or state.get("from_cache"))},
        {"项目": "质量通过", "值": bool_label(state.get("quality_passed"))},
        {"项目": "安全风险", "值": value_or_dash(state.get("safety_risk_level"))},
        {"项目": "人工审核", "值": value_or_dash(state.get("hitl_status"))},
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    summary = state.get("search_result_summary")
    if summary:
        with st.expander("检索摘要", expanded=False):
            st.write(summary)

    ragas_scores = state.get("ragas_scores")
    if ragas_scores:
        with st.expander("RAGAS 评分", expanded=False):
            st.json(ragas_scores)


def render_retrieved_docs(state: dict[str, Any]) -> None:
    docs = state.get("retrieved_docs") or []
    st.markdown("#### 参考来源")
    if not docs:
        st.caption("本次回答没有返回检索文档。")
        return

    for index, doc in enumerate(docs[:8], start=1):
        title = f"{index}. {source_name(doc)}"
        with st.expander(title, expanded=index <= 2):
            metadata = get_doc_metadata(doc)
            if metadata:
                tags = []
                for key in ["document_id", "page", "chunk_id", "score", "source"]:
                    if metadata.get(key) is not None:
                        tags.append(f"<span class='ar-pill'>{key}: {metadata.get(key)}</span>")
                if tags:
                    st.markdown("".join(tags), unsafe_allow_html=True)
            st.markdown(f"<div class='ar-source'>{truncate(get_doc_content(doc), 900)}</div>", unsafe_allow_html=True)


def render_chat_tab() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    left, right = st.columns([2.15, 1], gap="large")
    with left:
        st.subheader("问答")
        st.caption("输入问题后，系统会自动判断是否需要检索以及采用哪种 RAG 路径。")

        if not st.session_state.chat_messages:
            samples = [
                "这个项目的 Adaptive RAG 流程是怎样的？",
                "请总结已索引文档的主要内容。",
                "哪些问题适合走多步检索？",
            ]
            sample_cols = st.columns(len(samples))
            for col, sample in zip(sample_cols, samples):
                if col.button(sample, use_container_width=True):
                    st.session_state.pending_query = sample
                    st.rerun()

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input("输入你的问题")
        query = query or st.session_state.pop("pending_query", None)
        if query:
            st.session_state.chat_messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("正在分析问题、检索文档并生成回答..."):
                    from src.graph.workflow import run_adaptive_rag

                    final_state = run_coro(run_adaptive_rag(query, session_id=get_session_id()))
                    answer = (
                        final_state.get("final_response")
                        or final_state.get("generated_answer")
                        or "没有生成可展示的回答。"
                    )
                    st.markdown(answer)

            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.session_state.last_state = final_state
            st.rerun()

    with right:
        state = st.session_state.get("last_state") or {}
        render_trace_summary(state)
        render_retrieved_docs(state)


def render_documents_tab() -> None:
    st.subheader("文档摄入")
    st.caption("上传本地文档后写入向量索引，后续问答会基于这些内容检索。")

    uploaded_files = st.file_uploader(
        "上传文档",
        type=["txt", "md", "pdf", "docx", "csv"],
        accept_multiple_files=True,
    )
    col_strategy, col_size = st.columns([1, 1])
    with col_strategy:
        strategy = st.selectbox(
            "分块策略",
            ["auto", "recursive", "markdown", "semantic"],
            format_func=lambda value: {
                "auto": "自动选择",
                "recursive": "递归字符分块",
                "markdown": "Markdown 结构分块",
                "semantic": "语义分块",
            }.get(value, value),
        )
    with col_size:
        chunk_size = st.number_input("分块大小", min_value=200, max_value=4000, value=800, step=100)

    if st.button("开始处理文档", type="primary", disabled=not uploaded_files):
        results: list[dict[str, Any]] = []
        progress = st.progress(0)
        for index, uploaded_file in enumerate(uploaded_files or [], start=1):
            with st.status(f"正在处理 {uploaded_file.name}", expanded=False):
                results.append(run_coro(ingest_file(uploaded_file, strategy, int(chunk_size))))
            progress.progress(index / len(uploaded_files))
        st.success("文档索引完成")
        st.dataframe(results, use_container_width=True, hide_index=True)
        refresh_index_stats()

    st.divider()
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        if st.button("刷新来源列表", use_container_width=True):
            refresh_index_stats()
    with col_info:
        error = st.session_state.get("index_stats_error")
        if error:
            st.warning(error)

    sources = st.session_state.get("sources")
    if sources is not None:
        st.metric("已索引 chunks", st.session_state.get("indexed_chunks", 0))
        st.dataframe([{"文档来源": source} for source in sources], use_container_width=True, hide_index=True)


def render_evaluation_tab() -> None:
    st.subheader("三路对比评估")
    st.caption("对同一问题同时运行直接回答、标准 RAG 和 Adaptive RAG，比较耗时、召回和评估分。")

    query = st.text_area("评估问题", height=110, placeholder="输入一个需要比较的测试问题")
    ground_truth = st.text_area("参考答案，可选", height=80, placeholder="如果有标准答案，可以填在这里用于 RAGAS 评估")

    if st.button("运行对比", type="primary", disabled=not query.strip()):
        from src.evaluation.compare import run_comparison

        with st.spinner("正在运行三路对比..."):
            result = run_coro(run_comparison(query.strip(), ground_truth=ground_truth.strip() or None))

        st.markdown("#### 评估结论")
        st.write(result.conclusion)

        rows = []
        for name, payload in [
            ("直接回答", result.direct_answer),
            ("标准 RAG", result.standard_rag),
            ("Adaptive RAG", result.adaptive_rag),
        ]:
            scores = payload.get("scores") or {}
            rows.append({
                "路径": name,
                "耗时 ms": round(payload.get("time_ms") or 0, 1),
                "模型": payload.get("model"),
                "文档数": payload.get("docs_count"),
                "Token 估算": payload.get("tokens_est"),
                "Faithfulness": scores.get("faithfulness"),
                "Answer Relevancy": scores.get("answer_relevancy"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        answer_tabs = st.tabs(["直接回答", "标准 RAG", "Adaptive RAG"])
        for tab, payload in zip(answer_tabs, [result.direct_answer, result.standard_rag, result.adaptive_rag]):
            with tab:
                st.markdown(payload.get("answer") or "-")
                if payload.get("retrieved_sources"):
                    st.caption("来源：" + ", ".join(payload["retrieved_sources"]))
                if payload.get("eval_error"):
                    st.warning(payload["eval_error"])


def render_diagnostics_tab() -> None:
    st.subheader("运行诊断")
    from config.settings import get_settings
    from src.utils.observability import get_perf_tracker, get_token_tracker

    settings = get_settings()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Langfuse", "已启用" if settings.langfuse_enabled and settings.langfuse_public_key and settings.langfuse_secret_key else "未启用")
    col_b.metric("Base URL", settings.langfuse_base_url)
    col_c.metric("会话", get_session_id().split("-", 1)[0])

    diag_tabs = st.tabs(["性能", "Token", "最近状态"])
    with diag_tabs[0]:
        st.json(get_perf_tracker().stats())
    with diag_tabs[1]:
        st.json(get_token_tracker().stats)
    with diag_tabs[2]:
        state = st.session_state.get("last_state") or {}
        st.json({key: value for key, value in state.items() if key not in {"retrieved_docs", "messages"}})


def main() -> None:
    render_sidebar()
    render_header()

    tab_chat, tab_docs, tab_eval, tab_diag = st.tabs(["问答", "文档", "评估", "诊断"])
    with tab_chat:
        render_chat_tab()
    with tab_docs:
        render_documents_tab()
    with tab_eval:
        render_evaluation_tab()
    with tab_diag:
        render_diagnostics_tab()


main()