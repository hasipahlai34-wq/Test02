"""Streamlit frontend for the Adaptive RAG demo."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import queue
import tempfile
import threading
import time
import uuid

import streamlit as st
from ui.utils import run_async


PAGE_DOCS = "文档管理"
PAGE_CHAT = "智能问答"
PAGE_COMPARE = "对比评估"
PAGE_MONITOR = "系统监控"
PAGE_MEMORY = "记忆管理"


if "_rag_session_id" not in st.session_state:
    st.session_state["_rag_session_id"] = str(uuid.uuid4())
if "_active_document_ids" not in st.session_state:
    st.session_state["_active_document_ids"] = []
if "_active_sources" not in st.session_state:
    st.session_state["_active_sources"] = []


def _stream_text(text: str):
    """Yield text in small chunks so Streamlit renders a visible typewriter stream."""
    for char in text:
        yield char
        time.sleep(0.005)


def _write_streaming_answer(query: str, session_id: str, retrieval_filter: dict | None):
    """Stream graph updates into Streamlit and return (answer, final_state)."""
    from src.graph.workflow import run_adaptive_rag_stream

    chunks: list[str] = []
    final_state: dict = {}
    item_queue: queue.Queue = queue.Queue()
    sentinel = object()

    async def consume_stream() -> None:
        try:
            async for event in run_adaptive_rag_stream(
                query,
                session_id=session_id,
                config={"configurable": {"thread_id": session_id}},
                retrieval_filter=retrieval_filter,
            ):
                for update in event.values():
                    if not isinstance(update, dict):
                        continue
                    final_state.update(update)
                    answer_stream = update.get("answer_stream")
                    if answer_stream:
                        async for chunk in answer_stream:
                            if chunk:
                                item_queue.put(chunk)
                        continue
                    answer = update.get("generated_answer")
                    if answer:
                        item_queue.put(("answer", answer))
            item_queue.put(sentinel)
        except Exception as e:
            item_queue.put(e)

    def runner() -> None:
        asyncio.run(consume_stream())

    def sync_chunks():
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        rendered_answer = ""
        while True:
            item = item_queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and item[0] == "answer"
            ):
                answer = item[1]
                if not isinstance(answer, str):
                    continue
                if answer.startswith(rendered_answer):
                    delta = answer[len(rendered_answer):]
                elif not rendered_answer:
                    delta = answer
                else:
                    delta = ""
                rendered_answer = answer
                for chunk in _stream_text(delta):
                    chunks.append(chunk)
                    yield chunk
                continue
            chunks.append(item)
            yield item
        thread.join(timeout=1)

    rendered = st.write_stream(sync_chunks())
    return rendered or "".join(chunks), final_state


def render_ragas_score(scores: dict | None, metric: str, label: str) -> None:
    """Render a RAGAS score without treating missing metrics as zero."""
    if scores is None:
        st.metric(label, "评估失败")
        return

    if metric not in scores:
        st.metric(label, "未评估")
        return

    try:
        st.metric(label, f"{float(scores[metric]):.3f}")
    except (TypeError, ValueError):
        st.metric(label, "无效")


def current_retrieval_filter() -> dict | None:
    document_ids = st.session_state.get("_active_document_ids") or []
    if not document_ids:
        return None
    return {
        "session_id": st.session_state["_rag_session_id"],
        "document_ids": document_ids,
    }


st.set_page_config(
    page_title="Adaptive RAG 自适应文档问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "reranker_warmup_triggered" not in st.session_state:
    st.session_state.reranker_warmup_triggered = True
    try:
        from src.retrieval.single_step import warmup_reranker

        warmup_reranker()
    except Exception:
        pass


st.sidebar.title("📚 Adaptive RAG")
st.sidebar.caption("基于 LangGraph 的自适应文档问答系统")

reranker_status = "not_started"
try:
    from src.retrieval.single_step import get_reranker_status

    reranker_status = get_reranker_status()
except Exception:
    pass

if reranker_status == "warming":
    st.sidebar.info("正在加载检索模型，首次提问可能稍慢。")
elif reranker_status == "failed":
    st.sidebar.warning("检索模型预热失败，将在首次提问时加载。")

page = st.sidebar.radio(
    "导航",
    [PAGE_DOCS, PAGE_CHAT, PAGE_COMPARE, PAGE_MONITOR, PAGE_MEMORY],
)

st.sidebar.divider()
st.sidebar.caption("技术栈：LangGraph + ChromaDB + Streamlit")
st.sidebar.caption("论文：Adaptive-RAG (NAACL 2024)")
st.sidebar.divider()
st.sidebar.caption("当前检索范围")
st.sidebar.text(f"会话：{st.session_state['_rag_session_id'][:8]}")
active_sources = st.session_state.get("_active_sources", [])
if active_sources:
    st.sidebar.caption(f"当前文档：{len(active_sources)}")
    for source in active_sources:
        st.sidebar.text(f"- {source}")
else:
    st.sidebar.warning("请先上传并处理文档。")


if page == PAGE_DOCS:
    st.title("文档管理")
    st.caption("上传文档 → 自动分块 → 写入索引 → 支持检索")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("上传文档")
        uploaded_file = st.file_uploader(
            "支持 PDF / Word / Markdown / TXT / CSV",
            type=["pdf", "docx", "md", "txt", "csv"],
            help="拖拽文件到此处，或点击选择文件。",
        )

        if uploaded_file:
            temp_dir = Path(tempfile.gettempdir()) / "adaptive_rag_uploads"
            temp_dir.mkdir(exist_ok=True)
            filepath = temp_dir / uploaded_file.name

            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"已上传：{uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

            if st.button("处理文档（加载 → 分块 → 索引）", type="primary"):
                progress = st.progress(0, text="准备处理文档...")
                status = st.empty()
                try:
                    from config.settings import get_settings

                    settings = get_settings()
                    if settings.embedding_provider == "local":
                        st.info("正在加载本地 Embedding 模型，首次使用可能需要 1-2 分钟。")

                    def _on_index_progress(done: int, total: int) -> None:
                        progress.progress(
                            50 + int(50 * done / max(total, 1)),
                            text=f"正在索引 chunk {done}/{total}",
                        )

                    async def _process():
                        from src.ingestion.chunker import auto_chunk
                        from src.ingestion.indexer import DocumentIndexer
                        from src.ingestion.loader import load_document

                        rag_session_id = st.session_state["_rag_session_id"]
                        document_id = str(uuid.uuid4())
                        source_name = uploaded_file.name

                        status.info("正在加载文档...")
                        docs = await load_document(filepath)
                        st.session_state["_raw_docs"] = docs
                        progress.progress(25, text=f"已加载 {len(docs)} 个原始片段")

                        status.info("正在分块...")
                        chunks = auto_chunk(str(filepath))
                        for i, chunk in enumerate(chunks):
                            chunk.metadata["session_id"] = rag_session_id
                            chunk.metadata["document_id"] = document_id
                            chunk.metadata["source"] = str(filepath)
                            chunk.metadata["source_name"] = source_name
                            chunk.metadata["chunk_index"] = i
                        st.session_state["_chunks"] = chunks
                        progress.progress(50, text=f"已生成 {len(chunks)} 个 chunk")

                        status.info("正在写入向量索引...")
                        indexer = DocumentIndexer()
                        await indexer.delete_by_session(rag_session_id)
                        count = await indexer.index_documents(
                            chunks,
                            progress_callback=_on_index_progress,
                        )
                        return docs, chunks, count, document_id, source_name

                    raw_docs, chunks, count, document_id, source_name = run_async(_process)
                    if chunks and count == 0:
                        st.error("索引失败：没有 chunk 写入向量库。")
                    else:
                        st.session_state["_active_document_ids"] = [document_id]
                        st.session_state["_active_sources"] = [source_name]
                        progress.progress(
                            100,
                            text=f"处理完成：{count}/{len(chunks)} 个 chunk 已索引",
                        )
                        st.success(
                            f"处理完成：{len(raw_docs)} 个原始片段 → "
                            f"{len(chunks)} 个分块 → {count} 个已索引"
                        )
                except Exception as e:
                    st.error(f"处理失败：{e}")

        st.divider()
        st.subheader("已索引文档")
        if st.button("清空当前会话索引"):
            try:
                async def _clear_session():
                    from src.ingestion.indexer import DocumentIndexer

                    idx = DocumentIndexer()
                    return await idx.delete_by_session(st.session_state["_rag_session_id"])

                deleted = run_async(_clear_session)
                st.session_state["_active_document_ids"] = []
                st.session_state["_active_sources"] = []
                st.session_state["_chunks"] = []
                st.success(f"已清空当前会话索引：{deleted} 个 chunk")
            except Exception as e:
                st.warning(f"清空当前会话索引失败：{e}")

        if st.button("刷新列表"):
            try:
                async def _list_sources():
                    from src.ingestion.indexer import DocumentIndexer

                    idx = DocumentIndexer()
                    return await idx.get_sources(), await idx.count()

                sources, total = run_async(_list_sources)
                st.metric("文档总数", len(sources))
                st.metric("Chunk 总数", total)
                for source in sources:
                    st.text(f"文件：{source}")
            except Exception as e:
                st.warning(f"获取索引状态失败：{e}")

    with col2:
        st.subheader("分块预览")
        chunks = st.session_state.get("_chunks", [])
        if chunks:
            st.caption(f"分块策略：自动选择｜共 {len(chunks)} 个分块")
            for i, chunk in enumerate(chunks[:10]):
                with st.expander(f"Chunk {i + 1} ({len(chunk.page_content)} 字符)", expanded=i < 3):
                    st.text(chunk.page_content[:500])
                    if len(chunk.page_content) > 500:
                        st.caption(f"... 还有 {len(chunk.page_content) - 500} 个字符")
        else:
            st.info("上传并处理文档后，分块结果会显示在这里。")


elif page == PAGE_CHAT:
    st.title("智能问答")
    st.caption("Adaptive-RAG：查询分类 → 动态路由 → 检索 → 生成 → 审核")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    col_chat, col_viz = st.columns([3, 2])

    with col_chat:
        st.subheader("对话")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("输入你的问题...")
        if query:
            retrieval_filter = current_retrieval_filter()
            if retrieval_filter is None:
                st.warning("请先在“文档管理”页上传并处理文档，再进行智能问答。")
                st.stop()

            st.session_state.chat_messages.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("正在分析问题复杂度..."):
                    try:
                        from src.graph.state import GraphState
                        from src.graph.workflow import build_adaptive_rag_graph

                        session_id = st.session_state["_rag_session_id"]
                        initial_state: GraphState = {
                            "query": query,
                            "session_id": session_id,
                            "retrieval_filter": retrieval_filter,
                        }

                        try:
                            answer, final_state = _write_streaming_answer(
                                query,
                                session_id,
                                retrieval_filter,
                            )
                        except Exception:
                            app = build_adaptive_rag_graph()
                            final_state = run_async(
                                app.ainvoke,
                                initial_state,
                                {"configurable": {"thread_id": session_id}},
                            )
                            answer = final_state.get("generated_answer", "回答生成失败。")
                            answer = st.write_stream(_stream_text(answer)) or answer

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": answer,
                        })

                        st.session_state["_last_complexity"] = final_state.get("complexity", "N/A")
                        st.session_state["_last_confidence"] = final_state.get("complexity_confidence", 0)
                        st.session_state["_last_strategy"] = final_state.get("selected_strategy", "N/A")
                        st.session_state["_last_docs"] = final_state.get("retrieved_docs", [])
                        st.session_state["_last_quality"] = final_state.get("quality_score", 0)
                        st.session_state["_last_hallucination"] = final_state.get("quality_passed", True)
                        st.session_state["_last_hyde"] = final_state.get("hyde_hypothesis", "")
                    except Exception as e:
                        st.error(f"执行失败：{e}")
                        import traceback

                        st.code(traceback.format_exc())

    with col_viz:
        st.subheader("检索过程可视化")

        with st.container(border=True):
            st.caption("查询复杂度")
            complexity = st.session_state.get("_last_complexity", "-")
            confidence = st.session_state.get("_last_confidence", 0)
            strategy = st.session_state.get("_last_strategy", "-")

            color_map = {"simple": "简单", "medium": "中等", "complex": "复杂"}
            st.metric("复杂度", f"{color_map.get(complexity, complexity)}")
            st.metric("置信度", f"{confidence:.2f}")
            st.metric("选择策略", strategy)

        with st.container(border=True):
            st.caption("HyDE 假设文档")
            hyde = st.session_state.get("_last_hyde", "")
            if hyde:
                st.text(hyde[:300] + ("..." if len(hyde) > 300 else ""))
            else:
                st.caption("未触发，或简单查询跳过。")

        with st.container(border=True):
            st.caption("检索结果（Top 3）")
            docs = st.session_state.get("_last_docs", [])
            if docs:
                for i, doc in enumerate(docs[:3], 1):
                    st.text(f"Doc {i}: {doc.content[:100]}... (得分：{doc.score:.2f})")
            else:
                st.caption("无检索，或直接回答模式。")

        with st.container(border=True):
            st.caption("安全与质量")
            quality = st.session_state.get("_last_quality", 0)
            passed = st.session_state.get("_last_hallucination", True)
            st.metric("质量得分", f"{quality:.2f}")
            st.metric("审核结果", "通过" if passed else "需要人工审核")


elif page == PAGE_COMPARE:
    st.title("对比评估")
    st.caption("同一查询并行运行三条路径，并用 RAGAS 指标量化对比。")

    eval_query = st.text_area(
        "输入测试查询",
        placeholder="例如：2024 年 Q3 营收增长的主要驱动因素是什么？",
    )
    ground_truth = st.text_area(
        "标准答案（可选）",
        placeholder="填写后会额外评估上下文精确度和召回率；不填写时这些指标显示为未评估。",
    )

    if st.button("执行三路对比", type="primary", disabled=not eval_query):
        retrieval_filter = current_retrieval_filter()
        if retrieval_filter is None:
            st.warning("请先在“文档管理”页上传并处理文档，再执行三路对比。")
            st.stop()

        with st.spinner("正在并行执行三条路径..."):
            try:
                from src.evaluation.compare import run_comparison

                result = run_async(
                    run_comparison,
                    eval_query,
                    ground_truth.strip() or None,
                    None,
                    retrieval_filter,
                )
                st.session_state["_compare_result"] = result
                st.success("对比完成。")
            except Exception as e:
                st.error(f"对比执行失败：{e}")

    result = st.session_state.get("_compare_result")
    if result:
        col1, col2, col3 = st.columns(3)

        direct = result.direct_answer
        rag = result.standard_rag
        adaptive = result.adaptive_rag

        with col1:
            st.subheader("直接回答")
            st.caption("无检索，仅 LLM")
            with st.container(border=True):
                st.metric("耗时", f"{direct.get('time_ms', 0):.0f}ms")
                st.metric("Token", direct.get("tokens_est", 0))
            with st.expander("查看答案"):
                st.text(direct.get("answer", "")[:500])

        with col2:
            st.subheader("标准 RAG")
            st.caption("单步 BM25 + Dense + Rerank")
            with st.container(border=True):
                scores = rag.get("scores")
                if rag.get("eval_error"):
                    st.warning(f"评估失败：{rag['eval_error']}")
                render_ragas_score(scores, "faithfulness", "忠实度")
                render_ragas_score(scores, "answer_relevancy", "相关性")
                render_ragas_score(scores, "context_precision", "精确度")
                st.metric("耗时", f"{rag.get('time_ms', 0):.0f}ms")
                st.caption("来源：" + (", ".join(rag.get("retrieved_sources") or []) or "未检索到"))
            with st.expander("查看答案"):
                st.text(rag.get("answer", "")[:500])

        with col3:
            st.subheader("自适应 RAG")
            st.caption(f"策略：{adaptive.get('strategy', 'N/A')}")
            with st.container(border=True):
                scores = adaptive.get("scores")
                if adaptive.get("eval_error"):
                    st.warning(f"评估失败：{adaptive['eval_error']}")
                render_ragas_score(scores, "faithfulness", "忠实度")
                render_ragas_score(scores, "answer_relevancy", "相关性")
                render_ragas_score(scores, "context_precision", "精确度")
                st.metric("耗时", f"{adaptive.get('time_ms', 0):.0f}ms")
                st.caption("来源：" + (", ".join(adaptive.get("retrieved_sources") or []) or "未检索到"))
            with st.expander("查看答案"):
                st.text(adaptive.get("answer", "")[:500])

        st.divider()
        st.markdown(result.conclusion)


elif page == PAGE_MONITOR:
    st.title("系统监控")
    st.caption("熔断器状态、Token 消耗、缓存统计和性能指标。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("熔断器状态")
        try:
            from src.safety.circuit_breaker import FrequencyCircuitBreaker, QualityCircuitBreaker

            qcb = QualityCircuitBreaker()
            fcb = FrequencyCircuitBreaker()

            with st.container(border=True):
                st.caption("质量熔断（滑动窗口）")
                qstats = qcb.stats()
                st.metric("状态", qstats["state"].upper())
                st.progress(
                    1 - float(qstats["window_failure_rate"].rstrip("%")) / 100,
                    text=f"失败率：{qstats['window_failure_rate']}",
                )
                st.caption(
                    f"窗口：{qstats['total_in_window']}/{qstats['window_size']} | "
                    f"阈值：{qstats['threshold']}"
                )

            with st.container(border=True):
                st.caption("频率熔断（令牌桶）")
                fstats = fcb.stats()
                st.metric("状态", fstats["state"].upper())
                st.metric("可用令牌", fstats["tokens_available"])
                st.caption(f"速率：{fstats['refill_rate']} | 上限：{fstats['max_tokens']}")
        except Exception as e:
            st.warning(f"熔断器状态获取失败：{e}")

    with col2:
        st.subheader("Token 消耗")
        try:
            from src.utils.observability import get_perf_tracker, get_token_tracker

            token_tracker = get_token_tracker()
            perf_tracker = get_perf_tracker()

            with st.container(border=True):
                tokens = token_tracker.stats
                by_step = token_tracker.by_step()
                st.metric("总 Token", tokens["total_tokens"])
                st.metric("请求次数", tokens["requests_count"])
                if by_step:
                    for step, count in sorted(by_step.items()):
                        st.metric(f"{step}", f"{count} tokens")
                else:
                    st.caption("暂无数据")

            with st.container(border=True):
                perf = perf_tracker.stats()
                if perf:
                    for name, stat in sorted(perf.items()):
                        st.metric(
                            f"{name}",
                            f"avg={stat['avg_ms']:.0f}ms (x{stat['count']})",
                        )
                else:
                    st.caption("暂无数据")
        except Exception as e:
            st.warning(f"监控数据获取失败：{e}")

    st.subheader("缓存统计")
    try:
        st.caption("语义缓存：精确匹配 + 余弦相似度")
        st.info("缓存统计可通过 SemanticCache().stats() 获取。")
    except Exception as e:
        st.warning(f"缓存统计获取失败：{e}")


elif page == PAGE_MEMORY:
    st.title("记忆管理")
    st.caption("三级记忆系统：短期会话、中期偏好、长期知识。")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("短期记忆")
        st.caption("当前会话内的多轮上下文")
        with st.container(border=True):
            chat_msgs = st.session_state.get("chat_messages", [])
            if chat_msgs:
                st.metric("对话轮数", len(chat_msgs) // 2)
                for i, msg in enumerate(chat_msgs[-6:]):
                    role = "用户" if msg["role"] == "user" else "助手"
                    st.text(f"[{i + 1}] {role}: {msg['content'][:80]}...")
            else:
                st.caption("暂无对话。请到智能问答页面提问。")
        st.caption("配置：最多保留 10 轮；窗口管理：滑动窗口。")

    with col2:
        st.subheader("中期记忆")
        st.caption("跨会话的用户偏好学习")
        with st.container(border=True):
            try:
                from src.memory.medium_term import MediumTermMemory

                mm = MediumTermMemory()
                prefs = mm.get_preferences("default")

                st.metric("偏好风格", prefs.preferred_style)
                if prefs.preferred_topics:
                    for topic in prefs.preferred_topics:
                        st.text(f"- {topic}")
                else:
                    st.caption("尚未学习到偏好。")

                if prefs.frequently_asked:
                    st.caption("常用查询：")
                    for q in prefs.frequently_asked[:3]:
                        st.text(f"- {q}")
            except Exception as e:
                st.caption(f"数据库未初始化：{e}")
        st.caption("存储：SQLite；更新：会话结束后提取。")

    with col3:
        st.subheader("长期记忆")
        st.caption("持久化知识摘要")
        with st.container(border=True):
            try:
                from src.memory.long_term import LongTermMemory

                lm = LongTermMemory()
                if lm.count > 0:
                    st.metric("记忆条目", lm.count)
                    for entry in lm.get_all()[:3]:
                        with st.expander(f"记忆（重要性 {entry.importance:.2f}）"):
                            st.text(entry.content[:300])
                else:
                    st.caption("暂无条目。")
            except Exception as e:
                st.caption(f"未初始化：{e}")
        st.caption("触发：上下文超过阈值；存储：向量数据库。")


st.sidebar.divider()
st.sidebar.info(
    "**启动命令**：`streamlit run ui/app.py` 或 `python main.py ui`\n\n"
    "**技术论文**：\n"
    "- Adaptive-RAG (NAACL 2024)\n"
    "- HyDE (Gao et al., 2022)\n\n"
    "**WeKnora 参考**：Tencent/WeKnora"
)
