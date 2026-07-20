"""
# ============================================================
# 单步检索策略 (SingleStepStrategy)
# ← WeKnora: chat_pipeline/ 整个 RAG Pipeline
#   - search.go: CHUNK_SEARCH + CHUNK_SEARCH_PARALLEL (扇出搜索)
#   - rerank.go: CHUNK_RERANK (Cross-encoder 重排序)
#   - merge.go: CHUNK_MERGE (多源结果去重融合)
#   - filter_top_k.go: FILTER_TOP_K (截断取 TopK)
#
#   我们实现: BM25 关键词检索 + Dense 向量检索 → RRF 融合 → Rerank
# ============================================================

本模块实现标准单步 RAG 检索管道:
1. BM25 关键词检索 (← grep_chunks.go)
2. Dense 向量检索 (← knowledge_search.go)
3. RRF (Reciprocal Rank Fusion) 结果融合 (← merge.go)
4. Rerank 重排序 (← rerank.go)
5. 取 TopK 截断 (← filter_top_k.go)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional

from langchain_core.documents import Document as LCDocument

from src.retrieval.base import RetrievalStrategy
from src.types import (
    AgentState,
    Document,
    MatchType,
    SearchResult,
    RetrievalStrategy as StrategyType,
)
from config.settings import get_settings
from src.models.embeddings import EmbeddingModel
from src.ingestion.indexer import DocumentIndexer, get_document_indexer, get_index_generation
from src.retrieval.query_intent import QueryIntent, classify_query_intent
from src.utils.observability import langfuse_observation

logger = logging.getLogger(__name__)


def _extract_number(value: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def _clean_table_cell(value: str) -> str:
    return re.sub(r"[*`]", "", str(value)).strip()


def _parse_markdown_tables(text: str) -> list[list[dict[str, str]]]:
    """Parse simple Markdown pipe tables into row dictionaries."""
    tables: list[list[dict[str, str]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|") or "|" not in line[1:]:
            i += 1
            continue
        if i + 1 >= len(lines):
            i += 1
            continue
        separator = lines[i + 1].strip()
        if not separator.startswith("|") or not re.search(r"[-:]{3,}", separator):
            i += 1
            continue

        headers = [_clean_table_cell(cell) for cell in line.strip("|").split("|")]
        rows: list[dict[str, str]] = []
        i += 2
        while i < len(lines):
            row_line = lines[i].strip()
            if not row_line.startswith("|") or "|" not in row_line[1:]:
                break
            cells = [_clean_table_cell(cell) for cell in row_line.strip("|").split("|")]
            if len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
            i += 1
        if rows:
            tables.append(rows)
        continue
    return tables


def _find_column(headers: list[str], keywords: list[str]) -> str | None:
    for keyword in keywords:
        for header in headers:
            if keyword in header:
                return header
    return None


def calculate_markdown_table_aggregation(query: str, contexts: list[str]) -> str | None:
    """Return deterministic aggregate results for Markdown budget tables when possible."""
    if not contexts:
        return None
    if not re.search(r"(总预算|总支出|剩余|结余|合计|最多|最少)", query):
        return None

    for context in contexts:
        for table in _parse_markdown_tables(context):
            headers = list(table[0].keys()) if table else []
            project_col = _find_column(headers, ["项目"])
            budget_col = _find_column(headers, ["预算"])
            q1_col = _find_column(headers, ["Q1"])
            q2_col = _find_column(headers, ["Q2"])
            remaining_col = _find_column(headers, ["剩余", "结余", "余额"])
            if not all([project_col, budget_col, q1_col, q2_col, remaining_col]):
                if len(headers) >= 5 and any("Q1" in header for header in headers):
                    project_col, budget_col, q1_col, q2_col, remaining_col = headers[:5]
            if not all([project_col, budget_col, q1_col, q2_col, remaining_col]):
                continue

            project_rows = []
            for row in table:
                project = row.get(project_col or "", "")
                if "合计" in project:
                    continue
                budget = _extract_number(row.get(budget_col or "", ""))
                q1 = _extract_number(row.get(q1_col or "", ""))
                q2 = _extract_number(row.get(q2_col or "", ""))
                remaining = _extract_number(row.get(remaining_col or "", ""))
                if None in (budget, q1, q2, remaining):
                    continue
                project_rows.append({
                    "project": project,
                    "budget": budget,
                    "q1": q1,
                    "q2": q2,
                    "remaining": remaining,
                })

            if not project_rows:
                continue

            total_budget = sum(float(row["budget"]) for row in project_rows)
            total_spend = sum(float(row["q1"]) + float(row["q2"]) for row in project_rows)
            max_remaining = max(project_rows, key=lambda row: float(row["remaining"]))
            detail_lines = [
                (
                    f"- {row['project']}: 预算{row['budget']:.0f}万元, "
                    f"Q1支出{row['q1']:.0f}万元, Q2支出{row['q2']:.0f}万元, "
                    f"剩余{row['remaining']:.0f}万元"
                )
                for row in project_rows
            ]
            return (
                "[Markdown表格确定性计算结果]\n"
                + "\n".join(detail_lines)
                + "\n"
                f"总预算: {total_budget:.0f}万元\n"
                f"总支出: {total_spend:.0f}万元\n"
                f"剩余预算最多: {max_remaining['project']}, "
                f"{float(max_remaining['remaining']):.0f}万元"
            )

    return None


def _metadata_matches_filter(metadata: dict, retrieval_filter: dict | None) -> bool:
    """Return whether a metadata dict belongs to the active retrieval scope."""
    if not retrieval_filter:
        return True

    session_id = retrieval_filter.get("session_id")
    if session_id and metadata.get("session_id") != session_id:
        return False

    document_ids = retrieval_filter.get("document_ids")
    if document_ids and metadata.get("document_id") not in set(document_ids):
        return False

    sources = retrieval_filter.get("sources")
    if sources and metadata.get("source") not in set(sources):
        return False

    return True


def _to_chroma_filter(retrieval_filter: dict | None) -> dict | None:
    """Convert the active retrieval scope into a Chroma metadata filter."""
    if not retrieval_filter:
        return None

    clauses = []
    session_id = retrieval_filter.get("session_id")
    if session_id:
        clauses.append({"session_id": session_id})

    document_ids = retrieval_filter.get("document_ids") or []
    if len(document_ids) == 1:
        clauses.append({"document_id": document_ids[0]})
    elif len(document_ids) > 1:
        clauses.append({"document_id": {"$in": list(document_ids)}})

    sources = retrieval_filter.get("sources") or []
    if len(sources) == 1:
        clauses.append({"source": sources[0]})
    elif len(sources) > 1:
        clauses.append({"source": {"$in": list(sources)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

# ★ C5 修复: SingleStepStrategy 模块级单例 (避免 BM25 索引每次重建)
_single_step_instance: Optional["SingleStepStrategy"] = None
_single_step_lock: Optional[asyncio.Lock] = None


def _get_single_step_lock() -> asyncio.Lock:
    global _single_step_lock
    if _single_step_lock is None:
        _single_step_lock = asyncio.Lock()
    return _single_step_lock


async def get_single_step() -> "SingleStepStrategy":
    """
    ★ 获取 SingleStepStrategy 单例 (线程安全双重检查锁)

    面试可讲:
    "BM25 索引和 Cross-encoder 模型加载成本高 (100MB+ 模型文件),
    每次请求重建会导致 1-5s 额外延迟。我使用单例模式 + 双重检查锁
    确保整个进程生命周期内只构建一次索引。"
    """
    global _single_step_instance
    if _single_step_instance is None:
        async with _get_single_step_lock():
            if _single_step_instance is None:
                _single_step_instance = SingleStepStrategy()
                await _single_step_instance._ensure_bm25()  # 预热 BM25 索引
                logger.info("SingleStepStrategy 单例已创建 (BM25 索引已预热)")
    return _single_step_instance


class SingleStepStrategy(RetrievalStrategy):
    """
    单步检索策略
    ← WeKnora: chat_pipeline/ Pipeline "rag_stream" =
                LOAD_HISTORY → QUERY_UNDERSTAND → CHUNK_SEARCH_PARALLEL →
                CHUNK_RERANK → CHUNK_MERGE → FILTER_TOP_K →
                INTO_CHAT_MESSAGE → CHAT_COMPLETION_STREAM

    面试可讲:
    "我实现了标准的 RAG 检索管道:
    BM25 做关键词召回，Dense 做语义召回，
    用 RRF 算法融合两种结果，最后用 Cross-encoder 做精排。
    这比单一的向量检索在召回率和准确率上都有明显提升。"
    """

    def __init__(
        self,
        indexer: Optional[DocumentIndexer] = None,
        rerank_model: Optional[object] = None,
        bm25_top_k: int | None = None,
        dense_top_k: int | None = None,
        rerank_top_k: int | None = None,
        rerank_threshold: float | None = None,
    ):
        super().__init__(name="单步检索 (BM25 + Dense + Rerank)")
        settings = get_settings()
        self.strategy_type = StrategyType.SINGLE_STEP
        self._indexer = indexer or get_document_indexer()
        self._reranker: Optional[object] = None  # 懒加载
        self._rerank_provider = settings.rerank_provider.lower()
        self._rerank_model_name = settings.rerank_model

        self.bm25_top_k = bm25_top_k if bm25_top_k is not None else settings.bm25_top_k
        self.dense_top_k = dense_top_k if dense_top_k is not None else settings.dense_top_k
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else settings.rerank_top_k
        self.rerank_threshold = (
            rerank_threshold if rerank_threshold is not None else settings.rerank_threshold
        )
        self.rerank_candidate_top_k = max(self.rerank_top_k, settings.rerank_candidate_top_k)

        # BM25 语料库 (懒构建)
        self._bm25_corpus: list[str] = []
        self._bm25 = None
        self._bm25_generation = -1

    # ----------------------------------------------------------------
    # BM25 关键词检索 (← WeKnora: grep_chunks.go)
    # ----------------------------------------------------------------

    async def _ensure_bm25(self) -> None:
        """懒构建 BM25 索引"""
        current_generation = get_index_generation()
        if self._bm25 is not None and self._bm25_generation == current_generation:
            return

        from rank_bm25 import BM25Okapi

        # 从 ChromaDB 获取所有文档内容构建 BM25 语料库
        await self._indexer._ensure_initialized()
        results = self._indexer.get_all_documents()
        if results:
            self._bm25_corpus = [doc["content"] for doc in results]
            tokenized = [self._tokenize(doc) for doc in self._bm25_corpus]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_generation = current_generation
            logger.info("BM25 索引已构建: %d 文档", len(self._bm25_corpus))
        else:
            self._bm25_corpus = []
            self._bm25 = None
            self._bm25_generation = current_generation

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文和英文的分词"""
        import re
        # 简单分词: 按空格和标点分割，保留中文字符和英文单词
        tokens = re.findall(r'[一-鿿]|[a-zA-Z]+|\d+', text.lower())
        return [t for t in tokens if len(t) > 0]

    async def _bm25_search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_filter: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        """
        BM25 关键词检索
        ← WeKnora: grep_chunks.go — 使用 PostgreSQL ts_rank 的全文搜索
           我们使用 rank-bm25 库实现 Python 原生 BM25

        Returns:
            [(文档内容, BM25得分), ...]
        """
        with langfuse_observation(
            name="retrieval.bm25",
            as_type="retriever",
            input={"query": query, "top_k": top_k, "scoped": bool(retrieval_filter)},
        ) as observation:
            if retrieval_filter:
                from rank_bm25 import BM25Okapi

                await self._indexer._ensure_initialized()
                indexed_docs = [
                    doc for doc in self._indexer.get_all_documents()
                    if _metadata_matches_filter(doc.get("metadata") or {}, retrieval_filter)
                ]
                if not indexed_docs:
                    if observation is not None:
                        observation.update(output={"results": 0, "reason": "empty_scope"})
                    return []
                corpus = [doc["content"] for doc in indexed_docs]
                tokenized = [self._tokenize(doc) for doc in corpus]
                bm25 = BM25Okapi(tokenized)
            else:
                await self._ensure_bm25()
                if not self._bm25 or not self._bm25_corpus:
                    if observation is not None:
                        observation.update(output={"results": 0, "reason": "empty_index"})
                    return []
                indexed_docs = [
                    {"content": content, "metadata": {}}
                    for content in self._bm25_corpus
                ]
                corpus = self._bm25_corpus
                bm25 = self._bm25

            tokenized_query = self._tokenize(query)
            scores = bm25.get_scores(tokenized_query)

            # 排序取 TopK
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            results = []
            for idx, score in indexed_scores[:top_k]:
                if score > 0:
                    metadata = indexed_docs[idx].get("metadata") or {}
                    results.append((corpus[idx], float(score), metadata))

            if observation is not None:
                observation.update(output={"results": len(results)})
            logger.debug("BM25: query='%s...' → %d results", query[:50], len(results))
            return results

    # ----------------------------------------------------------------
    # Dense 向量检索 (← WeKnora: knowledge_search.go)
    # ----------------------------------------------------------------

    async def _dense_search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_filter: dict | None = None,
    ) -> list[tuple[LCDocument, float]]:
        """
        Dense 向量语义检索
        ← WeKnora: knowledge_search.go → vectorstore.Search()
        """
        with langfuse_observation(
            name="retrieval.dense",
            as_type="retriever",
            input={"query": query, "top_k": top_k, "scoped": bool(retrieval_filter)},
        ) as observation:
            results = await self._indexer.search(
                query,
                top_k=top_k,
                filter_dict=_to_chroma_filter(retrieval_filter),
            )
            if observation is not None:
                observation.update(output={"results": len(results)})
            return results

    # ----------------------------------------------------------------
    # RRF 融合 (← WeKnora: merge.go)
    # ----------------------------------------------------------------

    def _rrf_fusion(
        self,
        bm25_results: list[tuple[str, float, dict]],
        dense_results: list[tuple[LCDocument, float]],
        k: int = 60,
    ) -> list[tuple[str, float, dict]]:
        """
        Reciprocal Rank Fusion (RRF) 多源结果融合
        ← WeKnora: merge.go — 合并多知识库 + BM25 + Dense 结果

        RRF 公式: score(d) = Σ 1 / (k + rank_i(d))
        其中 k 是平滑参数 (通常设为 60)

        这个算法的优点是:
        - 不需要归一化不同来源的分数
        - 对排名位置敏感 (排在前面的结果权重更高)
        - 简单高效

        Args:
            bm25_results: BM25 结果 [(content, score), ...]
            dense_results: Dense 结果 [(LCDocument, score), ...]
            k: RRF 平滑参数

        Returns:
            [(content, rrf_score), ...] 按 RRF 得分降序排列
        """
        rrf_scores: dict[str, float] = {}
        content_map: dict[str, str] = {}
        metadata_map: dict[str, dict] = {}

        # BM25 的 RRF 得分
        for rank, (content, _, metadata) in enumerate(bm25_results, start=1):
            key = content[:100]  # 用前100字符作为去重key
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank)
            content_map[key] = content
            metadata_map[key] = dict(metadata or {})

        # Dense 的 RRF 得分
        for rank, (doc, _) in enumerate(dense_results, start=1):
            key = doc.page_content[:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = doc.page_content
            if key not in metadata_map:
                metadata_map[key] = dict(doc.metadata or {})

        # 按 RRF 得分排序
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(content_map[key], score, metadata_map.get(key, {})) for key, score in sorted_items]

    # ----------------------------------------------------------------
    # Rerank 重排序 (← WeKnora: rerank.go)
    # ----------------------------------------------------------------

    async def _ensure_reranker(self) -> None:
        """懒加载 Reranker 模型"""
        if self._reranker is not None:
            return
        if self._rerank_provider != "local":
            return

        try:
            from sentence_transformers import CrossEncoder
            from src.models.embeddings import _hf_model_cache_exists

            kwargs = {}
            if _hf_model_cache_exists(self._rerank_model_name):
                kwargs["local_files_only"] = True
                logger.info("Reranker: using local HuggingFace cache for %s", self._rerank_model_name)
            self._reranker = CrossEncoder(self._rerank_model_name, **kwargs)
            logger.info("Reranker 已加载: %s", self._rerank_model_name)
        except Exception as e:
            logger.warning("Reranker 加载失败 (%s)，将跳过重排序", e)
            self._reranker = None

    async def _dashscope_rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int,
        threshold: float,
    ) -> list[tuple[str, float]]:
        """Rerank candidates with DashScope qwen3-rerank."""
        import httpx

        settings = get_settings()
        api_key = settings.dashscope_api_key or settings.rerank_api_key or settings.embedding_api_key
        if not api_key:
            raise RuntimeError("RERANK_API_KEY or DASHSCOPE_API_KEY is required")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error = ""
        async with httpx.AsyncClient(timeout=settings.rerank_timeout) as client:
            for url in self._dashscope_rerank_urls(settings.rerank_base_url):
                payload = self._dashscope_rerank_payload(
                    url,
                    query=query,
                    documents=documents,
                    top_k=top_k,
                )
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code < 400:
                    data = response.json()
                    if not data.get("code"):
                        break
                    last_error = (
                        f"{url}: {data.get('code')} {data.get('message', '')}".strip()
                    )
                    continue

                detail = response.text
                try:
                    body = response.json()
                    detail = f"{body.get('code', '')} {body.get('message', '')}".strip()
                except Exception:
                    pass
                last_error = f"{url}: HTTP {response.status_code} {detail}"
            else:
                raise RuntimeError(f"DashScope rerank failed: {last_error}")

        raw_results = data.get("results")
        if raw_results is None:
            raw_results = (data.get("output") or {}).get("results") or []

        scored: list[tuple[str, float]] = []
        for item in raw_results:
            try:
                idx = int(item.get("index"))
                score = float(item.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(documents) and score >= threshold:
                scored.append((documents[idx], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _dashscope_rerank_payload(
        self,
        url: str,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> dict:
        """Build payload for DashScope compatible or native rerank endpoints."""
        settings = get_settings()
        top_n = min(top_k, len(documents))

        if "/compatible-" in url:
            payload = {
                "model": self._rerank_model_name,
                "documents": documents,
                "query": query,
                "top_n": top_n,
            }
            if settings.rerank_instruct:
                payload["instruct"] = settings.rerank_instruct
            return payload

        payload = {
            "model": self._rerank_model_name,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False,
            },
        }
        if settings.rerank_instruct:
            payload["parameters"]["instruct"] = settings.rerank_instruct
        return payload

    @staticmethod
    def _dashscope_rerank_urls(configured_url: str) -> list[str]:
        """Return rerank endpoint candidates for public and workspace DashScope hosts."""
        raw = configured_url.rstrip("/")
        candidates = [raw]

        replacements = []
        if raw.endswith("/compatible-mode/v1"):
            replacements.extend([
                raw + "/reranks",
                raw.replace("/compatible-mode/v1", "/compatible-api/v1/reranks"),
                raw.replace("/compatible-mode/v1", "/api/v1/services/rerank/text-rerank/text-rerank"),
            ])
        elif raw.endswith("/compatible-api/v1"):
            replacements.append(raw + "/reranks")
        elif raw.endswith("/api/v1"):
            replacements.append(raw + "/services/rerank/text-rerank/text-rerank")
        elif "/compatible-api/v1/reranks" in raw:
            replacements.extend([
                raw.replace("/compatible-api/v1/reranks", "/compatible-mode/v1/reranks"),
                raw.replace("/compatible-api/v1/reranks", "/api/v1/services/rerank/text-rerank/text-rerank"),
            ])
        elif "/compatible-mode/v1/reranks" in raw:
            replacements.extend([
                raw.replace("/compatible-mode/v1/reranks", "/compatible-api/v1/reranks"),
                raw.replace("/compatible-mode/v1/reranks", "/api/v1/services/rerank/text-rerank/text-rerank"),
            ])

        for url in replacements:
            if url not in candidates:
                candidates.append(url)
        return candidates

    async def _rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[tuple[str, float]]:
        """
        Cross-encoder 重排序
        ← WeKnora: rerank.go → CrossEncoderModel.Rerank()

        与 Bi-encoder (向量检索) 的区别:
        - Bi-encoder: query 和 doc 分别编码，用余弦相似度比较 (快但粗糙)
        - Cross-encoder: query + doc 一起输入模型，输出相关性得分 (慢但精确)
        - Rerank 是两阶段检索的关键: 粗排 (Bi-encoder) → 精排 (Cross-encoder)

        Args:
            query: 用户查询
            documents: 候选文档内容列表
            top_k: 保留的 TopK 数量
            threshold: 最低得分阈值

        Returns:
            [(文档内容, rerank得分), ...]
        """
        if not documents:
            return []

        with langfuse_observation(
            name="retrieval.rerank",
            as_type="retriever",
            input={
                "query": query,
                "candidates": len(documents),
                "top_k": top_k,
                "threshold": threshold,
            },
        ) as observation:
            settings = get_settings()
            if not settings.rerank_enabled:
                result = [(doc, 0.0) for doc in documents[:top_k]]
                if observation is not None:
                    observation.update(output={"results": len(result), "enabled": False})
                return result

            if self._rerank_provider == "dashscope":
                try:
                    reranked = await self._dashscope_rerank(query, documents, top_k, threshold)
                    if reranked:
                        logger.debug(
                            "DashScope rerank: %d candidates → %d results",
                            len(documents), len(reranked),
                        )
                        if observation is not None:
                            observation.update(output={"results": len(reranked), "provider": "dashscope"})
                        return reranked
                except Exception as e:
                    logger.warning("DashScope rerank failed; using fused order: %s", e)
                result = [(doc, 0.0) for doc in documents[:top_k]]
                if observation is not None:
                    observation.update(output={
                        "results": len(result),
                        "provider": "dashscope",
                        "fallback": True,
                    })
                return result

            await self._ensure_reranker()

            if not self._reranker:
                result = [(doc, 0.0) for doc in documents[:top_k]]
                if observation is not None:
                    observation.update(output={"results": len(result), "fallback": True})
                return result

            # 构建 query-doc pairs
            pairs = [[query, doc] for doc in documents]
            scores = self._reranker.predict(pairs)

            # 排序 + 过滤 + 截断
            scored = list(zip(documents, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            filtered = [(doc, float(score)) for doc, score in scored if score >= threshold]

            result = filtered[:top_k]
            if observation is not None:
                observation.update(output={"results": len(result), "provider": self._rerank_provider})
            logger.debug(
                "Rerank: %d candidates → %d results (top_k=%d, threshold=%.2f)",
                len(documents), len(filtered), top_k, threshold,
            )
            return result

    # ----------------------------------------------------------------
    # 主检索入口
    # ----------------------------------------------------------------

    async def _structure_scope_search(
        self,
        query: str,
        retrieval_filter: dict | None,
    ) -> SearchResult | None:
        """Return outline/structure documents for global scoped questions."""
        if not retrieval_filter:
            return None

        started = time.perf_counter()
        intent = classify_query_intent(query)
        if intent not in {
            QueryIntent.GLOBAL_COUNT,
            QueryIntent.GLOBAL_LIST,
            QueryIntent.TABLE_AGGREGATION,
            QueryIntent.SUMMARY,
            QueryIntent.COMPARISON,
        }:
            return None

        await self._indexer._ensure_initialized()
        indexed_docs = [
            doc for doc in self._indexer.get_all_documents()
            if _metadata_matches_filter(doc.get("metadata") or {}, retrieval_filter)
        ]
        if not indexed_docs:
            return None

        prefer_content_chunks = (
            intent in {
                QueryIntent.TABLE_AGGREGATION,
                QueryIntent.SUMMARY,
                QueryIntent.COMPARISON,
            }
            or bool(re.search(
                r"(分别|描述|介绍|详情|技术栈|时间|亮点|职责|内容|怎么|如何)",
                query,
            ))
        )

        def priority(item: dict) -> tuple[int, int]:
            metadata = item.get("metadata") or {}
            chunk_type = str(metadata.get("chunk_type") or "")
            element_type = str(metadata.get("element_type") or "")
            content = str(item.get("content") or "")
            if chunk_type == "document_outline" or element_type == "outline":
                return (0, -len(content))
            if prefer_content_chunks:
                if element_type in {"section", "table", "row_group"}:
                    return (1, -len(content))
                if element_type == "heading":
                    return (2, -len(content))
            else:
                if element_type == "heading":
                    return (1, -len(content))
                if element_type in {"section", "table", "row_group"}:
                    return (2, -len(content))
            return (3, -len(content))

        structural = sorted(
            [
                doc for doc in indexed_docs
                if (doc.get("metadata") or {}).get("chunk_type") == "document_outline"
                or (doc.get("metadata") or {}).get("element_type") in {"outline", "heading", "section", "table", "row_group"}
            ],
            key=priority,
        )
        if not structural:
            return None

        selected = structural[:18]
        documents: list[Document] = []
        for rank, item in enumerate(selected):
            metadata = dict(item.get("metadata") or {})
            typed_metadata = {str(k): str(v) for k, v in metadata.items() if v is not None}
            try:
                chunk_index = int(metadata.get("chunk_index") or rank)
            except (TypeError, ValueError):
                chunk_index = rank
            documents.append(Document(
                content=str(item.get("content") or ""),
                score=max(0.0, 1.0 - rank * 0.02),
                match_type=MatchType.KEYWORD,
                source=str(metadata.get("source") or metadata.get("upload_filename") or ""),
                source_path=str(metadata.get("source") or ""),
                chunk_index=chunk_index,
                metadata={**typed_metadata, "query_intent": intent.value, "retrieval_scope": "structure"},
            ))

        with langfuse_observation(
            name="retrieval.structure_scope",
            as_type="retriever",
            input={"query": query, "intent": intent.value, "scope_docs": len(indexed_docs)},
        ) as observation:
            output = {
                "intent": intent.value,
                "selected": len(documents),
                "scope_docs": len(indexed_docs),
            }
            if observation is not None:
                observation.update(output=output)
            logger.info(
                "Structure scoped retrieval: intent=%s selected=%d scope_docs=%d",
                intent.value,
                len(documents),
                len(indexed_docs),
            )
            return SearchResult(
                query=query,
                documents=documents,
                strategy=self.strategy_type,
                total_found=len(documents),
                search_time_ms=(time.perf_counter() - started) * 1000,
            )

    async def retrieve(self, query: str, state: AgentState | None = None, **kwargs) -> SearchResult:
        """
        执行单步检索管道: BM25 + Dense → RRF → Rerank → TopK
        这是标准 RAG 检索的完整实现

        面试可讲:
        "两阶段检索策略: 第一阶段用 Bi-encoder 快速召回候选
        (BM25 做稀疏召回，Dense 做稠密召回，RRF 融合)，
        第二阶段用 Cross-encoder 做精排。
        这样兼顾了速度 (Bi-encoder 可以预先算好向量) 和精度
        (Cross-encoder 做 joint encoding 更准确)。"
        """
        import time
        start_time = time.perf_counter()
        if state is None:
            state = AgentState(query=query)
        retrieval_filter = kwargs.get("retrieval_filter")
        requested_top_k = kwargs.get("top_k")

        structure_result = await self._structure_scope_search(query, retrieval_filter)
        if structure_result is not None:
            return structure_result

        # Step 1: 并行执行 BM25 和 Dense 检索
        bm25_task = self._bm25_search(
            query,
            top_k=self.bm25_top_k,
            retrieval_filter=retrieval_filter,
        )
        dense_task = self._dense_search(
            query,
            top_k=self.dense_top_k,
            retrieval_filter=retrieval_filter,
        )

        bm25_results, dense_results = await asyncio.gather(bm25_task, dense_task)

        # Step 2: RRF 多源融合 (← WeKnora: merge.go)
        fused = self._rrf_fusion(bm25_results, dense_results)

        # Step 3: Rerank 精排 (← WeKnora: rerank.go)
        if fused:
            # Keep cloud rerank bounded; it is paid per token and should only
            # refine the strongest fused candidates.
            rerank_pool_size = min(len(fused), self.rerank_candidate_top_k)
            fused_metadata = {content[:100]: metadata for content, _, metadata in fused}
            fused_contents = [content for content, _, _ in fused[:rerank_pool_size]]
            reranked = await self._rerank(
                query,
                fused_contents,
                top_k=self.rerank_top_k,
                threshold=self.rerank_threshold,
            )
        else:
            reranked = []
            fused_metadata = {}

        # Step 4: 组装结果
        documents = []
        for i, (content, score) in enumerate(reranked):
            metadata = dict(fused_metadata.get(content[:100], {}))
            metadata["rerank_score"] = str(score)
            chunk_index = metadata.get("chunk_index", i)
            try:
                chunk_index = int(chunk_index)
            except (TypeError, ValueError):
                chunk_index = i
            typed_metadata = {
                str(key): str(value)
                for key, value in metadata.items()
                if value is not None
            }
            documents.append(Document(
                content=content,
                score=score,
                match_type=MatchType.HYBRID,
                source=str(metadata.get("source", "")),
                source_path=str(metadata.get("source", "")),
                chunk_index=chunk_index,
                metadata=typed_metadata,
            ))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "SingleStep: BM25=%d Dense=%d Fused=%d Reranked=%d → %d results (%.0fms)",
            len(bm25_results), len(dense_results),
            len(fused), len(reranked), len(documents), elapsed_ms,
        )

        if requested_top_k is not None:
            try:
                documents = documents[:max(0, int(requested_top_k))]
            except (TypeError, ValueError):
                logger.debug("Ignoring invalid top_k value: %r", requested_top_k)

        return SearchResult(
            query=query,
            documents=documents,
            strategy=self.strategy_type,
            total_found=len(documents),
            search_time_ms=elapsed_ms,
        )


# ================================================================
# Reranker 预热 (应用启动时后台加载，消除首次查询卡顿)
# ================================================================

_reranker_warmed_up: bool = False
_reranker_warmup_status: str = "not_started"  # not_started / warming / ready / failed


def get_reranker_status() -> str:
    """获取 Reranker 预热状态。

    Returns:
        "not_started" | "warming" | "ready" | "failed"
    """
    return _reranker_warmup_status


async def _warmup_reranker_async() -> None:
    """异步预热 Reranker 模型 (触发 CrossEncoder 下载/加载)。

    调用 get_single_step() 获取单例并触发 _ensure_reranker()。
    失败时不抛异常，设置状态为 "failed" 并记录 warning。
    """
    global _reranker_warmed_up, _reranker_warmup_status
    try:
        settings = get_settings()
        if not settings.rerank_enabled or settings.rerank_provider.lower() != "local":
            _reranker_warmed_up = True
            _reranker_warmup_status = "ready"
            logger.info("Reranker warmup skipped for provider=%s", settings.rerank_provider)
            return
        _reranker_warmup_status = "warming"
        logger.info("Reranker 模型预热开始...")
        strategy = await get_single_step()
        await strategy._ensure_reranker()
        _reranker_warmed_up = True
        _reranker_warmup_status = "ready"
        logger.info("Reranker 模型预热完成")
    except Exception as e:
        _reranker_warmed_up = False
        _reranker_warmup_status = "failed"
        logger.warning("Reranker 模型预热失败（将在首次查询时懒加载）: %s", e)


def warmup_reranker() -> None:
    """同步触发 Reranker 后台预热 (不阻塞调用线程)。

    启动独立线程运行预热逻辑。适用于无法直接运行 async 代码的场景
    （如 FastAPI lifespan）。Streamlit 环境请使用 run_async(_warmup_reranker_async)。

    预热失败不影响应用正常启动，首次查询时自动懒加载。
    """
    import threading

    def _run():
        import asyncio
        try:
            asyncio.run(_warmup_reranker_async())
        except Exception as e:
            logger.warning("Reranker 预热线程异常: %s", e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
