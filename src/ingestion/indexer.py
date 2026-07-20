"""
# ============================================================
# 文档索引器 (Embedding + ChromaDB 入库)
# ← WeKnora: internal/application/service/ 索引管道
#   - knowledge.go: 创建知识条目 → 分块 → 向量化 → 入库
#   - 多种向量库后端的抽象层 (pgvector/ES/Milvus/Qdrant/...)
#
#   我们简化为 ChromaDB 单一后端:
#   1. 接收分块后的 Document 列表
#   2. 调用 Embedding 模型生成向量
#   3. 存入 ChromaDB (持久化到本地磁盘)
# ============================================================

本模块负责:
- 创建/获取 ChromaDB Collection
- 将分块后的文档向量化并存入向量库
- 支持增量添加和全量重建
- 文档删除

设计要点:
- ChromaDB 是 Python 原生的本地向量库，零配置
- 使用 LangChain Chroma wrapper 统一接口
- 支持 metadata 过滤 (按 source、chunk_index 等)
"""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import Callable, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import Settings, get_settings
from src.models.embeddings import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)

_shared_indexer: DocumentIndexer | None = None
_index_generation = 0


def mark_index_updated() -> int:
    """Bump the in-process index generation after writes or deletes."""
    global _index_generation
    _index_generation += 1
    return _index_generation


def get_index_generation() -> int:
    """Return the current in-process index generation."""
    return _index_generation


def _to_chroma_where(where: dict) -> dict:
    """Convert a plain equality dict into a Chroma-compatible where filter."""
    clean = {key: value for key, value in where.items() if value is not None}
    if len(clean) <= 1:
        return clean
    return {"$and": [{key: value} for key, value in clean.items()]}


class DocumentIndexer:
    """
    文档索引器
    ← WeKnora: 多个 vectorstore 实现 (pgvector/ES/Milvus/...)
               → 我们简化为 ChromaDB 单一后端

    用法:
        indexer = DocumentIndexer()
        await indexer.index_documents(chunks)
        results = await indexer.search("查询文本", top_k=5)
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | Path | None = None,
        embedding_model: EmbeddingModel | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._collection_name = collection_name or self._settings.chroma_collection_name
        self._persist_dir = str(persist_dir or self._settings.chroma_persist_dir)
        self._embedding_model = embedding_model

        # 确保持久化目录存在
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self._vectorstore: Optional[Chroma] = None
        self._initialized = False

        logger.info(
            "DocumentIndexer: collection=%s persist_dir=%s",
            self._collection_name, self._persist_dir,
        )

    # ----------------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        """懒初始化: 首次使用时加载/创建 ChromaDB Collection"""
        if self._initialized:
            return

        if self._embedding_model is None:
            try:
                self._embedding_model = get_embedding_model()
                await self._embedding_model._ensure_model()
            except Exception as e:
                settings = get_settings()
                raise RuntimeError(
                    f"Embedding 模型初始化失败\n"
                    f"  当前 provider: {settings.embedding_provider}\n"
                    f"  当前 model: {settings.embedding_model}\n"
                    f"  若使用 openai provider，请检查 EMBEDDING_BASE_URL={settings.embedding_base_url}\n"
                    f"  原始错误: {e}"
                ) from e

        # 获取 LangChain 兼容的 Embeddings 对象
        embeddings = self._get_lc_embeddings()

        self._vectorstore = Chroma(
            collection_name=self._collection_name,
            embedding_function=embeddings,
            persist_directory=self._persist_dir,
        )
        self._initialized = True

        # 获取已有文档数量
        try:
            count = self._vectorstore._collection.count()
            logger.info("ChromaDB 已连接: %d 个文档", count)
        except Exception as e:
            logger.info("ChromaDB 已连接 (新建 Collection, 原因: %s)", e)

    def _get_lc_embeddings(self):
        """
        获取 LangChain 兼容的 Embeddings 对象
        桥接我们的 EmbeddingModel 和 LangChain 接口
        """
        if self._embedding_model is None:
            raise RuntimeError("EmbeddingModel 未初始化")

        if self._embedding_model.provider == "openai":
            return self._embedding_model._openai_client
        else:
            return self._embedding_model._local_model

    # ----------------------------------------------------------------
    # 索引操作 (← WeKnora: knowledge.go CreateKnowledge + IndexChunks)
    # ----------------------------------------------------------------

    async def index_documents(
        self,
        chunks: list[Document],
        batch_size: int = 50,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """
        将分块后的文档向量化并存入 ChromaDB
        ← WeKnora: knowledge.go → embed → vectorstore.Insert()

        Args:
            chunks: 分块后的 LangChain Document 列表
            batch_size: 批量入库大小

        Returns:
            成功索引的文档数
        """
        if not chunks:
            logger.warning("没有文档需要索引")
            return 0

        await self._ensure_initialized()

        # 逐批入库 (ChromaDB 对大批次支持有限)
        effective_batch_size = max(1, min(batch_size, self._effective_embedding_batch_size()))
        total_indexed = 0
        failed_batches = 0
        for i in range(0, len(chunks), effective_batch_size):
            batch = chunks[i:i + effective_batch_size]
            try:
                ids = await self._vectorstore.aadd_documents(batch)
                total_indexed += len(ids)
                if progress_callback:
                    progress_callback(min(i + len(batch), len(chunks)), len(chunks))
                logger.debug(
                    "索引批次: %d/%d 完成 (batch=%d docs)",
                    i + len(batch), len(chunks), len(batch),
                )
            except Exception as e:
                failed_batches += 1
                logger.error("索引批次失败 (offset=%d): %s", i, e)

        if total_indexed == 0 and failed_batches > 0:
            settings = get_settings()
            error_msg = (
                f"索引失败：所有 {failed_batches} 个批次均入库失败。"
                f"请检查 Embedding 配置："
                f"provider={settings.embedding_provider}, "
                f"model={settings.embedding_model}"
            )
            if settings.embedding_provider == "openai":
                error_msg += f", base_url={settings.embedding_base_url}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if total_indexed > 0:
            mark_index_updated()

        logger.info("索引完成: %d/%d chunks 成功入库", total_indexed, len(chunks))
        return total_indexed

    async def index_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> int:
        """
        直接索引文本列表 (跳过 Document 包装)

        Args:
            texts: 文本列表
            metadatas: 对应的元数据列表

        Returns:
            成功索引的文本数
        """
        await self._ensure_initialized()

        if metadatas is None:
            metadatas = [{}] * len(texts)

        batch_size = self._effective_embedding_batch_size()
        total_indexed = 0
        for i in range(0, len(texts), batch_size):
            try:
                ids = await self._vectorstore.aadd_texts(
                    texts[i:i + batch_size],
                    metadatas[i:i + batch_size],
                )
                total_indexed += len(ids)
            except Exception as e:
                logger.error("文本索引失败 (offset=%d): %s", i, e)
        logger.info("文本索引完成: %d/%d 条", total_indexed, len(texts))
        return total_indexed

    def _effective_embedding_batch_size(self) -> int:
        """Return a safe embedding batch size for the configured provider."""
        batch_size = max(1, self._settings.embedding_batch_size)
        if (
            self._settings.embedding_provider == "openai"
            and (
                "dashscope.aliyuncs.com" in self._settings.embedding_base_url
                or "maas.aliyuncs.com" in self._settings.embedding_base_url
            )
        ):
            return min(batch_size, 10)
        return batch_size

    # ----------------------------------------------------------------
    # 检索操作
    # ----------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        """
        向量语义检索
        ← WeKnora: knowledge_search.go → vectorstore.Search()

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: ChromaDB metadata 过滤条件
            score_threshold: 最低相似度阈值 (None 则使用 settings 默认值)

        Returns:
            [(Document, 相似度得分), ...] 按得分降序排列
        """
        await self._ensure_initialized()

        if score_threshold is None:
            score_threshold = self._settings.retrieval_threshold

        try:
            raw_results = await self._vectorstore.asimilarity_search_with_score(
                query,
                k=top_k,
                filter=filter_dict,
            )
            results = [
                (doc, self._distance_to_similarity(distance))
                for doc, distance in raw_results
            ]
            if score_threshold > 0:
                results = [
                    (doc, score)
                    for doc, score in results
                    if score >= score_threshold
                ]

            logger.debug(
                "检索完成: query='%s...' → %d results (top_k=%d, threshold=%.2f)",
                query[:50], len(results), top_k, score_threshold,
            )
            return results

        except Exception as e:
            logger.error("检索失败: %s", e)
            return []

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Convert Chroma distance to a bounded higher-is-better score.

        LangChain's relevance-score adapter can emit negative scores for some
        embedding/vector-space combinations, and then `score_threshold=0.0`
        filters out every result. Chroma's native search already returns rows in
        best-first order, so use its raw distance and derive a stable score only
        for logging/downstream thresholds.
        """
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        if value <= 0:
            return 1.0
        return 1.0 / (1.0 + value)

    # ----------------------------------------------------------------
    # 管理操作
    # ----------------------------------------------------------------

    async def delete_by_metadata(self, where: dict) -> int:
        """Delete chunks matching Chroma metadata filters."""
        await self._ensure_initialized()
        if not where:
            return 0

        try:
            results = self._vectorstore.get(where=_to_chroma_where(where))
            ids = results.get("ids") if results else []
            if not ids:
                return 0
            self._vectorstore.delete(ids=ids)
            mark_index_updated()
            logger.info("Deleted %d chunks by metadata filter: %s", len(ids), where)
            return len(ids)
        except Exception as e:
            logger.error("Delete by metadata failed: %s - %s", where, e)
            return 0

    async def delete_by_session(self, session_id: str) -> int:
        """Delete chunks that belong to a UI session."""
        return await self.delete_by_metadata({"session_id": session_id})

    async def delete_by_source(self, source: str) -> int:
        """
        按来源文件删除文档
        ← WeKnora: knowledge.go DeleteKnowledge

        Args:
            source: 来源文件名

        Returns:
            删除的文档数
        """
        await self._ensure_initialized()

        try:
            # ChromaDB 的 delete 需要先查询后删除
            results = self._vectorstore.get(where={"source": source})
            if results and results["ids"]:
                self._vectorstore.delete(ids=results["ids"])
                count = len(results["ids"])
                mark_index_updated()
                logger.info("已删除: source=%s → %d chunks", source, count)
                return count
            return 0
        except Exception as e:
            logger.error("删除文档失败: %s — %s", source, e)
            return 0

    async def delete_by_ids(self, ids: list[str]) -> int:
        """
        按 ID 列表删除文档

        Args:
            ids: chunk ID 列表

        Returns:
            删除的文档数
        """
        await self._ensure_initialized()
        try:
            self._vectorstore.delete(ids=ids)
            mark_index_updated()
            logger.info("已删除: %d chunks", len(ids))
            return len(ids)
        except Exception as e:
            logger.error("按 ID 删除失败: %s", e)
            return 0

    async def clear_all(self) -> None:
        """
        清空整个 Collection
        ← WeKnora: 重建索引
        """
        await self._ensure_initialized()
        try:
            self._vectorstore.delete_collection()
            mark_index_updated()
            self._initialized = False
            self._vectorstore = None
            logger.info("Collection 已清空: %s", self._collection_name)
        except Exception as e:
            logger.error("清空 Collection 失败: %s", e)

    async def count(self) -> int:
        """获取索引中的文档总数"""
        await self._ensure_initialized()
        return self._vectorstore._collection.count()

    async def get_sources(self) -> list[str]:
        """
        获取所有已索引的文档来源
        ← WeKnora: knowledge.go ListKnowledge
        """
        await self._ensure_initialized()
        try:
            results = self._vectorstore.get()
            if results and results["metadatas"]:
                sources = set()
                for meta in results["metadatas"]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])
                    elif meta and "source_file" in meta:
                        sources.add(meta["source_file"])  # 向后兼容旧索引
                return sorted(sources)
            return []
        except Exception as e:
            logger.error("获取来源列表失败: %s", e)
            return []

    def get_all_documents(self) -> list[dict]:
        """返回索引中的所有文档内容、元数据和 ID。

        公共方法，替代直接访问 self._vectorstore.get()，
        避免外部代码依赖 ChromaDB 内部 API 稳定性。

        Returns:
            dict 列表，每项包含 content, metadata, id 键。
            索引未初始化或查询失败时返回空列表。
        """
        if self._vectorstore is None:
            logger.warning("get_all_documents: 索引未初始化")
            return []

        try:
            results = self._vectorstore.get()
            documents: list[dict] = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"]):
                    doc_dict: dict = {"content": doc}
                    if results.get("metadatas") and i < len(results["metadatas"]):
                        doc_dict["metadata"] = results["metadatas"][i]
                    if results.get("ids") and i < len(results["ids"]):
                        doc_dict["id"] = results["ids"][i]
                    documents.append(doc_dict)
            return documents
        except Exception as e:
            logger.warning("获取全部文档失败: %s", e)
            return []

    async def wait_until_visible(
        self,
        where: dict,
        timeout_seconds: float = 2.0,
        interval_seconds: float = 0.1,
    ) -> bool:
        """Wait briefly until recently written chunks are visible to this indexer."""
        await self._ensure_initialized()
        chroma_where = _to_chroma_where(where)
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            try:
                results = self._vectorstore.get(where=chroma_where)
                if results and results.get("ids"):
                    return True
            except Exception as e:
                logger.debug("wait_until_visible lookup failed: %s", e)

            try:
                for item in self.get_all_documents():
                    metadata = item.get("metadata") or {}
                    if all(metadata.get(key) == value for key, value in where.items() if value is not None):
                        return True
            except Exception as e:
                logger.debug("wait_until_visible fallback scan failed: %s", e)

            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(interval_seconds)

    async def list_session_documents(self, session_id: str) -> list[dict]:
        """Aggregate indexed chunks into document rows for one frontend session."""
        await self._ensure_initialized()
        documents = self.get_all_documents()
        grouped: dict[str, dict] = {}

        for item in documents:
            metadata = item.get("metadata") or {}
            if metadata.get("session_id") != session_id:
                continue

            document_id = str(metadata.get("document_id") or "")
            if not document_id:
                document_id = str(metadata.get("source_document_id") or metadata.get("source") or "unknown")

            row = grouped.setdefault(document_id, {
                "filename": metadata.get("upload_filename") or Path(str(metadata.get("source") or "")).name or "unknown",
                "raw_segments": 0,
                "chunks": 0,
                "indexed": 0,
                "status": "ok",
                "document_id": document_id,
                "source_document_id": metadata.get("source_document_id"),
                "parse_quality_score": metadata.get("parse_quality_score"),
                "outline_preview": metadata.get("outline_preview"),
                "element_count": metadata.get("element_count"),
                "warning_count": metadata.get("warning_count"),
                "chunk_strategy": metadata.get("chunk_strategy"),
                "target_tokens": metadata.get("chunk_target_tokens"),
                "overlap_tokens": metadata.get("chunk_overlap_tokens"),
                "chunk_plan_reason": metadata.get("chunk_plan_reason"),
                "uploaded_at": metadata.get("uploaded_at"),
                "error": None,
            })
            row["chunks"] += 1
            row["indexed"] += 1

            for key, meta_key in (
                ("parse_quality_score", "parse_quality_score"),
                ("outline_preview", "outline_preview"),
                ("element_count", "element_count"),
                ("warning_count", "warning_count"),
                ("chunk_strategy", "chunk_strategy"),
                ("target_tokens", "chunk_target_tokens"),
                ("overlap_tokens", "chunk_overlap_tokens"),
                ("chunk_plan_reason", "chunk_plan_reason"),
                ("uploaded_at", "uploaded_at"),
            ):
                if row.get(key) in (None, "", 0) and metadata.get(meta_key) not in (None, ""):
                    row[key] = metadata.get(meta_key)

        return sorted(
            grouped.values(),
            key=lambda row: str(row.get("uploaded_at") or ""),
            reverse=True,
        )


# ================================================================
# 便捷函数
# ================================================================


def get_document_indexer() -> DocumentIndexer:
    """Return the process-wide indexer so writes are immediately visible to reads."""
    global _shared_indexer
    if _shared_indexer is None:
        _shared_indexer = DocumentIndexer()
    return _shared_indexer


async def build_index(
    chunks: list[Document],
    collection_name: str | None = None,
) -> DocumentIndexer:
    """
    一键构建索引: 接收分块 → 向量化 → 入库

    Args:
        chunks: 分块后的 Document 列表
        collection_name: ChromaDB Collection 名称

    Returns:
        已初始化的 DocumentIndexer 实例
    """
    indexer = DocumentIndexer(collection_name=collection_name)
    count = await indexer.index_documents(chunks)
    logger.info("索引构建完成: %d 个 chunks", count)
    return indexer


async def ingest_pipeline(
    filepath: str | Path,
    strategy: str = "recursive",
    chunk_size: int = 800,
) -> tuple[list[Document], list[Document], int]:
    """
    完整的文档摄入管道: 加载 → 分块 → 索引
    ← WeKnora: 完整的 ingest pipeline (docreader → chunker → indexer)

    Args:
        filepath: 文档文件路径
        strategy: 分块策略
        chunk_size: 分块大小

    Returns:
        (原始文档, 分块文档, 已索引数量)
    """
    from src.ingestion.loader import load_document
    from src.ingestion.chunker import chunk_documents, ChunkingStrategy

    # Step 1: 加载
    raw_docs = await load_document(filepath)

    # Step 2: 分块
    chunk_strategy = ChunkingStrategy(strategy)
    chunks = chunk_documents(raw_docs, strategy=chunk_strategy, chunk_size=chunk_size)

    # Step 3: 索引
    indexer = DocumentIndexer()
    indexed = await indexer.index_documents(chunks)

    return raw_docs, chunks, indexed
