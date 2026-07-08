"""
# ============================================================
# 长期记忆管理 (持久化知识摘要 + 向量语义搜索)
# ← WeKnora: agent/memory/consolidator.go — LLM 对话摘要
#
# 长期记忆将重要对话内容压缩为持久化的结构化知识:
# - 对话摘要: 将多轮对话压缩为关键信息
# - 知识沉淀: 从对话中提取可复用的知识点
# - ★ 向量存储: ChromaDB 持久化索引 + Embedding 语义搜索
#
# 与短期记忆的区别:
# - 短期记忆是原始对话文本 (窗口有限)
# - 长期记忆是 LLM 压缩后的结构化知识 (持久化，可检索)
# ============================================================
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from src.types import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    长期记忆 — 持久化知识摘要 + ChromaDB 向量索引
    ← WeKnora: agent/memory/consolidator.go

    面试可讲:
    "长期记忆是对短期记忆的'蒸馏'——当对话轮数超过阈值时，
    我让 LLM 将早期的对话内容压缩为结构化的知识摘要，
    原始对话文本可以被丢弃，但摘要作为长期记忆持久化存储。
    ★ 我使用 EmbeddingModel 生成向量存入 ChromaDB，
    搜索时用余弦相似度做语义匹配，比关键词匹配更准确。
    当 Embedding 不可用时，自动降级为 Jaccard 关键词匹配。"
    """

    # ChromaDB collection 元数据
    _LTM_COLLECTION_NAME = "ltm_memory"

    def __init__(self):
        # ---- 内存索引 (快速访问) ----
        self._entries: list[MemoryEntry] = []
        self._id_index: dict[str, MemoryEntry] = {}  # entry.id → entry 快速查找

        # ---- Embedding (懒加载) ----
        self._embedding_model = None
        self._embedding_available = False

        # ---- ChromaDB (懒加载) ----
        self._chroma_client = None
        self._collection = None
        self._index_ready = False

    # ================================================================
    # 索引初始化
    # ================================================================

    async def _ensure_index(self) -> None:
        """
        懒初始化: ChromaDB 持久化客户端 + EmbeddingModel

        此方法在首次 add()/consolidate() 时自动调用，
        失败时自动降级（_embedding_available = False），不阻塞业务流程。
        """
        if self._index_ready:
            return

        settings = get_settings()
        persist_dir = Path(settings.ltm_chroma_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # ---- ChromaDB ----
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._LTM_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "LTM ChromaDB 已连接: %s (collection=%s, %d docs)",
                persist_dir, self._LTM_COLLECTION_NAME,
                self._collection.count(),
            )
        except Exception as e:
            logger.warning("LTM: ChromaDB 初始化失败 (%s)，降级为纯内存模式", e)
            self._collection = None

        # ---- Embedding ----
        try:
            from src.models.embeddings import get_embedding_model
            self._embedding_model = get_embedding_model()
            await self._embedding_model._ensure_model()
            self._embedding_available = True
            logger.info(
                "LTM Embedding 已就绪: provider=%s dim=%d",
                self._embedding_model.provider,
                self._embedding_model.dimensions,
            )
        except Exception as e:
            logger.warning(
                "LTM: Embedding 不可用 (%s)，搜索降级为关键词匹配。"
                "配置 EMBEDDING_PROVIDER=openai 并设置 API Key，"
                "或 EMBEDDING_PROVIDER=local 使用本地模型。",
                e,
            )
            self._embedding_available = False

        self._index_ready = True

    # ================================================================
    # ★ Embedding 计算 (内部辅助)
    # ================================================================

    async def _compute_embedding(self, text: str) -> Optional[list[float]]:
        """
        计算单条文本的 Embedding 向量

        Returns:
            向量列表，失败返回 None
        """
        if not self._embedding_available or self._embedding_model is None:
            return None
        try:
            return await self._embedding_model.embed_single(text)
        except Exception as e:
            logger.warning("LTM: Embedding 计算失败: %s", e)
            return None

    def _compute_embedding_sync(self, text: str) -> Optional[list[float]]:
        """同步计算 Embedding (供 search() 使用)。

        使用 asyncio.Runner (Python 3.11+) 避免嵌套事件循环。
        当已有事件循环运行时，Runner 会创建独立的事件循环。
        """
        if not self._embedding_available or self._embedding_model is None:
            return None

        try:
            with asyncio.Runner() as runner:
                return runner.run(self._embedding_model.embed_single(text))
        except AttributeError:
            # Python 3.10 及以下降级方案: 线程池 + asyncio.run()
            import concurrent.futures

            def _run():
                return asyncio.run(self._embedding_model.embed_single(text))

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run)
                    return future.result(timeout=30)
            except Exception as e:
                logger.warning("LTM: 同步 Embedding 失败: %s", e)
                return None
        except Exception as e:
            logger.warning("LTM: 同步 Embedding 失败: %s", e)
            return None

    # ================================================================
    # ChromaDB 索引操作 (内部辅助)
    # ================================================================

    def _index_entry(self, entry: MemoryEntry) -> None:
        """将单条 MemoryEntry 同步写入 ChromaDB"""
        if self._collection is None or entry.embedding is None:
            return
        try:
            self._collection.upsert(
                ids=[entry.id],
                documents=[entry.content],
                embeddings=[entry.embedding],
                metadatas=[{
                    "importance": entry.importance,
                    "memory_type": entry.memory_type.value,
                    "created_at": entry.created_at.isoformat(),
                    "access_count": entry.access_count,
                }],
            )
        except Exception as e:
            logger.warning("LTM: ChromaDB 写入失败 (id=%s): %s", entry.id[:8], e)

    def _remove_from_index(self, entry_id: str) -> None:
        """从 ChromaDB 删除指定条目"""
        if self._collection is None:
            return
        try:
            self._collection.delete(ids=[entry_id])
        except Exception as e:
            logger.warning("LTM: ChromaDB 删除失败 (id=%s): %s", entry_id[:8], e)

    # ================================================================
    # 核心操作
    # ================================================================

    async def consolidate(
        self,
        conversation: str,
        llm_client=None,
        max_length: int = 500,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """
        将对话内容压缩为结构化知识摘要
        ← WeKnora: consolidator.go — LLM 摘要

        ★ 压缩后自动计算 Embedding 并写入 ChromaDB 索引

        Args:
            conversation: 待压缩的对话内容
            llm_client: LLM 客户端
            max_length: 摘要最大字符数
            importance: 重要性评分

        Returns:
            包含摘要和 Embedding 的 MemoryEntry
        """
        await self._ensure_index()

        if llm_client is None:
            from src.models.llm import LLMClient
            llm_client = LLMClient()

        from src.utils.prompt_loader import load_prompt

        prompt = load_prompt(
            "conversation_summary",
            filename="summarization",
            conversation=conversation,
            max_length=str(max_length),
            language=get_settings().default_language,
        )

        try:
            summary = await llm_client.ask(
                prompt=prompt,
                model_name=get_settings().llm_simple_model,
            )

            entry = MemoryEntry(
                memory_type=MemoryType.LONG_TERM,
                content=summary,
                importance=importance,
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                metadata={"source": "conversation_consolidation"},
            )

            # ★ 计算 Embedding + 写入 ChromaDB
            entry.embedding = await self._compute_embedding(summary)
            self._entries.append(entry)
            self._id_index[entry.id] = entry
            self._index_entry(entry)

            logger.info(
                "长期记忆: 已压缩 %d chars → %d chars (embed=%s)",
                len(conversation), len(summary),
                "yes" if entry.embedding else "no",
            )
            return entry

        except Exception as e:
            logger.error("长期记忆压缩失败: %s", e)
            # 降级: 直接截断原始对话
            truncated = conversation[:max_length]
            entry = MemoryEntry(
                memory_type=MemoryType.LONG_TERM,
                content=truncated,
                importance=importance * 0.5,
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                metadata={"source": "truncated_fallback"},
            )

            entry.embedding = await self._compute_embedding(truncated)
            self._entries.append(entry)
            self._id_index[entry.id] = entry
            self._index_entry(entry)
            return entry

    async def add(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict = None,
    ) -> MemoryEntry:
        """
        手动添加长期记忆条目 (不使用 LLM 压缩)

        ★ 自动计算 Embedding 并写入 ChromaDB 索引
        """
        await self._ensure_index()

        entry = MemoryEntry(
            memory_type=MemoryType.LONG_TERM,
            content=content,
            importance=importance,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        # ★ 计算 Embedding + 写入 ChromaDB
        entry.embedding = await self._compute_embedding(content)
        self._entries.append(entry)
        self._id_index[entry.id] = entry
        self._index_entry(entry)

        logger.info(
            "长期记忆: 已添加 (importance=%.2f, embed=%s)",
            importance, "yes" if entry.embedding else "no",
        )
        return entry

    # ================================================================
    # ★ 搜索 (向量语义搜索 + 关键词降级)
    # ================================================================

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """简单分词: 中文逐字 + 英文按空格/标点分割"""
        tokens: set[str] = set()
        for ch in text:
            if '一' <= ch <= '鿿':
                tokens.add(ch)
        for token in re.findall(r'[a-zA-Z0-9]{2,}', text.lower()):
            tokens.add(token)
        return tokens

    def _keyword_search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        ★ 关键词搜索 (降级方案): Jaccard 相似度 + 重要性加权

        当 Embedding 不可用或向量搜索失败时使用。
        """
        if not self._entries:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return sorted(
                self._entries, key=lambda e: e.importance, reverse=True
            )[:top_k]

        scored: list[tuple[MemoryEntry, float]] = []
        for entry in self._entries:
            entry_tokens = self._tokenize(entry.content)
            if not entry_tokens:
                continue
            intersection = len(query_tokens & entry_tokens)
            union = len(query_tokens | entry_tokens)
            similarity = intersection / union if union > 0 else 0.0
            combined = similarity * 0.6 + entry.importance * 0.4
            if combined > 0:
                scored.append((entry, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:top_k]]

    def _vector_search(
        self, query: str, top_k: int = 5
    ) -> Optional[list[MemoryEntry]]:
        """
        ★ ChromaDB 向量语义搜索

        先计算 query embedding，再调用 ChromaDB.query() 做余弦相似度检索。

        Returns:
            搜索结果列表，失败返回 None (调用方降级到关键词搜索)
        """
        if self._collection is None:
            return None

        # 计算查询向量 (同步, 新线程中运行事件循环)
        query_vec = self._compute_embedding_sync(query)
        if query_vec is None:
            return None

        try:
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, max(1, self._collection.count())),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("LTM: ChromaDB 查询失败: %s", e)
            return None

        if not results or not results.get("ids") or not results["ids"][0]:
            return None

        entries: list[MemoryEntry] = []
        for i, mem_id in enumerate(results["ids"][0]):
            entry = self._id_index.get(mem_id)
            if entry is not None:
                # 更新访问信息
                entry.access_count += 1
                entry.last_accessed_at = datetime.now(timezone.utc)
                entries.append(entry)

        if entries:
            logger.debug(
                "LTM 向量搜索: '%s...' → %d results (top_k=%d)",
                query[:40], len(entries), top_k,
            )
        return entries if entries else None

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        ★ 语义搜索长期记忆

        搜索策略 (自动降级):
        1. ChromaDB 向量语义搜索 (需 Embedding 可用)
        2. → 失败时降级为 Jaccard 关键词匹配
        3. → 查询为空时返回重要性排序

        面试可讲:
        "搜索采用双路策略: 主路用 Embedding 做向量语义搜索，
        能匹配到'机器学习'→'深度学习'这种同义表达;
        当 Embedding 不可用时 (API Key 未配置 / 本地模型未下载)，
        自动降级为 Jaccard 关键词匹配 + 重要性加权，
        保证系统在任何环境都能正常运行。"

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            按相关性排序的记忆条目列表
        """
        if not self._entries:
            return []

        # ---- 第一优先: ChromaDB 向量语义搜索 ----
        vector_results = self._vector_search(query, top_k)
        if vector_results:
            return vector_results[:top_k]

        # ---- 降级: 关键词匹配 ----
        if not self._embedding_available:
            logger.debug(
                "LTM: Embedding 不可用，使用关键词搜索 (query='%s...')",
                query[:40],
            )

        return self._keyword_search(query, top_k)

    # ================================================================
    # ★ 索引重建
    # ================================================================

    async def rebuild_index(self) -> tuple[int, int]:
        """
        ★ 全量重建 ChromaDB 索引

        从 _entries 重新计算所有 Embedding 并写入 ChromaDB。
        适用于: 切换 Embedding 模型、索引损坏修复、初次部署迁移。

        Returns:
            (成功数, 总数)
        """
        await self._ensure_index()

        if self._collection is None:
            logger.error("LTM: ChromaDB 不可用，无法重建索引")
            return 0, len(self._entries)

        if not self._embedding_available:
            logger.error("LTM: Embedding 不可用，无法重建索引")
            return 0, len(self._entries)

        # 清空现有索引
        try:
            # ChromaDB 没有 truncate，通过删除 collection 并重建实现
            self._chroma_client.delete_collection(self._LTM_COLLECTION_NAME)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._LTM_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("LTM: 旧索引已清空，开始重建...")
        except Exception as e:
            logger.warning("LTM: 清空旧索引失败: %s", e)

        success = 0
        for entry in self._entries:
            entry.embedding = await self._compute_embedding(entry.content)
            if entry.embedding is not None:
                self._index_entry(entry)
                success += 1

        logger.info(
            "LTM: 索引重建完成 %d/%d entries",
            success, len(self._entries),
        )
        return success, len(self._entries)

    # ================================================================
    # 检索 (非搜索路径)
    # ================================================================

    def get_all(self) -> list[MemoryEntry]:
        """获取所有长期记忆 (按重要性降序)"""
        return sorted(self._entries, key=lambda e: e.importance, reverse=True)

    def get_by_importance(self, threshold: float = 0.5) -> list[MemoryEntry]:
        """按重要性过滤记忆"""
        return [e for e in self._entries if e.importance >= threshold]

    def get_context_for_llm(self, max_entries: int = 3) -> str:
        """
        获取长期记忆上下文 (注入 LLM Prompt)

        Returns:
            格式化的记忆文本
        """
        entries = self.get_by_importance(0.3)[:max_entries]
        if not entries:
            return ""

        lines = ["## 长期记忆 (历史对话摘要)\n"]
        for i, entry in enumerate(entries, 1):
            lines.append(f"### 记忆 {i} (重要性: {entry.importance:.2f})")
            lines.append(entry.content)
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # 管理
    # ================================================================

    @property
    def count(self) -> int:
        return len(self._entries)

    def forget(self, index: int) -> Optional[MemoryEntry]:
        """
        删除指定记忆条目

        ★ 同步删除 ChromaDB 索引中的对应文档
        """
        if 0 <= index < len(self._entries):
            entry = self._entries.pop(index)
            self._id_index.pop(entry.id, None)
            self._remove_from_index(entry.id)
            logger.info("长期记忆: 已删除 (id=%s)", entry.id[:8])
            return entry
        return None

    def clear(self) -> None:
        """
        清空全部长期记忆

        ★ 同步清空 ChromaDB Collection
        """
        self._entries.clear()
        self._id_index.clear()

        if self._chroma_client is not None and self._collection is not None:
            try:
                self._chroma_client.delete_collection(self._LTM_COLLECTION_NAME)
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self._LTM_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("长期记忆: 内存 + ChromaDB 已清空")
            except Exception as e:
                logger.warning("长期记忆: ChromaDB 清空失败: %s", e)
                logger.info("长期记忆: 内存已清空")
        else:
            self._entries.clear()
            self._id_index.clear()
            logger.info("长期记忆已清空")
