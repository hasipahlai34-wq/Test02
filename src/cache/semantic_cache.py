"""
# ============================================================
# ★ 语义缓存 (Semantic Cache)
# ← 本项目设计: 精确匹配 + 余弦相似度语义匹配
#   WeKnora 无此功能 — Redis 仅用于 Asynq 任务队列
#
# 两阶段缓存:
# 1. 精确匹配: 完全相同的问题 → 直接返回缓存结果 (O(1))
# 2. 语义匹配: 语义相似的问题 → 余弦相似度 > 阈值 → 返回缓存结果
#
# 这是面试亮点: "我不仅缓存精确查询，还通过余弦相似度做语义匹配，
# 相似问题可以直接命中缓存，避免重复的 LLM 调用。"
# ============================================================
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from src.models.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

# 缓存条目 TTL (秒) — 默认 1 小时
DEFAULT_TTL = 3600
_semantic_cache_instance: Optional["SemanticCache"] = None


def get_semantic_cache() -> "SemanticCache":
    """Return the process-wide semantic cache instance."""
    global _semantic_cache_instance
    if _semantic_cache_instance is None:
        _semantic_cache_instance = SemanticCache()
    return _semantic_cache_instance


class SemanticCache:
    """
    语义缓存 — 精确匹配 + Embedding 相似度匹配

    用法:
        cache = SemanticCache()
        result = await cache.lookup("查询文本")
        if result:
            return result  # 缓存命中
        answer = generate_answer()
        await cache.store("查询文本", answer)
    """

    def __init__(
        self,
        ttl: int = DEFAULT_TTL,
        similarity_threshold: float = 0.92,
        max_entries: int = 1000,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

        # 存储结构: {exact_key: (timestamp, answer)}
        self._exact_cache: dict[str, tuple[float, str]] = {}

        # 语义缓存: [(query_text, embedding, timestamp, answer)]
        self._semantic_cache: list[tuple[str, list[float], float, str]] = []

        self._embedding = embedding_model or EmbeddingModel()

        # 统计
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0

    # ----------------------------------------------------------------
    # 查找
    # ----------------------------------------------------------------

    async def lookup(self, query: str) -> Optional[str]:
        """
        查找缓存: 精确匹配 → 语义匹配

        Args:
            query: 查询文本

        Returns:
            缓存的答案 (如果有)，None 表示未命中
        """
        # Phase 1: 精确匹配
        exact_key = self._exact_key(query)
        if exact_key in self._exact_cache:
            ts, answer = self._exact_cache[exact_key]
            if not self._is_expired(ts):
                self._exact_hits += 1
                logger.debug("缓存命中: 精确匹配 (%d hits)", self._exact_hits)
                return answer

        # Phase 2: 语义匹配 — 确保 Embedding 模型已就绪
        try:
            await self._embedding._ensure_model()
            query_embedding = await self._embedding.embed_single(query)
        except Exception as e:
            logger.warning("语义缓存: Embedding 失败: %s", e)
            self._misses += 1
            return None

        best_similarity = 0.0
        best_answer = None

        for cached_query, cached_embedding, ts, answer in self._semantic_cache:
            if self._is_expired(ts):
                continue

            similarity = self._embedding.cosine_similarity(
                query_embedding, cached_embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = answer

            if similarity >= self.similarity_threshold:
                self._semantic_hits += 1
                logger.debug(
                    "缓存命中: 语义匹配 (sim=%.3f, '%s...' ≈ '%s...')",
                    similarity, query[:30], cached_query[:30],
                )
                return answer

        # 记录最近的最佳相似度
        if best_similarity > 0:
            logger.debug(
                "缓存未命中: 最近相似度=%.3f (阈值=%.3f)",
                best_similarity, self.similarity_threshold,
            )

        self._misses += 1
        return None

    def lookup_exact(self, query: str) -> Optional[str]:
        """Fast exact-cache lookup without embedding model initialization."""
        exact_key = self._exact_key(query)
        if exact_key not in self._exact_cache:
            return None

        ts, answer = self._exact_cache[exact_key]
        if self._is_expired(ts):
            return None

        self._exact_hits += 1
        return answer

    def has_semantic_entries(self) -> bool:
        """Return whether semantic lookup can run without a guaranteed miss."""
        now = time.time()
        return any((now - ts) <= self.ttl for _, _, ts, _ in self._semantic_cache)

    def embedding_ready(self) -> bool:
        """Return whether semantic operations can avoid cold model initialization."""
        return getattr(self._embedding, "_model", None) is not None

    # ----------------------------------------------------------------
    # 存储
    # ----------------------------------------------------------------

    async def store(self, query: str, answer: str) -> None:
        """
        存储缓存条目

        Args:
            query: 查询文本
            answer: 生成的答案
        """
        ts = time.time()

        # 精确缓存
        exact_key = self._exact_key(query)
        self._exact_cache[exact_key] = (ts, answer)

        # 语义缓存 (生成 Embedding)
        try:
            query_embedding = await self._embedding.embed_single(query)
            self._semantic_cache.append((query, query_embedding, ts, answer))
        except Exception as e:
            logger.warning("语义缓存存储: Embedding 失败: %s", e)

        # 淘汰: 超过最大条目数时清除最旧的
        if len(self._exact_cache) > self.max_entries:
            self._evict_exact()
        if len(self._semantic_cache) > self.max_entries:
            self._evict_semantic()

        logger.debug("缓存已存储: '%s...'", query[:40])

    def store_exact(self, query: str, answer: str) -> None:
        """Store an exact-cache entry without embedding model initialization."""
        ts = time.time()
        exact_key = self._exact_key(query)
        self._exact_cache[exact_key] = (ts, answer)
        if len(self._exact_cache) > self.max_entries:
            self._evict_exact()

    def set(self, query: str, answer: str, contexts: list[str] | None = None) -> None:
        """Compatibility helper for legacy exact-cache callers."""
        self.store_exact(query, {"answer": answer, "contexts": contexts or []})

    # ----------------------------------------------------------------
    # 淘汰策略 (LRU: 清除最旧的条目)
    # ----------------------------------------------------------------

    def _evict_exact(self) -> None:
        """精确缓存 LRU 淘汰"""
        # 按时间戳排序，清除最旧的 20%
        items = sorted(self._exact_cache.items(), key=lambda x: x[1][0])
        to_remove = max(1, len(items) // 5)
        for key, _ in items[:to_remove]:
            del self._exact_cache[key]
        logger.debug("精确缓存淘汰: %d 条目", to_remove)

    def _evict_semantic(self) -> None:
        """语义缓存 LRU 淘汰"""
        self._semantic_cache.sort(key=lambda x: x[2])  # 按时间戳
        to_remove = max(1, len(self._semantic_cache) // 5)
        self._semantic_cache = self._semantic_cache[to_remove:]
        logger.debug("语义缓存淘汰: %d 条目", to_remove)

    # ----------------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------------

    @staticmethod
    def _exact_key(query: str) -> str:
        """生成精确匹配的 key (SHA256 hash)"""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """检查缓存是否过期"""
        return (time.time() - timestamp) > self.ttl

    @property
    def hit_rate(self) -> float:
        total = self._exact_hits + self._semantic_hits + self._misses
        if total == 0:
            return 0.0
        return (self._exact_hits + self._semantic_hits) / total

    def stats(self) -> dict:
        return {
            "exact_entries": len(self._exact_cache),
            "semantic_entries": len(self._semantic_cache),
            "exact_hits": self._exact_hits,
            "semantic_hits": self._semantic_hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "ttl_seconds": self.ttl,
            "threshold": self.similarity_threshold,
        }

    def clear(self) -> None:
        """清空所有缓存"""
        self._exact_cache.clear()
        self._semantic_cache.clear()
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        logger.info("语义缓存已清空")
