"""
# ============================================================
# Embedding 模型封装
# ← WeKnora: internal/models/embedding/ 多种 Embedding 后端
#   我们简化为两种模式:
#   1. OpenAI text-embedding-3 (高质量，需 API Key)
#   2. 本地 sentence-transformers (离线可用，零成本)
# ============================================================

本模块提供统一的 Embedding 接口，屏蔽不同后端的差异。
支持 OpenAI 云端和 sentence-transformers 本地两种模式。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _hf_model_cache_exists(model_name: str) -> bool:
    """Return whether a HuggingFace model appears to be available locally."""
    cache_root = (
        os.environ.get("HF_HOME")
        or os.environ.get("TRANSFORMERS_CACHE")
        or str(Path.home() / ".cache" / "huggingface")
    )
    model_cache = Path(cache_root) / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots = model_cache / "snapshots"
    if not snapshots.exists():
        return False
    return any(path.is_dir() and any(path.iterdir()) for path in snapshots.iterdir())


class EmbeddingModel:
    """
    统一的 Embedding 模型封装
    ← WeKnora: internal/models/embedding/ 精简版

    用法:
        model = EmbeddingModel()
        vectors = await model.embed(["查询文本", "文档1", "文档2"])
        single = await model.embed_single("单条文本")
    """

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self.provider = provider or self._settings.embedding_provider
        self.model_name = model_name or self._settings.embedding_model
        self.dimensions = self._settings.embedding_dimensions

        self._model: Optional[object] = None
        self._openai_client = None

    # ----------------------------------------------------------------
    # 懒加载模型 (首次调用时初始化)
    # ----------------------------------------------------------------

    async def _ensure_model(self):
        """确保模型已加载"""
        if self._model is not None:
            return

        if self.provider == "openai":
            await self._init_openai()
        elif self.provider == "local":
            await self._init_local()
        else:
            raise ValueError(f"不支持的 Embedding 提供商: {self.provider}")

    async def _init_openai(self):
        """初始化 OpenAI Embedding ← WeKnora: embeddings/openai.go"""
        from langchain_openai import OpenAIEmbeddings

        # ★ 使用独立的 Embedding Base URL，避免与 Chat 混用
        # DashScope embeddings must use an embedding-specific key to avoid
        # accidentally sending requests with an unrelated chat-provider key.
        is_dashscope = (
            "dashscope.aliyuncs.com" in self._settings.embedding_base_url
            or "maas.aliyuncs.com" in self._settings.embedding_base_url
        )
        if is_dashscope:
            api_key = self._settings.dashscope_api_key or self._settings.embedding_api_key
        else:
            api_key = self._settings.embedding_api_key or self._settings.llm_api_key
        if not api_key or api_key == "sk-placeholder":
            raise ValueError(
                "EMBEDDING_API_KEY or DASHSCOPE_API_KEY is required for cloud embeddings"
            )
        embedding_kwargs = {
            "model": self.model_name,
            "api_key": api_key,
            "base_url": self._settings.embedding_base_url,
            "dimensions": self.dimensions,
            "chunk_size": max(1, self._settings.embedding_batch_size),
        }
        if is_dashscope:
            # DashScope OpenAI-compatible embeddings require `input` to be a
            # string or list of strings. LangChain's default context-length
            # check may tokenize inputs and send token-id arrays, which
            # DashScope rejects with `input.contents` validation errors.
            embedding_kwargs.update({
                "check_embedding_ctx_length": False,
                "tiktoken_enabled": False,
                "chunk_size": min(max(1, self._settings.embedding_batch_size), 10),
            })

        self._openai_client = OpenAIEmbeddings(**embedding_kwargs)
        self._model = self._openai_client
        logger.info("Embedding: OpenAI %s (dim=%d)", self.model_name, self.dimensions)

    async def _init_local(self):
        """
        初始化本地 sentence-transformers 模型
        首次运行会自动下载模型（约 80-120MB），之后使用缓存
        """
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_kwargs = {"device": "cpu"}
        if _hf_model_cache_exists(self.model_name):
            model_kwargs["local_files_only"] = True
            logger.info("Embedding: using local HuggingFace cache for %s", self.model_name)

        self._local_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._model = self._local_model
        logger.info("Embedding: Local %s (CPU)", self.model_name)

    # ----------------------------------------------------------------
    # 核心接口
    # ----------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量文本向量化
        ← WeKnora: embeddings/embedder.go Embed()

        Args:
            texts: 待向量化的文本列表

        Returns:
            等长的向量列表，每个向量维度为 self.dimensions
        """
        await self._ensure_model()
        if not texts:
            return []

        if self.provider == "openai":
            batch_size = max(1, self._settings.embedding_batch_size)
            if (
                "dashscope.aliyuncs.com" in self._settings.embedding_base_url
                or "maas.aliyuncs.com" in self._settings.embedding_base_url
            ):
                batch_size = min(batch_size, 10)
            vectors = []
            for i in range(0, len(texts), batch_size):
                vectors.extend(await self._openai_client.aembed_documents(texts[i:i + batch_size]))
        else:
            vectors = self._local_model.embed_documents(texts)

        return vectors

    async def embed_single(self, text: str) -> list[float]:
        """
        单条文本向量化
        ← WeKnora: embeddings/embedder.go EmbedQuery()

        Args:
            text: 单条文本

        Returns:
            向量表示
        """
        results = await self.embed([text])
        return results[0] if results else []

    # ----------------------------------------------------------------
    # 相似度计算（辅助方法）
    # ----------------------------------------------------------------

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        计算两个向量的余弦相似度
        ← WeKnora: 各向量库内部实现，我们在此提供通用工具

        Args:
            vec_a, vec_b: 两个等长向量

        Returns:
            余弦相似度 [-1, 1]，越高越相似
        """
        if len(vec_a) != len(vec_b):
            raise ValueError(f"向量维度不匹配: {len(vec_a)} vs {len(vec_b)}")

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    """获取全局 Embedding 模型单例"""
    return EmbeddingModel()
