"""
# ============================================================
# ★ HyDE (Hypothetical Document Embeddings) 假设文档嵌入
# ← HyDE 论文 (Gao et al., 2022): Precise Zero-Shot Dense Retrieval
#   Without Relevance Labels
#
# 核心思想:
#   用户查询通常是短小口语化的 ("Q3赚钱最多的是哪个部门？")
#   但文档内容是正式、完整的 ("2024年第三季度，云计算事业部以35%的
#   营收占比成为公司最大营收来源...")
#   这两个在 Embedding 空间中的距离很大，导致检索效果差
#
#   HyDE 的解决方案:
#   1. 让 LLM 根据查询生成一份"假设的答案文档"
#   2. 用假设文档的 Embedding 代替查询的 Embedding 做向量检索
#   3. 因为假设文档和真实文档语言风格一致，Embedding 距离更近
#
# ← WeKnora 无此功能 (GAP_ANALYSIS.md #2: HyDE 完全缺失)
# ============================================================
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import get_settings
from src.models.llm import LLMClient
from src.models.embeddings import EmbeddingModel
from src.utils.prompt_loader import load_prompt_with_default

logger = logging.getLogger(__name__)


class HyDEGenerator:
    """
    HyDE 假设文档生成器

    用法:
        hyde = HyDEGenerator()
        hypothesis = await hyde.generate("云计算营收增长驱动因素")
        # → "2024年第三季度云计算业务营收同比增长35%,
        #    主要驱动力包括: AI平台服务增长28%,
        #    企业级SaaS产品线同比增长42%..."

    面试可讲:
    "HyDE 的核心洞察是: Embedding 空间中，
    假设答案的向量比口语查询的向量更接近真实文档的向量。
    因为假设答案和真实文档都是正式、完整的陈述，而用户查询是碎片化的问题。
    实验表明在开放域 QA 任务上，HyDE 可以将 Recall@10 提升 10-15 个百分点。"
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self._llm = llm_client or LLMClient()
        self._embedding = embedding_model or EmbeddingModel()

    async def generate(self, query: str, language: str = "中文") -> str:
        """
        生成 HyDE 假设文档

        Args:
            query: 用户原始查询
            language: 输出语言

        Returns:
            假设的文档段落 (100-200字)
        """
        prompt = load_prompt_with_default(
            "hyde",
            query=query,
            language=language,
        )

        try:
            hypothesis = await self._llm.ask(
                prompt=prompt,
                system_prompt="你是一个文档生成专家，请生成一段假设的答案文档来帮助搜索。",
                model_name=get_settings().llm_simple_model,  # HyDE 用便宜模型即可
            )
            logger.info(
                "HyDE 生成完成: query='%s...' → hypothesis(len=%d)",
                query[:50], len(hypothesis),
            )
            return hypothesis.strip()

        except Exception as e:
            logger.error("HyDE 生成失败: %s，降级为原始查询", e)
            return query  # 降级: 直接使用原始查询

    async def embed_hypothesis(self, hypothesis: str) -> list[float]:
        """
        将假设文档向量化 — 此向量用于替代原始查询向量做检索

        Args:
            hypothesis: HyDE 生成的假设文档

        Returns:
            假设文档的 Embedding 向量
        """
        return await self._embedding.embed_single(hypothesis)

    async def generate_and_embed(self, query: str) -> tuple[str, list[float]]:
        """
        生成假设文档并向量化 (一步完成)

        Args:
            query: 用户查询

        Returns:
            (假设文档文本, 假设文档向量)
        """
        hypothesis = await self.generate(query)
        embedding = await self.embed_hypothesis(hypothesis)
        return hypothesis, embedding
