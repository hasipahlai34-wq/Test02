"""
# ============================================================
# ★ 动态 Token 管理
# ← WeKnora: internal/agent/token/estimator.go (BPE Token 估算)
#           internal/agent/token/compress.go (上下文窗口管理)
# ← 原项目 B: 按查询复杂度动态分配 Token 预算 + 选择模型
#
#   我们的增强:
#   - simple   → max 500 tokens  → gpt-4o-mini ($0.15/1M input)
#   - medium   → max 2000 tokens → gpt-4o-mini ($0.15/1M input)
#   - complex  → max 4000 tokens → gpt-4o     ($2.50/1M input)
#
#   面试可讲:
#   "简单查询用小模型少Token，复杂查询用大模型多Token，
#   在成本和效果之间自动平衡。这比固定使用一个模型节省约 40-60% 的成本。"
# ============================================================
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from config.settings import get_settings
from src.models.llm import LLMClient
from src.utils.observability import TokenTracker

logger = logging.getLogger(__name__)

# 成本估算 (USD per 1M tokens, 2024年价格)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
}


class TokenBudget:
    """
    动态 Token 预算管理
    ← WeKnora: token/estimator.go + compress.go
    ← 原项目 B: 按复杂度分配

    预算分配规则:
      simple:  总预算 500  (输入 400 + 输出 100)
      medium:  总预算 2000 (输入 1500 + 输出 500)
      complex: 总预算 4000 (输入 3000 + 输出 1000)
    """

    BUDGETS = {
        "simple": {"total": 500, "input": 400, "output": 100},
        "medium": {"total": 2000, "input": 1500, "output": 500},
        "complex": {"total": 4000, "input": 3000, "output": 1000},
    }

    @classmethod
    def _get_model_map(cls) -> dict[str, str]:
        """★ 从 Settings 动态读取模型映射 (单一数据源)"""
        s = get_settings()
        return {
            "simple": s.llm_simple_model,
            "medium": s.llm_medium_model,
            "complex": s.llm_complex_model,
        }

    @classmethod
    def for_complexity(cls, complexity: str) -> dict:
        """获取指定复杂度的预算配置"""
        return cls.BUDGETS.get(complexity, cls.BUDGETS["medium"])

    @classmethod
    def model_for_complexity(cls, complexity: str) -> str:
        """★ 获取指定复杂度的推荐模型 (单一数据源: Settings)"""
        model_map = cls._get_model_map()
        return model_map.get(complexity, model_map["medium"])


def estimate_tokens(text: str) -> int:
    """
    估算文本的 Token 数量
    ← WeKnora: token/estimator.go — BPE (Byte Pair Encoding) 估算
               我们使用 tiktoken 库 (OpenAI 官方 tokenizer)

    Args:
        text: 文本内容

    Returns:
        估算的 Token 数量
    """
    try:
        import tiktoken
        # 使用 cl100k_base 编码 (GPT-4/GPT-3.5-turbo 共用)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logger.debug("tiktoken 估算失败: %s，使用字符估算", e)
        # 降级: 粗略估算 (中文约 1 char ≈ 2 tokens, 英文约 1 word ≈ 1.3 tokens)
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 2 + other_chars * 0.3)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str | None = None,
) -> float:
    """
    估算 API 调用费用
    ← 本项目设计: 实时成本追踪

    Args:
        input_tokens: 输入 Token 数量
        output_tokens: 输出 Token 数量
        model: 模型名称 (默认使用 simple 模型)

    Returns:
        估算费用 (USD)
    """
    if model is None:
        model = get_settings().llm_simple_model
    pricing = MODEL_PRICING.get(model, MODEL_PRICING.get(get_settings().llm_simple_model, MODEL_PRICING["gpt-4o-mini"]))
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def should_compress(
    current_tokens: int,
    max_tokens: int,
    threshold: float = 0.5,
) -> bool:
    """
    判断是否需要压缩上下文
    ← WeKnora: token/compress.go — 50% 阈值触发 LLM 摘要压缩

    Args:
        current_tokens: 当前 Token 数
        max_tokens: 最大 Token 预算
        threshold: 触发压缩的阈值 (默认 50%)

    Returns:
        True → 需要压缩，False → 无需压缩
    """
    ratio = current_tokens / max_tokens if max_tokens > 0 else 0
    if ratio >= threshold:
        logger.info(
            "上下文压缩触发: %d/%d tokens (%.0f%% ≥ %.0f%%)",
            current_tokens, max_tokens, ratio * 100, threshold * 100,
        )
        return True
    return False


async def compress_context(
    content: str,
    max_tokens: int,
    llm_client: Optional[LLMClient] = None,
) -> str:
    """
    使用 LLM 摘要压缩上下文
    ← WeKnora: token/compress.go — LLM 摘要 → 不是简单截断

    Args:
        content: 需要压缩的文本
        max_tokens: 压缩后的目标 Token 数
        llm_client: LLM 客户端

    Returns:
        压缩后的文本
    """
    if llm_client is None:
        llm_client = LLMClient()

    prompt = f"""请将以下内容压缩到 {max_tokens} tokens 以内，保留关键信息。

## 原始内容
{content}

## 压缩要求
1. 保留所有关键事实、数字、名称
2. 删除冗余描述和重复内容
3. 保持原始信息的准确性
4. 使用简洁的语句

## 压缩结果"""

    try:
        compressed = await llm_client.ask(
            prompt=prompt,
            model_name=get_settings().llm_simple_model,  # 压缩用便宜模型
        )
        logger.info(
            "上下文压缩: %d chars → %d chars",
            len(content), len(compressed),
        )
        return compressed
    except Exception as e:
        logger.warning("上下文压缩失败: %s，降级为截断", e)
        # 降级: 简单截断
        return content[:max_tokens * 4]  # 粗略: 1 token ≈ 4 chars
