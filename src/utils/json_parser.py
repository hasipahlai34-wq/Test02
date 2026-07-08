"""
# ============================================================
# LLM JSON 响应解析器 (共享工具函数)
# ============================================================

处理 LLM 响应中常见的 JSON 格式问题:
- markdown code fence (```json ... ```)
- 前后的非 JSON 文本
- 不完整的 code fence
- 纯 JSON (直接解析)

此函数在以下位置被复用 (消除重复代码):
- src/graph/router.py: classify_query()
- src/agents/reviewer.py: review_answer()
- src/retrieval/adaptive.py: AdaptiveStrategy.classify()
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def parse_llm_json_response(response: str) -> dict:
    """从 LLM 响应中安全解析 JSON dict。

    自动处理常见的 LLM 输出格式问题:
    1. 纯 JSON → 直接解析
    2. markdown code fence → 提取 fence 内容后解析
    3. "json" 前缀 → 剥离后解析
    4. 混合文本中的 JSON 对象 → 提取 { } 块后解析

    Args:
        response: LLM 原始响应字符串

    Returns:
        解析后的 dict。解析失败时返回空 dict {}。
    """
    if not response or not response.strip():
        return {}

    text = response.strip()

    # 1. 尝试直接解析 (最快路径)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 剥离 markdown code fence
    if text.startswith("```"):
        lines = text.split("\n")
        # 保留第一行之后、最后一行之前的内容
        if len(lines) >= 2:
            if lines[-1].startswith("```"):
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])
        text = text.strip()

    # 3. 剥离 "json" 语言标识前缀
    if text.startswith("json"):
        text = text[4:].strip()

    # 4. 再次尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 5. 正则提取第一个 JSON 对象 (最后降级手段)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("无法从 LLM 响应中解析 JSON: %s...", response[:200])
    return {}
