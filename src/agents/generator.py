"""
# ============================================================
# 生成 Agent 节点
# ← WeKnora: internal/agent/think.go + chat_pipeline/chat_completion.go
#            + chat_completion_stream.go
#   - LLM 调用 (OpenAI SDK)
#   - 流式输出 (SSE → AsyncGenerator)
#   - Context 窗口管理
#
#   我们的实现:
#   - 组装 System Prompt + Context + 用户问题
#   - 按查询复杂度选择模型 (simple→mini, complex→gpt-4o)
#   - 支持流式和非流式两种输出模式
# ============================================================

本模块是生成 Agent 的 LangGraph 节点实现。
负责将检索结果组装为 LLM 上下文并生成最终答案。
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from config.settings import get_settings
from src.graph.state import GraphState
from src.models.llm import LLMClient, get_llm_client
from src.types import Document
from src.utils.prompt_loader import load_prompt_with_default
from src.utils.token_manager import TokenBudget

logger = logging.getLogger(__name__)


# ================================================================
# 上下文组装
# ================================================================


def _build_context(
    documents: list[Document],
    max_tokens: int = 6000,
) -> str:
    """
    将检索到的文档组装为 LLM 上下文
    ← WeKnora: chat_pipeline/into_chat_message.go — 组装上下文
               token/compress.go — Token 预算管理

    策略:
    - 按相关度分数降序排列
    - 每个文档标注序号和来源
    - 超过 Token 预算时截断末尾文档

    Args:
        documents: 检索到的文档列表
        max_tokens: 最大 Token 预算 (粗略按 1 token ≈ 2 中文字符估算)

    Returns:
        格式化的 Markdown 上下文文本
    """
    if not documents:
        return "（未检索到相关文档内容，请根据常识回答并明确告知用户）"

    parts = []
    char_limit = max_tokens * 2  # 粗略: 1 token ≈ 2 chars
    current_chars = 0

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", doc.source or "未知来源")
        content = doc.content[:1500]  # 每个文档最多取 1500 字符（C1/D3 修复：减少截断遗漏）
        metadata_hints = []
        for key, label in (
            ("retrieval_scope", "检索范围"),
            ("query_intent", "查询意图"),
            ("element_type", "结构类型"),
            ("section_path", "章节路径"),
            ("row_range", "行范围"),
        ):
            value = doc.metadata.get(key)
            if value:
                metadata_hints.append(f"{label}: {value}")
        metadata_line = f"**结构元数据**: {'; '.join(metadata_hints)}\n" if metadata_hints else ""

        part = (
            f"### 文档片段 {i}\n"
            f"**来源**: {source}\n"
            f"**相关度**: {doc.score:.2f}\n"
            f"{metadata_line}"
            f"\n{content}\n"
        )

        if current_chars + len(part) > char_limit:
            remaining = len(documents) - i
            if remaining > 0:
                parts.append(f"\n*(...还有 {remaining} 个文档片段因长度限制未展示)*")
            break

        parts.append(part)
        current_chars += len(part)

    return "\n---\n".join(parts)


# ================================================================
# 生成节点
# ================================================================


async def generate_answer(
    state: GraphState,
    stream: bool = False,
) -> dict:
    """
    LangGraph 生成节点: 组装 Prompt → LLM 生成答案
    ← WeKnora: chat_completion.go + chat_completion_stream.go

    Args:
        state: 当前 GraphState
        stream: 是否流式输出

    Returns:
        state 部分更新 (generated_answer, answer_stream)
    """
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    complexity = state.get("complexity", "medium")

    if complexity != "simple" and not docs:
        message = (
            "抱歉，未能在您上传的文档中找到相关信息。"
            "请确认文档已成功索引（检查左侧'已索引文档'列表），或尝试换个问法。"
        )
        return {
            "context_prompt": "",
            "generated_answer": message,
            "completed": True,
        }

    # Step 1: 组装上下文
    contexts = _build_context(docs)

    # Step 2: 加载并渲染系统提示词 (按复杂度分级)
    system_prompt = load_prompt_with_default(
        "system_prompt",
        contexts=contexts,
        query=query,
        complexity=complexity,
        language=get_settings().default_language,
    )

    # Step 3: 按复杂度选择模型 (★ 单一数据源: Settings → TokenBudget)
    llm = get_llm_client()  # ★ M2: 复用单例
    model_name = TokenBudget.model_for_complexity(complexity)

    # Step 4: 准备消息
    messages = [{"role": "user", "content": query}]

    logger.info(
        "生成Agent: complexity=%s model=%s docs=%d context_chars=%d",
        complexity, model_name, len(docs), len(contexts),
    )

    try:
        if stream:
            # 流式模式: 返回 AsyncGenerator
            async_gen = llm.generate_stream(
                messages=messages,
                system_prompt=system_prompt,
                model_name=model_name,
            )
            return {
                "system_prompt": system_prompt,
                "context_prompt": contexts,
                "answer_stream": async_gen,
            }
        else:
            # 同步模式: 等待完整结果
            answer = await llm.generate(
                messages=messages,
                system_prompt=system_prompt,
                model_name=model_name,
            )
            logger.info("生成Agent: answer=%d chars", len(answer))
            return {
                "system_prompt": system_prompt,
                "context_prompt": contexts,
                "generated_answer": answer,
                "completed": True,
            }
    except (ConnectionError, TimeoutError) as e:
        # 网络/LLM 服务不可用 → 返回用户友好的错误信息
        logger.warning("[generate] LLM 服务不可用: %s", e)
        return {
            "system_prompt": system_prompt,
            "context_prompt": contexts,
            "generated_answer": "抱歉，AI 服务暂时不可用，请稍后重试。",
            "completed": True,
            "error": f"网络异常: {e}",
        }
    except (ValueError, KeyError) as e:
        # 模板渲染错误 → 属于代码 bug，应抛出
        logger.error("[generate] 模板渲染失败: %s", e, exc_info=True)
        raise
    except Exception as e:
        # 其他未预期异常 → 记录 critical 并返回错误信息
        logger.critical("[generate] 生成异常: %s", e, exc_info=True)
        return {
            "system_prompt": system_prompt,
            "context_prompt": contexts,
            "generated_answer": f"生成答案时发生内部错误，请联系管理员。",
            "completed": True,
            "error": f"内部错误: {type(e).__name__}",
        }


async def generate_answer_stream(
    state: GraphState,
) -> AsyncGenerator[str, None]:
    """
    流式生成答案 — 逐 token yield
    ← WeKnora: chat_completion_stream.go SSE 流式

    Yields:
        每个 token 的文本片段
    """
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    complexity = state.get("complexity", "medium")

    contexts = _build_context(docs)
    system_prompt = load_prompt_with_default(
        "system_prompt",
        contexts=contexts,
        query=query,
        complexity=complexity,
        language=get_settings().default_language,
    )

    llm = get_llm_client()  # ★ M2: 复用单例
    model_name = TokenBudget.model_for_complexity(complexity)

    messages = [{"role": "user", "content": query}]

    async for chunk in llm.generate_stream(
        messages=messages,
        system_prompt=system_prompt,
        model_name=model_name,
    ):
        yield chunk
