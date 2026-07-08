"""
# ============================================================
# 短期记忆管理 (会话内多轮上下文)
# ← WeKnora: chat_pipeline/load_history.go — 加载最近 N 轮对话
#           Session.Messages — 消息存储
#
# 短期记忆是最基础但最关键的记记层:
# - 存储当前会话的对话历史
# - 控制上下文窗口大小 (最近 N 轮)
# - 拼接为 LLM 可用的对话历史文本
# ============================================================
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from src.types import History, Message, MessageRole, ConversationContext

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROUNDS = 10  # ← WeKnora: AgentConfig.HistoryTurns


class ShortTermMemory:
    """
    短期记忆 — 会话内多轮上下文管理
    ← WeKnora: load_history.go + Session.Messages

    使用 deque 实现固定大小的滑动窗口。
    超过最大轮数时自动丢弃最早的对话。

    用法:
        memory = ShortTermMemory(max_rounds=10)
        memory.add("用户查询", "AI回答")
        context = memory.get_context()  # → "用户: xxx\nAI: xxx\n..."
    """

    def __init__(self, max_rounds: int = DEFAULT_MAX_ROUNDS):
        self.max_rounds = max_rounds
        self._history: deque[History] = deque(maxlen=max_rounds)
        self._messages: deque[Message] = deque(maxlen=max_rounds * 2)  # 对话轮数 × 2

    @property
    def round_count(self) -> int:
        """当前对话轮数"""
        return len(self._history)

    @property
    def is_empty(self) -> bool:
        return len(self._history) == 0

    # ----------------------------------------------------------------
    # 核心操作
    # ----------------------------------------------------------------

    def add(self, query: str, answer: str = "", knowledge_refs: list = None) -> None:
        """
        添加一轮对话
        ← WeKnora: Session.Messages append → DB

        Args:
            query: 用户查询
            answer: AI 回答
            knowledge_refs: 引用的知识片段
        """
        entry = History(
            query=query,
            answer=answer,
            knowledge_refs=knowledge_refs or [],
        )
        self._history.append(entry)

        # 同时添加消息 (兼容 Message 格式)
        self._messages.append(Message(content=query, role=MessageRole.USER))
        if answer:
            self._messages.append(Message(content=answer, role=MessageRole.ASSISTANT))

        logger.debug("短期记忆: +1轮 (总计 %d 轮)", len(self._history))

    def get_last_query(self) -> Optional[str]:
        """获取最近一次查询"""
        if self._history:
            return self._history[-1].query
        return None

    def get_last_answer(self) -> Optional[str]:
        """获取最近一次回答"""
        if self._history:
            return self._history[-1].answer
        return None

    # ----------------------------------------------------------------
    # 上下文组装 (← WeKnora: load_history.go → 拼接为 Prompt)
    # ----------------------------------------------------------------

    def get_history_text(self, max_chars: int = 2000) -> str:
        """
        获取格式化的对话历史文本 (用于 LLM Prompt)
        ← WeKnora: load_history.go → 格式化为文本注入 Prompt

        Args:
            max_chars: 最大字符数限制

        Returns:
            如果无历史返回空字符串
        """
        if not self._history:
            return ""

        lines = []
        total_chars = 0

        # 从旧到新排列
        for i, entry in enumerate(self._history, 1):
            line = f"用户: {entry.query}\n"
            if entry.answer:
                line += f"AI: {entry.answer}\n"

            if total_chars + len(line) > max_chars:
                lines.append(f"\n... (前 {i-1} 轮对话已省略，共 {len(self._history)} 轮)")
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    def get_messages(self, max_messages: int = 20) -> list[dict[str, str]]:
        """
        获取消息列表 (兼容 OpenAI API 格式)

        Returns:
            [{"role": "user/assistant", "content": "..."}]
        """
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in list(self._messages)[-max_messages:]
        ]

    def get_context(self) -> ConversationContext:
        """
        获取会话上下文对象

        Returns:
            ConversationContext
        """
        return ConversationContext(
            session_id="",
            messages=list(self._messages),
            history=list(self._history),
            max_rounds=self.max_rounds,
            current_round=len(self._history),
        )

    # ----------------------------------------------------------------
    # 管理操作
    # ----------------------------------------------------------------

    def clear(self) -> None:
        """清空短期记忆 (开始新会话时)"""
        self._history.clear()
        self._messages.clear()
        logger.info("短期记忆已清空")

    def summarize_for_llm(self) -> str:
        """
        生成记忆摘要 (用于给 LLM 的上下文概览)

        Returns:
            简短的记忆摘要文本
        """
        if not self._history:
            return "（无对话历史）"

        topics = [h.query[:30] for h in self._history]
        return f"共 {len(self._history)} 轮对话，最近话题: {' → '.join(topics[-3:])}"
