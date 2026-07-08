"""
# ============================================================
# ★ 中期记忆管理 (跨会话用户偏好学习)
# ← 原项目 B 特性
# ← WeKnora 无此功能: 仅有 Session + 消息持久化，无跨会话偏好
#
# 中期记忆存储跨会话的用户偏好:
# - 偏好的回答风格 (详细 vs 简洁)
# - 关注的主题领域
# - 常用的查询模式
# - 偏好的文档
#
# 实现: SQLite 持久化 (简单的 key-value + JSON)
# ============================================================
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from src.types import UserPreference

logger = logging.getLogger(__name__)


class MediumTermMemory:
    """
    中期记忆 — 跨会话用户偏好
    ← 原项目 B 特性
    ← WeKnora 无此概念

    实现方式: SQLite 持久化的用户偏好存储。
    每完成一次会话后，异步更新用户偏好。

    面试可讲:
    "跨会话的记忆对个性化体验很关键。我用中期记忆层存储用户偏好——
    包括关注的话题、偏好的回答风格、常用的查询模式。
    这些信息在每次会话结束后被 LLM 提取并更新到 SQLite，
    下一次会话时自动加载，影响 Prompt 组装和检索策略选择。"
    """

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            from config.settings import get_settings
            db_path = get_settings().memory_medium_term_db

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化数据库表"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferred_topics TEXT DEFAULT '[]',
                    preferred_style TEXT DEFAULT 'detailed',
                    frequently_asked TEXT DEFAULT '[]',
                    document_preferences TEXT DEFAULT '[]',
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    query_count INTEGER DEFAULT 0,
                    dominant_topic TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.commit()
        logger.debug("中期记忆数据库已初始化: %s", self._db_path)

    # ----------------------------------------------------------------
    # 读写操作
    # ----------------------------------------------------------------

    def get_preferences(self, user_id: str = "default") -> UserPreference:
        """获取用户偏好"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if row is None:
            return UserPreference(user_id=user_id)

        return UserPreference(
            user_id=user_id,
            preferred_topics=json.loads(row["preferred_topics"]),
            preferred_style=row["preferred_style"],
            frequently_asked=json.loads(row["frequently_asked"]),
            updated_at=row["updated_at"],
        )

    def update_preferences(self, pref: UserPreference) -> None:
        """更新或插入用户偏好"""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO user_preferences
                    (user_id, preferred_topics, preferred_style,
                     frequently_asked, document_preferences, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(user_id) DO UPDATE SET
                    preferred_topics = excluded.preferred_topics,
                    preferred_style = excluded.preferred_style,
                    frequently_asked = excluded.frequently_asked,
                    document_preferences = excluded.document_preferences,
                    updated_at = datetime('now')
            """, (
                pref.user_id,
                json.dumps(pref.preferred_topics, ensure_ascii=False),
                pref.preferred_style,
                json.dumps(pref.frequently_asked, ensure_ascii=False),
                json.dumps(getattr(pref, 'document_preferences', []), ensure_ascii=False),
            ))
            conn.commit()
        logger.debug("用户偏好已更新: user=%s", pref.user_id)

    def record_session(
        self,
        user_id: str,
        session_id: str,
        query_count: int = 0,
        dominant_topic: str = "",
    ) -> None:
        """
        记录会话信息 (用于话题趋势分析)

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            query_count: 本轮查询数量
            dominant_topic: 主要话题
        """
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO session_history (user_id, session_id, query_count, dominant_topic) "
                "VALUES (?, ?, ?, ?)",
                (user_id, session_id, query_count, dominant_topic),
            )
            conn.commit()

    def get_common_topics(self, user_id: str = "default", limit: int = 5) -> list[str]:
        """
        获取用户最常关注的话题 (从会话历史中统计)

        Returns:
            话题列表，按频率降序
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT dominant_topic, COUNT(*) as cnt FROM session_history "
                "WHERE user_id = ? AND dominant_topic != '' "
                "GROUP BY dominant_topic ORDER BY cnt DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [row["dominant_topic"] for row in rows]

    # ----------------------------------------------------------------
    # LLM 提取偏好
    # ----------------------------------------------------------------

    async def extract_preferences_from_conversation(
        self,
        conversation: str,
        user_id: str = "default",
        llm_client=None,  # Optional[LLMClient]
    ) -> UserPreference:
        """
        使用 LLM 从对话中提取用户偏好
        ← 原项目 B 特性

        Args:
            conversation: 对话内容
            user_id: 用户 ID
            llm_client: LLM 客户端 (可选)

        Returns:
            提取的用户偏好
        """
        if llm_client is None:
            from src.models.llm import LLMClient
            llm_client = LLMClient()

        from src.utils.prompt_loader import load_prompt

        prompt = load_prompt(
            "user_preference_extraction",
            filename="summarization",
            conversation=conversation,
        )

        try:
            response = await llm_client.ask(prompt=prompt, model_name=get_settings().llm_simple_model)
            # 解析 JSON
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(response.strip())

            pref = UserPreference(
                user_id=user_id,
                preferred_topics=data.get("preferred_topics", []),
                preferred_style=data.get("preferred_style", "detailed"),
                frequently_asked=data.get("frequently_asked", []),
            )
            self.update_preferences(pref)
            return pref

        except Exception as e:
            logger.warning("偏好提取失败: %s", e)
            return self.get_preferences(user_id)

    def clear(self, user_id: str = "default") -> None:
        """清除用户的所有数据"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM session_history WHERE user_id = ?", (user_id,))
            conn.commit()
        logger.info("中期记忆已清除: user=%s", user_id)
