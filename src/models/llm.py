"""
# ============================================================
# LLM 统一封装 (OpenAI 兼容协议)
# ← WeKnora: internal/models/chat/remote_api.go
#   RemoteAPIChat 实现了完整的 OpenAI SDK 封装，处理:
#   - 多 Provider 适配 (Azure, 自定义 endpoint, 各种厂商)
#   - 流式/非流式统一接口
#   - 重试逻辑
#   - Thinking 模式
#
#   我们的简化版:
#   - 仅支持 OpenAI 兼容协议 (覆盖 OpenAI/DeepSeek/Ollama 三家)
#   - 保留流式输出 + AsyncGenerator
#   - 保留重试 + 超时 + 优雅降级
#   - 新增: 按查询复杂度选择模型 (简单→mini, 复杂→gpt-4o)
#   - ★ M6: model_name 级 client 缓存 (避免 ChatOpenAI 连接池碎片化)
# ============================================================

本模块提供:
- LLMClient: 统一的 LLM 调用接口
- 支持 OpenAI / DeepSeek / Ollama / vLLM 等兼容后端
- 同步和流式两种调用模式
- 自动重试 + 超时 + 降级
- ★ 启动时 API Key 验证 + 友好错误提示
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
)

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# ★ 模块级 retry 配置 (从 settings 读取，启动时确定)
_retry_cfg = get_settings()
_default_retry = retry(
    stop=stop_after_attempt(_retry_cfg.llm_max_retries),
    wait=wait_fixed(3),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
)

# ★ M2 修复: LLMClient 模块级单例 (避免重复创建连接)
_llm_client_instance: Optional["LLMClient"] = None


def get_llm_client() -> "LLMClient":
    """
    ★ 获取 LLMClient 全局单例

    每个 LangGraph 节点不需要创建自己的 LLMClient，
    统一复用此实例，避免多个 ChatOpenAI 连接池碎片化。

    Raises:
        ValueError: 如果 LLM_API_KEY 未配置（含占位值）
    """
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
        logger.info("LLMClient 单例已创建")
    return _llm_client_instance


class LLMClient:
    """
    统一的 LLM 客户端 (OpenAI 兼容协议)
    ← WeKnora: internal/models/chat/remote_api.go RemoteAPIChat

    特性:
    - 支持 OpenAI / DeepSeek / Ollama / vLLM (通过 OpenAI 兼容 base_url)
    - 同步生成 + 流式生成
    - 自动重试 (指数退避)
    - 超时控制 + 优雅降级
    - ★ M6: model_name 级 client 缓存，避免重复创建 ChatOpenAI
    - ★ 启动时 API Key 验证
    """

    # ★ M6: 类级别 client 缓存，按 (model, temperature, max_tokens, timeout, max_retries) 索引
    _client_cache: dict[str, ChatOpenAI] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """
        ★ 清空客户端缓存

        用途:
        - 测试环境重置
        - 配置热更新后刷新连接
        """
        cls._client_cache.clear()
        logger.info("LLMClient 缓存已清空")

    def __init__(
        self,
        model_name: str | None = None,
        settings: Settings | None = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            model_name: 模型名称，默认从 settings 取 llm_default_model
            settings: 全局配置对象

        Raises:
            ValueError: API Key 未配置时给出友好提示
        """
        self._settings = settings or get_settings()
        self.model_name = model_name or self._settings.llm_default_model

        # ★ 启动时 API Key 检查 —— 友好报错而非运行时崩溃
        api_key = self._settings.llm_api_key
        if api_key in ("", "sk-placeholder", "sk-your-api-key-here"):
            raise ValueError(
                "\n[ERROR] LLM API Key is not configured.\n"
                "Please follow these steps:\n"
                "  1. cp .env.example .env\n"
                "  2. Edit .env and set LLM_API_KEY=your-key\n"
                "  3. For Ollama: LLM_BASE_URL=http://localhost:11434/v1\n"
                "  4. Re-run the application\n"
                f"Current LLM_API_KEY = '{api_key}'\n"
                f"Current LLM_BASE_URL = '{self._settings.llm_base_url}'"
            )

        # 默认客户端通过缓存获取
        self._client = self._get_or_create_client(self.model_name)

        # 惰性属性 (首次访问时通过缓存创建)
        self._simple_client: Optional[ChatOpenAI] = None
        self._complex_client: Optional[ChatOpenAI] = None

    # ----------------------------------------------------------------
    # ★ M6: 按 model_name 缓存 ChatOpenAI 实例
    # ----------------------------------------------------------------

    @staticmethod
    def _make_cache_key(
        model_name: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        max_retries: int,
    ) -> str:
        """构建缓存键"""
        return f"{model_name}|t={temperature}|mt={max_tokens}|to={timeout}|mr={max_retries}"

    def _get_or_create_client(
        self,
        model_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> ChatOpenAI:
        """
        ★ 从缓存获取或创建 ChatOpenAI 实例

        Args:
            model_name: 模型名称
            temperature: 温度（None 则用 settings 默认值）
            max_tokens: 最大 Token（None 则用 settings 默认值）
            timeout: 超时秒数（None 则用 settings 默认值）
            max_retries: 最大重试次数（None 则用 settings 默认值）

        Returns:
            缓存的 ChatOpenAI 实例
        """
        t = temperature if temperature is not None else self._settings.llm_temperature
        mt = max_tokens if max_tokens is not None else self._settings.llm_max_tokens
        to = timeout or self._settings.llm_timeout
        mr = max_retries if max_retries is not None else self._settings.llm_max_retries

        key = self._make_cache_key(model_name, t, mt, to, mr)
        if key not in self._client_cache:
            self._client_cache[key] = ChatOpenAI(
                model=model_name,
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_base_url,
                temperature=t,
                max_tokens=mt,
                timeout=to,
                max_retries=mr,
            )
            logger.debug("ChatOpenAI 缓存创建: model=%s", model_name)
        return self._client_cache[key]

    # ----------------------------------------------------------------
    # 按复杂度选择模型 (★ 原项目 B 特性)
    # ----------------------------------------------------------------

    @property
    def simple_client(self) -> ChatOpenAI:
        """简单模型 → 用于简单查询 (零温度、短输出)"""
        if self._simple_client is None:
            self._simple_client = self._get_or_create_client(
                model_name=self._settings.llm_simple_model,
                temperature=0.0,   # 简单问题零温度保证确定性
                max_tokens=500,    # 简单回答 500 tokens 足够
                max_retries=1,     # 简单查询少重试
            )
        return self._simple_client

    @property
    def complex_client(self) -> ChatOpenAI:
        """复杂模型 → 用于复杂多步推理"""
        if self._complex_client is None:
            self._complex_client = self._get_or_create_client(
                model_name=self._settings.llm_complex_model,
            )
        return self._complex_client

    def get_client_for_complexity(self, complexity: str) -> ChatOpenAI:
        """
        ★ 根据查询复杂度选择模型 ← 原项目 B 动态模型选择

        Args:
            complexity: "simple" / "medium" / "complex"

        Returns:
            对应等级的 ChatOpenAI 客户端
        """
        if complexity == "simple":
            return self.simple_client
        elif complexity == "complex":
            return self.complex_client
        else:
            return self._client  # medium → 默认

    # ----------------------------------------------------------------
    # 核心调用接口 (← WeKnora: remote_api.go ChatCompletion / ChatCompletionStream)
    # ----------------------------------------------------------------

    @_default_retry
    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        同步生成回答 (非流式)
        ← WeKnora: RemoteAPIChat.Chat() — 非流式完成

        Args:
            messages: [{"role": "user/assistant", "content": "..."}]
            system_prompt: 系统提示词
            model_name: 覆盖默认模型
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大 Token

        Returns:
            LLM 生成的完整文本
        """
        if model_name and model_name != self.model_name:
            client = self._get_or_create_client(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            client = self._client

        lc_messages: list[BaseMessage] = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            # assistant 消息的处理按需添加

        logger.debug("LLM call: model=%s, messages=%d", client.model_name, len(lc_messages))
        response = await client.ainvoke(lc_messages)
        content = str(response.content)
        if not content.strip():
            logger.warning("LLM returned empty content; retrying once: model=%s", client.model_name)
            response = await client.ainvoke(lc_messages)
            content = str(response.content)
        try:
            from src.utils.observability import get_token_tracker

            input_text = (system_prompt or "") + "\n".join(m["content"] for m in messages)
            input_tokens = max(1, len(input_text) // 4)
            output_tokens = max(1, len(content) // 4)
            get_token_tracker().record(
                step="llm.generate",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=client.model_name,
            )
        except Exception as e:
            logger.debug("Token tracking skipped: %s", e)
        logger.debug("LLM response: len=%d", len(content))
        return content

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        model_name: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式生成回答
        ← WeKnora: RemoteAPIChat.ChatCompletionStream() — SSE 流式输出

        Args:
            messages: 对话消息
            system_prompt: 系统提示词
            model_name: 模型名称

        Yields:
            生成的文本片段 (delta)
        """
        if model_name and model_name != self.model_name:
            client = self._get_or_create_client(model_name=model_name)
        else:
            client = self._client

        lc_messages: list[BaseMessage] = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))

        logger.debug("LLM stream: model=%s", client.model_name)
        async for chunk in client.astream(lc_messages):
            content = str(chunk.content) if chunk.content else ""
            if content:
                yield content

    # ----------------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------------

    async def ask(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
    ) -> str:
        """
        单次问答 (简化接口)

        Args:
            prompt: 用户问题
            system_prompt: 系统提示词
            model_name: 模型名称

        Returns:
            LLM 回答
        """
        return await self.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            model_name=model_name,
        )

    async def ask_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """单次流式问答"""
        async for chunk in self.generate_stream(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            model_name=model_name,
        ):
            yield chunk
