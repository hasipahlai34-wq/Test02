"""
# ============================================================
# Streamlit UI 工具函数
# ============================================================

本模块提供 Streamlit UI 的环境适配工具。
核心: run_async() — 在 Streamlit 事件循环环境中安全运行异步函数。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TypeVar

T = TypeVar("T")


def run_async(async_func, *args, timeout: int = 120, **kwargs) -> T:
    """在 Streamlit 环境中安全运行异步函数。

    自动检测当前事件循环状态，兼容 Streamlit 自有事件循环:
    - 事件循环未运行 → 使用 loop.run_until_complete()
    - 事件循环已运行 → 使用线程池隔离 asyncio.run()
      (避免 RuntimeError: asyncio.run() cannot be called from a running event loop)

    Args:
        async_func: 异步函数 (coroutine function，不是 coroutine object)
        *args: 传递给 async_func 的位置参数
        timeout: 线程池模式下的超时秒数 (默认 120)
        **kwargs: 传递给 async_func 的关键字参数

    Returns:
        async_func 的返回值

    Example:
        async def _fetch():
            return await some_async_api()

        result = run_async(_fetch)                # 无参函数
        result = run_async(some_coro_func, arg1)  # 带参函数
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Streamlit 事件循环已在运行 → 线程池隔离
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                async_func(*args, **kwargs),
            )
            return future.result(timeout=timeout)
    else:
        return loop.run_until_complete(async_func(*args, **kwargs))
