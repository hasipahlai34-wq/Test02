"""
# ============================================================
# 结构化日志
# ← WeKnora: internal/logger/ — 基于 zap 的结构化日志
#   我们使用 structlog + rich 实现类似的结构化日志，
#   在终端模式下提供彩色格式化输出，在文件中提供 JSON 格式。
# ============================================================

本模块配置了全项目统一的结构化日志系统。
使用 structlog 提供结构化字段，rich 提供终端美化输出。

用法:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("检索完成", query="营收增长", docs_count=5, latency_ms=120)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog
from config.settings import get_settings


def setup_logging(
    log_level: str | None = None,
    log_file: str | Path | None = None,
) -> None:
    """
    初始化全局日志配置

    Args:
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)，默认从 settings 读取
        log_file: 日志文件路径（可选），启用后同时输出到文件
    """
    settings = get_settings()
    level = (log_level or settings.log_level).upper()

    # 配置 structlog 共享处理器链:
    #   structlog → stdlib logging → RichHandler (终端) / FileHandler (文件)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,      # 合并上下文变量
        structlog.processors.add_log_level,            # 添加 level 字段
        structlog.processors.TimeStamper(fmt="iso"),   # ISO 时间戳
        structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=shared_processors,  # type: ignore[arg-type]
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 如果指定了日志文件，添加文件处理器
    if log_file:
        from logging import FileHandler, Formatter
        file_handler = FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(getattr(logging, level, logging.INFO))
        file_handler.setFormatter(
            Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logging.getLogger().addHandler(file_handler)

    # 抑制第三方库的嘈杂日志
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    获取结构化日志记录器

    Args:
        name: 通常是 __name__，用于标识日志来源模块

    Returns:
        structlog.BoundLogger 实例

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("检索开始", query="营收增长", top_k=10)
        >>> logger.error("API调用失败", model="gpt-4o", error=str(e))
    """
    return structlog.get_logger(name)


# 注意: setup_logging() 不在此自动调用
# 请在应用入口显式调用 (如 main.py):
#     from src.utils.logger import setup_logging
#     setup_logging()
