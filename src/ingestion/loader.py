"""
# ============================================================
# 文档加载器 (多格式支持)
# ← WeKnora: internal/docreader/ — gRPC 独立文档解析服务
#   我们简化为 LangChain Document Loaders 进程内加载
#   支持: PDF / Word / Markdown / TXT / CSV
# ============================================================

本模块负责:
- 从本地文件加载各类文档格式
- 提取纯文本内容 + 基础元数据
- 统一的 Document 接口 (LangChain Document 格式)

设计要点:
- 使用 LangChain 的社区 Document Loaders，而非 WeKnora 的独立 gRPC 服务
- 支持 5 种常见格式 (PDF/Word/Markdown/TXT/CSV)
- 自动检测文件格式并选择合适的 Loader
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# 支持的文件格式
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF 文档",
    ".docx": "Word 文档",
    ".doc": "Word 文档 (旧格式)",
    ".md": "Markdown 文档",
    ".markdown": "Markdown 文档",
    ".txt": "纯文本文档",
    ".csv": "CSV 数据表格",
}


def is_supported(filepath: str | Path) -> bool:
    """检查文件格式是否支持"""
    ext = Path(filepath).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def detect_loader(filepath: str | Path) -> str:
    """
    根据文件扩展名检测应使用的 Loader 类型

    Returns:
        Loader 类型标识: "pdf" / "docx" / "markdown" / "text" / "csv"
    """
    ext = Path(filepath).suffix.lower()

    loader_map = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".csv": "csv",
    }
    return loader_map.get(ext, "text")


# ================================================================
# 各格式 Loader 实现
# ================================================================


async def load_pdf(filepath: str | Path) -> list[Document]:
    """
    加载 PDF 文档
    ← WeKnora: docreader/pdf_parser.go — 使用 PyMuPDF (fitz)
    """
    try:
        from langchain_community.document_loaders import PyMuPDFLoader

        loader = PyMuPDFLoader(str(filepath))
        docs = loader.load()
        logger.info("PDF 加载完成: %s → %d 页", Path(filepath).name, len(docs))
        return docs
    except ImportError:
        # 降级: 使用 PyPDFLoader
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(filepath))
        docs = loader.load()
        logger.info("PDF 加载完成 (PyPDF): %s → %d 页", Path(filepath).name, len(docs))
        return docs


async def load_docx(filepath: str | Path) -> list[Document]:
    """
    加载 Word 文档
    ← WeKnora: docreader/docx_parser.go
    """
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(str(filepath))
    docs = loader.load()
    logger.info("DOCX 加载完成: %s → %d 段", Path(filepath).name, len(docs))
    return docs


async def load_markdown(filepath: str | Path) -> list[Document]:
    """
    加载 Markdown 文档
    ← WeKnora: docreader/markdown_parser.go

    使用 UnstructuredMarkdownLoader 保留标题层级等结构信息
    """
    from langchain_community.document_loaders import UnstructuredMarkdownLoader

    loader = UnstructuredMarkdownLoader(str(filepath), mode="elements")
    try:
        docs = loader.load()
    except Exception:
        # 降级: 简单文本加载
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(filepath), encoding="utf-8")
        docs = loader.load()

    logger.info("Markdown 加载完成: %s → %d 元素", Path(filepath).name, len(docs))
    return docs


async def load_text(filepath: str | Path) -> list[Document]:
    """加载纯文本文档"""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(filepath), encoding="utf-8")
    docs = loader.load()
    logger.info("TXT 加载完成: %s → %d 行", Path(filepath).name, len(docs))
    return docs


async def load_csv(filepath: str | Path) -> list[Document]:
    """加载 CSV 数据表格"""
    from langchain_community.document_loaders import CSVLoader

    loader = CSVLoader(str(filepath), encoding="utf-8")
    docs = loader.load()
    logger.info("CSV 加载完成: %s → %d 行", Path(filepath).name, len(docs))
    return docs


# ================================================================
# 统一加载接口
# ================================================================


async def load_document(filepath: str | Path) -> list[Document]:
    """
    统一的文档加载入口 — 自动检测格式并选择合适的 Loader
    ← WeKnora: docreader/ 统一入口 → 我们简化为函数式调度

    Args:
        filepath: 文档文件路径

    Returns:
        LangChain Document 列表

    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"文档不存在: {filepath}")

    if not is_supported(filepath):
        raise ValueError(
            f"不支持的文件格式: {filepath.suffix}，"
            f"支持的格式: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )

    loader_type = detect_loader(filepath)

    loader_map = {
        "pdf": load_pdf,
        "docx": load_docx,
        "markdown": load_markdown,
        "text": load_text,
        "csv": load_csv,
    }

    loader_func = loader_map.get(loader_type, load_text)

    # 为每个 Document 补充来源信息
    docs = await loader_func(filepath)
    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = filepath.name
        doc.metadata["file_path"] = str(filepath.resolve())
        doc.metadata["file_type"] = filepath.suffix.lower()

    return docs


async def load_documents(
    filepaths: list[str | Path],
    recursive: bool = False,
) -> dict[str, list[Document]]:
    """
    批量加载多个文档

    Args:
        filepaths: 文档路径列表
        recursive: 是否递归加载目录中的文档

    Returns:
        {文件名: [Document列表]}
    """
    result: dict[str, list[Document]] = {}
    paths_to_load: list[Path] = []

    for fp in filepaths:
        p = Path(fp)
        if p.is_file():
            paths_to_load.append(p)
        elif p.is_dir() and recursive:
            for ext in SUPPORTED_EXTENSIONS:
                paths_to_load.extend(p.rglob(f"*{ext}"))
        elif p.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                paths_to_load.extend(p.glob(f"*{ext}"))

    for filepath in paths_to_load:
        try:
            docs = await load_document(filepath)
            result[filepath.name] = docs
        except Exception as e:
            logger.error("加载文档失败: %s — %s", filepath, e)

    logger.info("批量加载完成: %d/%d 个文档成功", len(result), len(paths_to_load))
    return result
