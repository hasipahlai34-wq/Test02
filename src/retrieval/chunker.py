"""Compatibility wrappers for document chunking."""

from langchain_core.documents import Document

from src.ingestion.chunker import (
    ChunkingStrategy,
    auto_chunk,
    auto_chunk_legacy,
    chunk_documents,
)


def chunk_document(
    document: Document,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    return chunk_documents(
        [document],
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


__all__ = [
    "ChunkingStrategy",
    "auto_chunk",
    "auto_chunk_legacy",
    "chunk_document",
    "chunk_documents",
]
