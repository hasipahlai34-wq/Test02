"""
# ============================================================
# ★ 长期记忆测试 (Embedding 语义搜索 + 关键词降级)
#
# 验证:
# 1. 关键词 fallback 正常工作 (无 Embedding 时)
# 2. 语义搜索 — 向量相似度匹配 "机器学习" → "深度学习"
# 3. Embedding 不可用时自动降级
# 4. forget() 删除条目 + 清理索引
# 5. add() 计算并存储 embedding
# ============================================================
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.types import MemoryEntry, MemoryType


# ================================================================
# 辅助
# ================================================================


def _make_entry(content: str, importance: float = 0.5) -> MemoryEntry:
    return MemoryEntry(
        memory_type=MemoryType.LONG_TERM,
        content=content,
        importance=importance,
    )


# ================================================================
# 测试 1: 关键词 fallback (无 Embedding)
# ================================================================


def test_keyword_search_fallback():
    """
    ★ Embedding 不可用时 → 关键词 Jaccard 搜索正常工作

    验证:
    - 精确关键词匹配返回高分结果
    - 无匹配时返回空列表
    - 重要性影响排序
    """
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    # 直接操作内部列表 (绕过 async add)
    ltm._entries = [
        _make_entry("Python 是一种流行的编程语言", importance=0.9),
        _make_entry("机器学习使用神经网络进行训练", importance=0.8),
        _make_entry("Java 广泛用于企业应用开发", importance=0.5),
    ]

    # 搜索 "Python" → 应命中第一个条目
    results = ltm.search("Python 编程")
    assert len(results) > 0, "关键词搜索应返回结果"
    assert "Python" in results[0].content, "第一条应包含 Python"

    # 搜索不相关的 → 应返回空或低分结果
    results = ltm.search("量子计算")
    assert len(results) == 0 or all(
        "量子" not in e.content for e in results
    ), "不应匹配不相关内容"


def test_keyword_empty_query_returns_by_importance():
    """空查询 → 按重要性降序返回"""
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    ltm._entries = [
        _make_entry("内容 A", importance=0.3),
        _make_entry("内容 B", importance=0.9),
        _make_entry("内容 C", importance=0.6),
    ]

    results = ltm.search("", top_k=2)
    assert len(results) == 2
    assert results[0].importance == 0.9  # 最高重要性在前
    assert results[1].importance == 0.6


# ================================================================
# 测试 2: 语义搜索 (Mock EmbeddingModel)
# ================================================================


@pytest.mark.asyncio
async def test_add_computes_embedding():
    """
    ★ add() 应计算 Embedding 并存储到 MemoryEntry.embedding

    模拟 EmbeddingModel 返回固定向量，验证 embedding 字段被填充。
    """
    from src.memory.long_term import LongTermMemory

    mock_embed_model = MagicMock()
    mock_embed_model.provider = "openai"
    mock_embed_model.dimensions = 1536
    mock_embed_model._ensure_model = AsyncMock()
    mock_embed_model.embed_single = AsyncMock(
        return_value=[0.1, 0.2, 0.3]  # 固定向量
    )

    ltm = LongTermMemory()
    # 手动设置 (绕过 _ensure_index 中的 ChromaDB 初始化)
    ltm._embedding_model = mock_embed_model
    ltm._embedding_available = True
    ltm._index_ready = True

    entry = await ltm.add("机器学习是人工智能的核心")

    assert entry.embedding is not None, "应计算 Embedding"
    assert entry.embedding == [0.1, 0.2, 0.3], "Embedding 应为 Mock 返回值"
    assert len(ltm._entries) == 1
    assert ltm._entries[0].embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_semantic_search_with_embeddings():
    """
    ★ 语义搜索: "机器学习" 应匹配到 "深度学习"、"神经网络" 相关内容

    使用真实的余弦相似度计算，模拟 EmbeddingModel 返回有语义关系的向量。
    """
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()

    # 构造有语义关系的向量:
    #   "机器学习" query → [0.9, 0.1, 0.0]
    #   "深度学习神经网络" → [0.8, 0.2, 0.0]  (高相似度)
    #   "Python编程语言"   → [0.1, 0.9, 0.0]  (低相似度)
    #   "Java企业应用"     → [0.0, 0.1, 0.9]  (极低相似度)
    ltm._entries = [
        _make_entry("深度学习与神经网络训练", importance=0.8),
        _make_entry("Python 是一种流行的编程语言", importance=0.5),
        _make_entry("Java 广泛用于企业级应用开发", importance=0.6),
    ]
    ltm._entries[0].embedding = [0.8, 0.2, 0.0]
    ltm._entries[1].embedding = [0.1, 0.9, 0.0]
    ltm._entries[2].embedding = [0.0, 0.1, 0.9]
    ltm._id_index = {e.id: e for e in ltm._entries}

    # Mock ChromaDB collection 返回按向量排序的结果
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    # query 应返回: 深度学习 (最近) > Python > Java
    mock_collection.query.return_value = {
        "ids": [
            [ltm._entries[0].id, ltm._entries[1].id, ltm._entries[2].id]
        ],
        "documents": [
            ["深度学习与神经网络训练", "Python...", "Java..."]
        ],
        "metadatas": [
            [{"importance": 0.8}, {"importance": 0.5}, {"importance": 0.6}]
        ],
        "distances": [[0.01, 0.5, 0.95]],
    }
    ltm._collection = mock_collection

    # Mock EmbeddingModel
    mock_embed_model = MagicMock()
    mock_embed_model.provider = "mock"
    mock_embed_model.dimensions = 3
    mock_embed_model.embed_single = AsyncMock(
        return_value=[0.9, 0.1, 0.0]  # query: "机器学习"
    )
    ltm._embedding_model = mock_embed_model
    ltm._embedding_available = True

    # Mock _compute_embedding_sync 避免 asyncio.run 问题
    with patch.object(
        ltm, "_compute_embedding_sync", return_value=[0.9, 0.1, 0.0]
    ):
        results = ltm.search("机器学习", top_k=3)

    assert len(results) >= 1, "语义搜索应返回结果"
    # 第一个结果应是 "深度学习" (高语义相似度)
    assert "深度学习" in results[0].content or "神经网络" in results[0].content, (
        f"语义搜索应优先返回语义相关条目，实际第一条: '{results[0].content}'"
    )


def test_semantic_search_falls_back_when_collection_none():
    """
    ★ ChromaDB collection 为 None → 降级到关键词搜索

    验证: 即使 _embedding_available=True，没有 collection 也应降级。
    """
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    ltm._entries = [
        _make_entry("深度学习神经网络", importance=0.9),
        _make_entry("Python编程语言", importance=0.5),
    ]
    ltm._id_index = {e.id: e for e in ltm._entries}
    ltm._embedding_available = True
    ltm._collection = None  # ← 无 ChromaDB

    results = ltm.search("深度学习")
    assert len(results) > 0
    # 关键词仍能匹配到
    assert "深度学习" in results[0].content


def test_semantic_search_falls_back_when_embedding_unavailable():
    """
    ★ EmbeddingModel 不可用 → 降级到关键词搜索

    模拟 _ensure_index 失败后 _embedding_available=False 的场景。
    """
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    ltm._entries = [
        _make_entry("深度学习与神经网络", importance=0.8),
        _make_entry("微积分与线性代数", importance=0.6),
    ]
    ltm._id_index = {e.id: e for e in ltm._entries}
    ltm._embedding_available = False  # ← Embedding 不可用

    # 搜索 — 应降级为关键词匹配
    results = ltm.search("深度学习", top_k=2)
    assert len(results) >= 1
    # 关键词搜索能匹配到"深度学习"
    assert "深度学习" in results[0].content


# ================================================================
# 测试 3: forget() 清理索引
# ================================================================


def test_forget_removes_from_index():
    """
    ★ forget() 删除条目时应同步清理 ChromaDB

    验证:
    - _entries 长度减少
    - _id_index 清理
    - ChromaDB collection.delete() 被调用
    """
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    entry = _make_entry("测试记忆")
    ltm._entries = [entry]
    ltm._id_index = {entry.id: entry}

    mock_collection = MagicMock()
    ltm._collection = mock_collection

    result = ltm.forget(0)
    assert result is not None
    assert len(ltm._entries) == 0
    assert entry.id not in ltm._id_index

    # ChromaDB delete 被调用
    mock_collection.delete.assert_called_once_with(ids=[entry.id])


def test_forget_invalid_index_returns_none():
    """无效索引 → 返回 None，不崩溃"""
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    ltm._entries = [_make_entry("测试")]

    result = ltm.forget(99)  # 越界
    assert result is None
    assert len(ltm._entries) == 1  # 未删除


# ================================================================
# 测试 4: 边界场景
# ================================================================


def test_search_empty_memory_returns_empty():
    """空记忆库 → 返回空列表"""
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    assert ltm.search("任何查询") == []


def test_get_context_for_llm():
    """get_context_for_llm 应正确格式化"""
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    ltm._entries = [
        _make_entry("用户偏好简洁回答", importance=0.9),
        _make_entry("上次讨论了Python性能优化", importance=0.7),
    ]

    context = ltm.get_context_for_llm(max_entries=2)
    assert "长期记忆" in context
    assert "简洁回答" in context
    assert "Python性能优化" in context


def test_get_by_importance():
    """按重要性过滤"""
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory()
    ltm._entries = [
        _make_entry("高重要性", importance=0.9),
        _make_entry("低重要性", importance=0.2),
        _make_entry("中等重要性", importance=0.5),
    ]

    high = ltm.get_by_importance(0.5)
    assert len(high) == 2
    assert all(e.importance >= 0.5 for e in high)
