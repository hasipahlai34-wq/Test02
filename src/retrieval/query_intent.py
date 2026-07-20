
"""Deterministic query intent classification for structure-aware retrieval."""

from __future__ import annotations

import re
from enum import Enum


class QueryIntent(str, Enum):
    LOCAL_FACT = "local_fact"
    GLOBAL_COUNT = "global_count"
    GLOBAL_LIST = "global_list"
    TABLE_AGGREGATION = "table_aggregation"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


_COUNT_RE = re.compile(r"(\u51e0\u4e2a|\u591a\u5c11\u4e2a|\u4e00\u5171|\u603b\u5171|\u5171\u6709|\u6570\u91cf|how many|count)", re.I)
_LIST_RE = re.compile(r"(\u6709\u54ea\u4e9b|\u5217\u51fa|\u5217\u4e3e|\u5206\u522b\u662f|\u5305\u542b\u54ea\u4e9b|\u90fd\u6709|list|which)", re.I)
_AGG_RE = re.compile(r"(\u5e73\u5747|\u6700\u5927|\u6700\u5c0f|\u603b\u548c|\u5408\u8ba1|\u6392\u540d|\u6700\u9ad8|\u6700\u4f4e|avg|average|max|min|sum|rank|top)", re.I)
_SUMMARY_RE = re.compile(r"(\u603b\u7ed3|\u6982\u62ec|\u4e3b\u8981\u5185\u5bb9|\u6458\u8981|summary|summarize)", re.I)
_COMPARE_RE = re.compile(r"(\u6bd4\u8f83|\u5bf9\u6bd4|\u533a\u522b|\u5dee\u5f02|compare|difference|vs\.?)", re.I)


def classify_query_intent(query: str) -> QueryIntent:
    normalized = (query or "").strip()
    if not normalized:
        return QueryIntent.UNKNOWN
    if _AGG_RE.search(normalized):
        return QueryIntent.TABLE_AGGREGATION
    if _COUNT_RE.search(normalized):
        return QueryIntent.GLOBAL_COUNT
    if _LIST_RE.search(normalized):
        return QueryIntent.GLOBAL_LIST
    if _SUMMARY_RE.search(normalized):
        return QueryIntent.SUMMARY
    if _COMPARE_RE.search(normalized):
        return QueryIntent.COMPARISON
    return QueryIntent.LOCAL_FACT
