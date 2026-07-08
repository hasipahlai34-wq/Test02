"""
# ============================================================
# CSV 聚合查询处理器
# ============================================================

本模块为 CSV 文档提供 pandas 聚合查询支持。
向量检索可以回答"文档内容是什么"类语义问题，
但无法回答"薪资最高的是谁""总共多少人"等聚合问题。
此模块检测聚合查询并直接对原始 CSV 文件执行 pandas 计算。

设计要点:
- 纯函数设计，无副作用，不依赖外部状态
- 所有方法返回 Optional[str]，调用方自行处理 None
- 聚合检测使用正则模式，只对 CSV 文件生效
- 失败时静默降级，不阻断主检索流程
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 聚合关键词模式
AGGREGATE_PATTERNS: dict[str, str] = {
    "max": r"(最高|最大|最多|最贵|最老|最年长|top\s*\d+|排名前)",
    "min": r"(最低|最小|最少|最便宜|最年轻)",
    "avg": r"(平均|均值|平均数)",
    "sum": r"(合计|总和|总计)",
    "count": r"(总共|一共|多少[条行个人]|一共有多少|总共有多少|数量|人数|几个|几行|几条|几个)",
    "sort": r"(排序|升序|降序|从高到低|从低到高|排列)",
}


def is_aggregate_query(query: str) -> bool:
    """检测查询是否为聚合类查询。

    聚合查询的特征: 包含求和/求平均/排序/计数/最值等关键词。
    这些查询无法通过向量检索回答，需要 pandas 直接计算。

    Args:
        query: 用户查询文本

    Returns:
        True 如果查询包含聚合关键词
    """
    for pattern in AGGREGATE_PATTERNS.values():
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


def _detect_target_column(query: str, columns: list[str]) -> Optional[str]:
    """从查询中检测目标列名。

    两级匹配:
    1. 精确匹配: 列名完整出现在 query 中 (如 "月薪" in "月薪最高")
    2. 模糊匹配: 列名中至少有一个字符出现在 query 中 (如 "月薪" 匹配 "薪资")
       取共享字符数最多的列名

    Args:
        query: 用户查询文本
        columns: CSV 列名列表

    Returns:
        匹配的列名，或 None
    """
    # 一级: 精确子串匹配，优先长列名
    sorted_cols = sorted(columns, key=len, reverse=True)
    for col in sorted_cols:
        if col in query:
            return col

    # 二级: 模糊字符重叠匹配
    best_col = None
    best_overlap = 0
    for col in columns:
        overlap = sum(1 for ch in col if ch in query)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    # 至少共享 1 个字符才算匹配
    if best_overlap >= 1:
        return best_col
    return None


def _detect_numeric_columns(df: "pd.DataFrame") -> list[str]:
    """检测 DataFrame 中的数值列。"""
    import pandas as pd
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def _format_row(row_dict: dict) -> str:
    """将单行字典格式化为可读文本。"""
    parts = [f"{k}: {v}" for k, v in row_dict.items()]
    return "，".join(parts)


def execute_csv_aggregation(csv_path: str, query: str) -> Optional[str]:
    """对 CSV 文件执行聚合查询，返回格式化的文本结果。

    支持的聚合类型:
    - count: 计数查询（"总共有多少人"）
    - max:   最大值查询（"薪资最高的是谁"）
    - min:   最小值查询（"薪资最低的是谁"）
    - avg:   平均值查询（"平均薪资多少"）
    - sum:   总和查询（"薪资合计多少"）
    - sort:  排序查询（"按薪资从高到低排列"）

    Args:
        csv_path: CSV 文件的完整路径（必须在磁盘上存在）
        query:    用户查询文本

    Returns:
        格式化的中文聚合结果文本，或 None（文件不存在/解析失败/不匹配）
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        logger.warning("CSV 聚合: 文件不存在 %s", csv_path)
        return None

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding="gbk")
        except Exception as e:
            logger.warning("CSV 聚合: 读取失败 %s: %s", csv_path, e)
            return None
    except Exception as e:
        logger.warning("CSV 聚合: 读取失败 %s: %s", csv_path, e)
        return None

    if df.empty:
        return None

    numeric_cols = _detect_numeric_columns(df)
    result_parts: list[str] = []

    # ── 计数查询 ──
    if re.search(AGGREGATE_PATTERNS["count"], query, re.IGNORECASE):
        result_parts.append(f"共 {len(df)} 条记录")

    # ── 最大值查询 ──
    if re.search(AGGREGATE_PATTERNS["max"], query, re.IGNORECASE):
        target_col = _detect_target_column(query, numeric_cols)
        if target_col:
            top_row = df.nlargest(1, target_col).iloc[0].to_dict()
            result_parts.append(f"{target_col}最高: {_format_row(top_row)}")
        elif numeric_cols:
            # 未指定列名，取第一个数值列
            col = numeric_cols[0]
            top_row = df.nlargest(1, col).iloc[0].to_dict()
            result_parts.append(f"{col}最高: {_format_row(top_row)}")

    # ── 最小值查询 ──
    if re.search(AGGREGATE_PATTERNS["min"], query, re.IGNORECASE):
        target_col = _detect_target_column(query, numeric_cols)
        if target_col:
            bottom_row = df.nsmallest(1, target_col).iloc[0].to_dict()
            result_parts.append(f"{target_col}最低: {_format_row(bottom_row)}")
        elif numeric_cols:
            col = numeric_cols[0]
            bottom_row = df.nsmallest(1, col).iloc[0].to_dict()
            result_parts.append(f"{col}最低: {_format_row(bottom_row)}")

    # ── 平均值查询 ──
    if re.search(AGGREGATE_PATTERNS["avg"], query, re.IGNORECASE):
        target_col = _detect_target_column(query, numeric_cols)
        if target_col:
            avg_val = df[target_col].mean()
            result_parts.append(f"{target_col}平均值: {avg_val:.2f}")
        elif numeric_cols:
            col = numeric_cols[0]
            avg_val = df[col].mean()
            result_parts.append(f"{col}平均值: {avg_val:.2f}")

    # ── 总和查询 ──
    if re.search(AGGREGATE_PATTERNS["sum"], query, re.IGNORECASE):
        target_col = _detect_target_column(query, numeric_cols)
        if target_col:
            total = df[target_col].sum()
            result_parts.append(f"{target_col}总和: {total:.2f}")
        elif numeric_cols:
            col = numeric_cols[0]
            total = df[col].sum()
            result_parts.append(f"{col}总和: {total:.2f}")

    # ── 排序查询 ──
    if re.search(AGGREGATE_PATTERNS["sort"], query, re.IGNORECASE):
        target_col = _detect_target_column(query, df.columns.tolist())
        if not target_col and numeric_cols:
            target_col = numeric_cols[0]  # 降级: 取第一个数值列
        if target_col:
            ascending = "升序" in query or "从低到高" in query
            sorted_df = df.sort_values(by=target_col, ascending=ascending)
            direction = "升序" if ascending else "降序"
            lines = [f"按{target_col}{direction}排列:"]
            for _, row in sorted_df.iterrows():
                lines.append(f"  {_format_row(row.to_dict())}")
            result_parts.append("\n".join(lines))

    if not result_parts:
        return None

    return "\n".join(result_parts)


def find_csv_sources_from_docs(documents: list) -> list[str]:
    """从检索到的文档列表中提取 CSV 文件来源路径。

    遍历文档的 metadata，找到 source 字段中以 .csv 结尾的文件路径。
    去重后返回。

    Args:
        documents: 检索结果中的 Document 列表（来自 retrieved_docs）

    Returns:
        去重后的 CSV 文件完整路径列表
    """
    csv_paths: list[str] = []
    seen: set[str] = set()

    for doc in documents:
        source = None
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            source = doc.metadata.get("source", "")
        if not source:
            continue
        if not source.lower().endswith(".csv"):
            continue
        if source in seen:
            continue
        seen.add(source)
        csv_paths.append(source)

    return csv_paths
