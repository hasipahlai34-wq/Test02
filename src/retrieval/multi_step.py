"""
# ============================================================
# 多步迭代检索策略 (MultiStepStrategy)
# ← Adaptive-RAG 论文: 复杂查询需要多步检索 → 迭代 推理+检索
# ← WeKnora: engine.go ReAct 循环 (think → act → observe)
#   但我们这里不是 ReAct Agent 的工具调用，而是检索策略层面的迭代
#
# 流程:
#   Rewrite Query → Search → Evaluate → [不充分? 改写→搜索→评估] → Return
#   最多迭代 3 轮，每轮重新评估检索质量
# ============================================================

本模块实现复杂查询的多步迭代检索:
1. 查询改写 (调用 QueryRewriter)
2. HyDE 假设文档生成
3. 单步检索 (调用 SingleStepStrategy)
4. 结果评估 (LLM 判断是否充分)
5. 如果不充分 → 改写查询 → 重新检索 (最多3轮)
6. 多轮结果去重合并
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional

from src.retrieval.base import RetrievalStrategy
from src.retrieval.single_step import SingleStepStrategy
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.query_rewriter import QueryRewriter
from src.types import (
    AgentState,
    Document,
    RetrievalStrategy as StrategyType,
    SearchResult,
)
from config.settings import get_settings
from src.models.llm import LLMClient

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3  # 最大迭代轮数


IMPLICIT_INFERENCE_PATTERNS = [
    r"(谁|哪位|哪个人).*(可能|适合|抽调|支援|帮忙|候选)",
    r"(如果|假如).*(需要|紧急|支援|加人)",
    r"(为什么|原因).*(可能|适合|推荐)",
]


def is_implicit_inference_query(query: str) -> bool:
    normalized = (query or "").strip()
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in IMPLICIT_INFERENCE_PATTERNS)


def is_table_or_numeric_query(query: str) -> bool:
    """Return whether HyDE is unlikely to help a table/list/numeric lookup."""
    normalized = (query or "").strip()
    return bool(
        re.search(
            r"(预算|支出|剩余|金额|总计|合计|总预算|总支出|表格|列表|列出|多少|几个|哪些)",
            normalized,
            re.IGNORECASE,
        )
    )


EVIDENCE_PATTERNS = {
    "personnel": [
        "姓名",
        "职位",
        "技能",
        "所属部门",
    ],
    "project_status": [
        "项目",
        "进度",
        "预计",
        "发布",
        "滞后",
    ],
    "timeline_resource": [
        "预算",
        "时间线",
        "Q1",
        "Q2",
        "支出",
        "剩余",
        "测试中",
    ],
}


def _document_content(doc: Document) -> str:
    content = getattr(doc, "content", None)
    if content is None:
        content = getattr(doc, "page_content", "")
    return str(content or "")


def _is_personnel_evidence_document(doc: Document) -> bool:
    content = _document_content(doc)
    metadata = getattr(doc, "metadata", None) or {}
    heading = " ".join(
        str(metadata.get(key, ""))
        for key in ("heading_path", "h1", "h2", "h3", "source")
    )
    return (
        ("姓名" in content and "所属部门" in content and "技能" in content)
        or ("| 姓名 |" in content and "| 职位 |" in content)
        or "团队成员" in heading
        or "团队成员与职责" in heading
    )


def _ensure_evidence_coverage(
    documents: list[Document],
    query: str,
) -> list[Document]:
    """Ensure implicit-inference retrieval keeps key evidence types near the top."""
    if not is_implicit_inference_query(query):
        return documents

    if len(documents) < 4:
        return documents

    reordered = list(documents)

    def _match_type(doc: Document, evidence_type: str) -> bool:
        if evidence_type == "personnel":
            return _is_personnel_evidence_document(doc)
        content_lower = _document_content(doc).lower()
        metadata = getattr(doc, "metadata", None) or {}
        source_lower = str(metadata.get("source", "")).lower()
        for pattern in EVIDENCE_PATTERNS[evidence_type]:
            pattern_lower = pattern.lower()
            if pattern_lower in content_lower or pattern_lower in source_lower:
                return True
        return False

    top_window = min(6, len(reordered))
    insert_pos = top_window
    promoted_ids: set[int] = set()

    for evidence_type in EVIDENCE_PATTERNS:
        covered = any(
            _match_type(reordered[i], evidence_type)
            for i in range(min(insert_pos, len(reordered)))
            if id(reordered[i]) not in promoted_ids
        )
        if covered:
            continue

        best_idx = -1
        for i in range(insert_pos, len(reordered)):
            if id(reordered[i]) in promoted_ids:
                continue
            if _match_type(reordered[i], evidence_type):
                best_idx = i
                break

        if best_idx < 0:
            continue

        doc = reordered[best_idx]
        promoted_ids.add(id(doc))
        original_position = best_idx
        del reordered[best_idx]
        reordered.insert(insert_pos, doc)
        logger.info(
            "证据覆盖保护: 类型=%s 文档(chunk=%s) 从位置%d提升到%d, score=%.3f",
            evidence_type,
            (getattr(doc, "metadata", None) or {}).get("chunk_index", "?"),
            original_position,
            insert_pos,
            float(getattr(doc, "score", 0.0) or 0.0),
        )
        insert_pos += 1

    if not promoted_ids:
        logger.info("证据覆盖保护: 所有证据类型已覆盖 (top-%d), 无需调整", top_window)

    return reordered


class MultiStepStrategy(RetrievalStrategy):
    """
    多步迭代检索策略
    ← Adaptive-RAG 论文: complex 查询 → 多步检索
    ← WeKnora: engine.go ReAct 循环 → 我们简化为检索策略层面的迭代

    面试可讲:
    "对于复杂查询，单次检索往往不够。比如用户问'分析营收增长的驱动因素'，
    需要分别检索各业务线的营收数据、市场环境分析、竞争格局等多个维度，
    然后综合回答。我实现了多步迭代检索: 每轮检索后 LLM 评估是否充分，
    不充分则自动改写查询进入下一轮，最多3轮，之后去重合并。"
    """

    def __init__(
        self,
        single_step_strategy: Optional[SingleStepStrategy] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(name="多步迭代检索 (Multi-Step)")
        self.strategy_type = StrategyType.MULTI_STEP

        self._single_step = single_step_strategy or SingleStepStrategy()
        self._llm = llm_client or LLMClient()
        self._rewriter = QueryRewriter(llm_client=self._llm)
        self._hyde = HyDEGenerator(llm_client=self._llm)

    # ----------------------------------------------------------------
    # 检索质量评估 (LLM 判断当前检索结果是否充分)
    # ----------------------------------------------------------------

    async def _evaluate_sufficiency(
        self,
        query: str,
        documents: list[Document],
    ) -> tuple[bool, str]:
        """
        LLM 评估检索结果是否足以回答用户问题

        Args:
            query: 用户查询
            documents: 当前检索到的文档

        Returns:
            (是否充分, 不足的原因/建议)
        """
        if not documents:
            return False, "没有检索到任何相关内容"

        # 组装检索内容摘要
        docs_summary = "\n---\n".join(
            f"[文档{i+1}] (得分:{doc.score:.2f}) {doc.content[:200]}"
            for i, doc in enumerate(documents[:5])
        )

        prompt = f"""请判断以下检索结果是否足够回答用户问题。

用户问题: {query}

检索到的文档内容:
{docs_summary}

请判断:
1. 这些检索结果是否包含回答问题所需的**关键信息**？
2. 如果不够充分，缺少哪些方面的信息？请给出下一步搜索的建议关键词。

输出 JSON 格式:
{{"sufficient": true/false, "reason": "简短理由", "suggestion": "下一步搜索建议(如果不充分)"}}
只输出 JSON。"""

        try:
            response = await self._llm.ask(prompt, model_name=get_settings().llm_simple_model)
            import json
            result = json.loads(response.strip())
            sufficient = result.get("sufficient", True)
            reason = result.get("reason", "")
            logger.debug("检索评估: sufficient=%s reason=%s", sufficient, reason)
            return sufficient, result.get("suggestion", reason)
        except Exception as e:
            logger.warning("检索质量评估失败: %s，默认判定为充分", e)
            return True, ""

    # ----------------------------------------------------------------
    # 主检索入口
    # ----------------------------------------------------------------

    async def retrieve(self, query: str, state: AgentState, **kwargs) -> SearchResult:
        """
        执行多步迭代检索

        流程:
        1. HyDE 生成假设文档
        2. 用假设文档做第一次检索
        3. LLM 评估检索质量
        4. 不充分 → 改写查询 → 重新检索 (最多3轮)
        5. 去重合并所有轮次的结果
        """
        start_time = time.perf_counter()
        all_documents: list[Document] = []
        seen_contents: set[str] = set()
        completed_iterations = 0

        def add_documents(documents: list[Document]) -> int:
            new_docs = 0
            for doc in documents:
                content_key = doc.content[:200]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    all_documents.append(doc)
                    new_docs += 1
            return new_docs

        async def retrieve_once(search_query: str, *, top_k: int | None = None) -> list[Document]:
            import copy

            iter_state = copy.copy(state)
            iter_state.query = search_query
            retrieve_kwargs = dict(kwargs)
            if top_k is not None:
                retrieve_kwargs["top_k"] = top_k
            result = await self._single_step.retrieve(search_query, iter_state, **retrieve_kwargs)
            return result.documents

        if is_implicit_inference_query(query):
            sub_queries = [
                f"{query} 团队成员及其技能",
                f"{query} 各项目当前进度和人力需求",
                f"{query} 技术栈匹配情况",
            ]

            async def retrieve_sub_query(sub_query: str) -> tuple[str, list[Document]]:
                try:
                    return sub_query, await retrieve_once(sub_query, top_k=3)
                except Exception as e:
                    logger.warning("隐含推断补充检索失败: %s", e)
                    return sub_query, []

            sub_results = await asyncio.gather(
                *(retrieve_sub_query(sub_query) for sub_query in sub_queries)
            )
            for sub_query, documents in sub_results:
                added = add_documents(documents)
                logger.info(
                    "隐含推断补充检索: query='%s...' added=%d",
                    sub_query[:50], added,
                )
            if not any(_is_personnel_evidence_document(doc) for doc in all_documents):
                personnel_queries = [
                    "团队成员 职位 所属部门 当前主要投入 技能特长",
                    "姓名 职位 所属部门 技能特长",
                ]
                for personnel_query in personnel_queries:
                    try:
                        personnel_docs = await retrieve_once(personnel_query, top_k=5)
                    except Exception as e:
                        logger.warning("隐含推断人员表兜底检索失败: %s", e)
                        continue
                    added = add_documents(personnel_docs)
                    logger.info(
                        "隐含推断人员表兜底检索: query='%s' added=%d",
                        personnel_query, added,
                    )
                    if any(_is_personnel_evidence_document(doc) for doc in all_documents):
                        break

        # Step 1: HyDE 生成假设文档。数值/表格类查询保留跳过 HyDE 的低风险优化。
        hyde_hypothesis = ""
        if not is_table_or_numeric_query(query):
            try:
                hyde_hypothesis = await self._hyde.generate(query)
                state.hyde_hypothesis = hyde_hypothesis
            except Exception as e:
                logger.warning("HyDE 生成失败: %s", e)

        # 使用 HyDE 假设文档作为检索查询 (如果生成成功)
        search_query = hyde_hypothesis or query

        # Step 2-4: 迭代检索循环，恢复每轮检索后都评估质量的原始语义。
        current_query = search_query
        for iteration in range(1, MAX_ITERATIONS + 1):
            logger.info(
                "多步检索: iteration=%d/%d query='%s...'",
                iteration, MAX_ITERATIONS, current_query[:50],
            )
            try:
                documents = await retrieve_once(current_query)
            except Exception as e:
                logger.warning("多步迭代检索失败: %s", e)
                documents = []
            completed_iterations = iteration
            new_docs = add_documents(documents)
            logger.info(
                "  轮次%d: 检索到%d个 → 新增%d个 (去重)",
                iteration, len(documents), new_docs,
            )

            sufficient, suggestion = await self._evaluate_sufficiency(
                query, all_documents,
            )

            if sufficient:
                logger.info("  检索评估: 充分 ✓，停止迭代")
                break

            logger.info("  检索评估: 不充分 ✗ → %s", suggestion[:60])
            if iteration < MAX_ITERATIONS:
                current_query = f"{query} {suggestion}"
            else:
                logger.info("  已达最大迭代次数，停止")

        # Step 5: 排序 — 按分数降序
        all_documents.sort(key=lambda d: d.score, reverse=True)

        # Step 5.5: 证据覆盖保护 — 隐含推断查询确保关键证据类型不被挤出 top 窗口
        all_documents = _ensure_evidence_coverage(all_documents, query)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "MultiStep 完成: %d iterations → %d unique docs (%.0fms)",
            completed_iterations if len(all_documents) > 0 else 0,
            len(all_documents), elapsed_ms,
        )

        return SearchResult(
            query=query,
            documents=all_documents,
            strategy=self.strategy_type,
            total_found=len(all_documents),
            search_time_ms=elapsed_ms,
        )
