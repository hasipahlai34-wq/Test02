"""
# ============================================================
# ★ HITL 人工审核 CLI 工具
#
# 用法:
#   python cli/review_pending.py list            → 列出所有待审核项
#   python cli/review_pending.py show <id>       → 显示审核项详情
#   python cli/review_pending.py approve <id>    → 批准回答
#   python cli/review_pending.py reject <id>     → 拒绝回答
#   python cli/review_pending.py edit <id>       → 交互式编辑后批准
#   python cli/review_pending.py stats           → 审核统计
#
# 设计要点:
# - 统一处理 interrupt 和 file_queue 两种模式产生的队列项
# - 审核结果写入 results 目录归档
# - 统计信息: 通过率、平均处理时间、触发原因分布
# ============================================================
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# 修复 Windows GBK 编码问题：强制 stdout/stderr 使用 UTF-8
# 仅在 stdout/stderr 支持 reconfigure 时执行（TTY 或普通文件），管道等场景静默跳过
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings


def _get_queue_dir() -> Path:
    settings = get_settings()
    return Path(settings.hitl_queue_dir)


def _get_results_dir() -> Path:
    settings = get_settings()
    return Path(settings.hitl_results_dir)


# ================================================================
# 命令实现
# ================================================================


def cmd_list():
    """列出所有待审核项"""
    queue_dir = _get_queue_dir()
    if not queue_dir.exists():
        print("📭 待审核队列为空 (队列目录不存在)")
        return

    items = []
    for filepath in sorted(queue_dir.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                item = json.load(f)
            if item.get("hitl_status") in ("pending", "pending_timeout"):
                items.append(item)
        except Exception as e:
            print(f"  ⚠️ 读取失败: {filepath.name} - {e}")

    if not items:
        print("📭 待审核队列为空")
        return

    print(f"\n{'=' * 80}")
    print(f"📋 待审核队列 — {len(items)} 项")
    print(f"{'=' * 80}\n")

    for item in items:
        rid = item.get("review_id", "?")
        query = item.get("query", "")[:60]
        status = item.get("hitl_status", "?")
        mode = item.get("mode", "?")
        reasons = ", ".join(item.get("trigger_reasons", []))
        created = item.get("created_at", "")[:19]

        status_icon = "⏰" if status == "pending_timeout" else "⏳"
        mode_icon = "🔴" if mode == "interrupt" else "📄"

        print(f"  {status_icon} {mode_icon} [{rid}] {query}")
        print(f"     触发: {reasons}  |  创建: {created}")
        print()


def cmd_show(review_id: str):
    """显示审核项详情"""
    queue_dir = _get_queue_dir()
    filepath = queue_dir / f"{review_id}.json"

    if not filepath.exists():
        # 尝试 results 目录
        results_dir = _get_results_dir()
        filepath = results_dir / f"{review_id}.json"
        if not filepath.exists():
            print(f"❌ 审核项不存在: {review_id}")
            return

    with open(filepath, "r", encoding="utf-8") as f:
        item = json.load(f)

    print(f"\n{'=' * 80}")
    print(f"📋 审核项详情: {review_id}")
    print(f"{'=' * 80}\n")

    print(f"  状态:       {item.get('hitl_status', '?')}")
    print(f"  模式:       {item.get('mode', '?')}")
    print(f"  会话:       {item.get('session_id', '?')}")
    print(f"  复杂度:     {item.get('complexity', '?')}")
    print(f"  策略:       {item.get('selected_strategy', '?')}")
    print(f"  创建时间:   {item.get('created_at', '?')}")
    print(f"  超时时间:   {item.get('timeout_at', '?')}")
    print()

    print(f"  ❓ 用户问题:")
    print(f"     {item.get('query', '(无)')}")
    print()

    print(f"  💬 AI 回答:")
    answer = item.get("answer", "(无)")
    for line in answer.split("\n")[:20]:
        print(f"     {line}")
    if len(answer.split("\n")) > 20:
        print(f"     ... (共 {len(answer.split(chr(10)))} 行)")
    print()

    print(f"  📊 质量评分: {item.get('quality_score', 0):.2f} "
          f"({'通过' if item.get('quality_passed') else '未通过'})")

    ragas = item.get("ragas_scores")
    if ragas and isinstance(ragas, dict):
        print(f"  📈 RAGAS 评分:")
        for k, v in ragas.items():
            print(f"     {k}: {v:.3f}")
    else:
        print(f"  📈 RAGAS 评分: (无)")

    print(f"  🛡️ 安全风险: {item.get('safety_risk_level', '?')}")
    print(f"  🚨 触发原因: {', '.join(item.get('trigger_reasons', []))}")
    review_reason = item.get("review_reason")
    if review_reason:
        print(f"  📝 审核说明: {review_reason}")

    docs = item.get("retrieved_docs_preview", [])
    if docs:
        print(f"\n  📚 检索文档 ({len(docs)} 个):")
        for i, doc in enumerate(docs, 1):
            score = doc.get("score", "?")
            content = doc.get("content", "")[:100]
            print(f"     [{i}] (分数:{score}) {content}...")
    print()


def cmd_approve(review_id: str):
    """批准审核项"""
    from src.graph.hitl import update_queue_item, archive_queue_item

    result = update_queue_item(review_id, {
        "hitl_status": "approved",
        "hitl_decision": "approve",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    if result:
        archive_queue_item(review_id)
        print(f"✅ 已批准: {review_id}")
    else:
        print(f"❌ 审核项不存在或已处理: {review_id}")


def cmd_reject(review_id: str):
    """拒绝审核项"""
    from src.graph.hitl import update_queue_item, archive_queue_item

    result = update_queue_item(review_id, {
        "hitl_status": "rejected",
        "hitl_decision": "reject",
        "hitl_edited_answer": "回答未通过质量审核，请重新提问",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    if result:
        archive_queue_item(review_id)
        print(f"❌ 已拒绝: {review_id}")
        print(f"   用户将看到: '回答未通过质量审核，请重新提问'")
    else:
        print(f"❌ 审核项不存在或已处理: {review_id}")


def cmd_edit(review_id: str):
    """交互式编辑后批准"""
    queue_dir = _get_queue_dir()
    filepath = queue_dir / f"{review_id}.json"

    if not filepath.exists():
        print(f"❌ 审核项不存在: {review_id}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        item = json.load(f)

    answer = item.get("answer", "")
    print(f"\n原回答 ({len(answer)} 字符):")
    print("-" * 60)
    print(answer)
    print("-" * 60)

    print("\n请输入修改后的回答 (输入 END 结束, Ctrl+C 取消):")
    lines = []
    try:
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
    except KeyboardInterrupt:
        print("\n⚠️ 已取消")
        return

    edited_answer = "\n".join(lines)

    if not edited_answer.strip():
        print("⚠️ 编辑内容为空, 已取消")
        return

    from src.graph.hitl import update_queue_item, archive_queue_item

    result = update_queue_item(review_id, {
        "hitl_status": "edited",
        "hitl_decision": "edit",
        "hitl_edited_answer": edited_answer,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    if result:
        archive_queue_item(review_id)
        print(f"✅ 已编辑并批准: {review_id}")
        print(f"   修改后回答 ({len(edited_answer)} 字符):")
        print(f"   {edited_answer[:200]}...")
    else:
        print(f"❌ 审核项不存在或已处理: {review_id}")


def cmd_stats():
    """审核统计信息"""
    results_dir = _get_results_dir()
    queue_dir = _get_queue_dir()

    # 统计已处理项
    processed = []
    if results_dir.exists():
        for filepath in results_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    item = json.load(f)
                processed.append(item)
            except Exception:
                pass

    # 统计待处理项
    pending = []
    if queue_dir.exists():
        for filepath in queue_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    item = json.load(f)
                if item.get("hitl_status") in ("pending", "pending_timeout"):
                    pending.append(item)
            except Exception:
                pass

    print(f"\n{'=' * 60}")
    print(f"📊 HITL 审核统计")
    print(f"{'=' * 60}\n")

    print(f"  待审核:     {len(pending)} 项")
    print(f"  已处理:     {len(processed)} 项")

    if processed:
        approved = sum(1 for p in processed if p.get("hitl_decision") == "approve")
        rejected = sum(1 for p in processed if p.get("hitl_decision") == "reject")
        edited = sum(1 for p in processed if p.get("hitl_decision") == "edit")
        timeout = sum(1 for p in processed if p.get("hitl_decision") == "pending_timeout")

        total_decided = approved + rejected + edited
        if total_decided > 0:
            approval_rate = (approved + edited) / total_decided * 100
            print(f"\n  批准:       {approved} 项")
            print(f"  编辑通过:   {edited} 项")
            print(f"  拒绝:       {rejected} 项")
            print(f"  超时:       {timeout} 项")
            print(f"  通过率:     {approval_rate:.1f}%")

        # 平均处理时间
        durations = []
        for p in processed:
            created = p.get("created_at", "")
            updated = p.get("updated_at", "")
            if created and updated:
                try:
                    t0 = datetime.fromisoformat(created)
                    t1 = datetime.fromisoformat(updated)
                    durations.append((t1 - t0).total_seconds())
                except (ValueError, TypeError):
                    pass

        if durations:
            avg_seconds = sum(durations) / len(durations)
            minutes = avg_seconds / 60
            print(f"  平均处理时间: {minutes:.1f} 分钟")

    # 触发原因分布
    all_reasons: dict[str, int] = {}
    for p in processed + pending:
        for reason in p.get("trigger_reasons", []):
            # 简化原因标签
            short = reason.split("_")[0] if "_" in reason else reason
            all_reasons[short] = all_reasons.get(short, 0) + 1

    if all_reasons:
        print(f"\n  触发原因分布:")
        for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count} 次")

    print()


# ================================================================
# 入口
# ================================================================


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "list":
        cmd_list()
    elif cmd == "show":
        if len(sys.argv) < 3:
            print("用法: python cli/review_pending.py show <review_id>")
            return
        cmd_show(sys.argv[2])
    elif cmd == "approve":
        if len(sys.argv) < 3:
            print("用法: python cli/review_pending.py approve <review_id>")
            return
        cmd_approve(sys.argv[2])
    elif cmd == "reject":
        if len(sys.argv) < 3:
            print("用法: python cli/review_pending.py reject <review_id>")
            return
        cmd_reject(sys.argv[2])
    elif cmd == "edit":
        if len(sys.argv) < 3:
            print("用法: python cli/review_pending.py edit <review_id>")
            return
        cmd_edit(sys.argv[2])
    elif cmd == "stats":
        cmd_stats()
    else:
        print(f"未知命令: {cmd}")
        print("可用命令: list, show, approve, reject, edit, stats")


if __name__ == "__main__":
    main()
