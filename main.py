"""
# ============================================================
# ★ 唯一入口: python main.py [ui|ask|eval|serve|chat]
# ← WeKnora: cmd/server/main.go — Gin 服务启动入口
# ============================================================

用法:
  python main.py ui                    → 启动 Streamlit UI (演示视频主力)
  python main.py ask "你的问题"         → 单次命令行问答
  python main.py eval "你的问题"        → 三路对比评估
  python main.py serve                 → 启动 FastAPI 服务
  python main.py chat                  → 终端交互对话模式
  python main.py ingest <filepath>     → 摄入文档到知识库
"""

from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path

# 修复 Windows GBK 编码问题：强制 stdout/stderr 使用 UTF-8
# 仅在 stdout/stderr 支持 reconfigure 时执行（TTY 或普通文件），管道等场景静默跳过
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass  # 管道/重定向场景不支持 reconfigure，保持默认编码
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_ui():
    """启动 Streamlit UI"""
    print("🧠 启动 Adaptive RAG Streamlit UI...")
    print("   浏览器将自动打开: http://localhost:8501")
    from streamlit.web import cli as stcli

    # 跳过 Streamlit 首次启动的邮箱收集提示
    os.environ["STREAMLIT_SUPPRESS_ONBOARDING_EMAIL"] = "1"

    ui_path = Path(__file__).resolve().parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(ui_path), "--server.headless", "true"]
    stcli.main()


def cmd_ask(query: str):
    """单次命令行问答"""
    async def _ask():
        from src.graph.workflow import build_adaptive_rag_graph
        from src.graph.state import GraphState

        print(f"🤔 问题: {query}")
        print("=" * 60)

        app = build_adaptive_rag_graph()
        result = await app.ainvoke(GraphState(query=query, session_id="cli"))

        print(f"\n📊 复杂度: {result.get('complexity', 'N/A')}")
        print(f"🔀 策略: {result.get('selected_strategy', 'N/A')}")
        print(f"📚 检索: {result.get('search_count', 0)} 个文档")
        print(f"✅ 质量: {result.get('quality_score', 0):.2f}")
        print(f"\n💬 回答:\n{result.get('generated_answer', '(无回答)')}")

    asyncio.run(_ask())


def cmd_eval(query: str):
    """三路对比评估"""
    async def _eval():
        from src.evaluation.compare import run_comparison
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"[bold]📊 三路对比评估[/bold]: {query}")
        console.print("=" * 60)

        result = await run_comparison(query)

        # 表格展示
        table = Table(title="对比结果")
        table.add_column("指标", style="cyan")
        table.add_column("直接回答", style="yellow")
        table.add_column("标准RAG", style="yellow")
        table.add_column("自适应RAG ⭐", style="green")

        direct = result.direct_answer
        rag = result.standard_rag
        adaptive = result.adaptive_rag

        table.add_row("耗时", f"{direct['time_ms']:.0f}ms", f"{rag['time_ms']:.0f}ms", f"{adaptive['time_ms']:.0f}ms")
        table.add_row("Token估算", str(direct['tokens_est']), str(rag['tokens_est']), str(adaptive['tokens_est']))

        rag_scores = rag.get("scores", {})
        adaptive_scores = adaptive.get("scores", {})
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            table.add_row(
                metric,
                "N/A",
                f"{rag_scores.get(metric, 0):.3f}",
                f"{adaptive_scores.get(metric, 0):.3f}"
            )

        console.print(table)
        console.print(f"\n[bold]结论:[/bold]\n{result.conclusion}")

    asyncio.run(_eval())


def cmd_serve():
    """启动 FastAPI 服务"""
    import uvicorn
    from api.routes import app

    print("🚀 启动 FastAPI 服务: http://localhost:8000")
    print("   API 文档: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def cmd_chat():
    """终端交互对话模式"""
    async def _chat():
        from src.graph.workflow import build_adaptive_rag_graph
        from src.graph.state import GraphState
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        console.print("[bold green]🧠 Adaptive RAG 交互模式[/bold green]")
        console.print("输入问题开始对话，输入 [bold]quit[/bold] 退出\n")

        app = build_adaptive_rag_graph()
        session_id = "cli_chat_session"

        while True:
            try:
                query = console.input("[cyan]🙋 你: [/cyan]")
                if query.lower() in ("quit", "exit", "q"):
                    console.print("[bold]再见！👋[/bold]")
                    break
                if not query.strip():
                    continue

                with console.status("[bold green]🤔 思考中...[/bold green]"):
                    result = await app.ainvoke(
                        GraphState(query=query, session_id=session_id),
                        {"configurable": {"thread_id": session_id}},
                    )

                console.print(f"\n[dim]复杂度: {result.get('complexity', 'N/A')} | "
                             f"策略: {result.get('selected_strategy', 'N/A')} | "
                             f"检索: {result.get('search_count', 0)} docs | "
                             f"质量: {result.get('quality_score', 0):.2f}[/dim]")
                console.print(Markdown(result.get("generated_answer", "(无回答)")))
                console.print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]❌ 错误: {e}[/red]")

    asyncio.run(_chat())


def cmd_ingest(filepath: str):
    """摄入文档"""
    async def _ingest():
        from src.ingestion.loader import load_document
        from src.ingestion.chunker import auto_chunk
        from src.ingestion.indexer import DocumentIndexer

        print(f"📄 摄入文档: {filepath}")
        docs = await load_document(filepath)
        print(f"   加载: {len(docs)} 个原始段")

        chunks = auto_chunk(str(Path(filepath).resolve()))
        print(f"   分块: {len(chunks)} 个 chunks")

        indexer = DocumentIndexer()
        count = await indexer.index_documents(chunks)
        print(f"   索引: {count} 个 chunks 已入库 ✅")

    asyncio.run(_ingest())


def main():
    """主入口 — 解析命令行参数分发到各子命令"""
    # ★ 全局日志初始化：彩色终端 + 第三方库静音
    from src.utils.logger import setup_logging
    setup_logging()

    # ★ 启动时 API Key 检查 —— 友好报错
    from config.settings import get_settings
    settings = get_settings()
    if settings.llm_api_key in ("", "sk-placeholder", "sk-your-api-key-here"):
        print(
            "\n[ERROR] LLM API Key is not configured.\n"
            "Please follow these steps:\n"
            "  1. cp .env.example .env\n"
            "  2. Edit .env and set LLM_API_KEY=your-key\n"
            "  3. For Ollama: LLM_BASE_URL=http://localhost:11434/v1\n"
            "     (see .env.example for more examples)\n"
            "  4. Re-run: python main.py <command>\n"
        )
        sys.exit(1)
    if len(sys.argv) < 2:
        print(__doc__)
        print("可用命令:")
        print("  python main.py ui              → 启动 Streamlit UI")
        print("  python main.py ask <query>     → 单次问答")
        print("  python main.py eval <query>    → 三路对比评估")
        print("  python main.py serve           → FastAPI 服务")
        print("  python main.py chat            → 终端交互")
        print("  python main.py ingest <path>   → 摄入文档")
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "ui":
        cmd_ui()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("用法: python main.py ask \"你的问题\"")
            sys.exit(1)
        cmd_ask(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        if len(sys.argv) < 3:
            print("用法: python main.py eval \"你的问题\"")
            sys.exit(1)
        cmd_eval(" ".join(sys.argv[2:]))
    elif cmd == "serve":
        cmd_serve()
    elif cmd == "chat":
        cmd_chat()
    elif cmd == "ingest":
        if len(sys.argv) < 3:
            print("用法: python main.py ingest <文件路径>")
            sys.exit(1)
        cmd_ingest(sys.argv[2])
    else:
        print(f"未知命令: {cmd}")
        print("可用命令: ui, ask, eval, serve, chat, ingest")
        sys.exit(1)


if __name__ == "__main__":
    main()
