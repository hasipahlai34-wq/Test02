"""Command line entrypoints for Adaptive RAG."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))


HELP = """Usage:
  python main.py ui
  python main.py ask <query>
  python main.py eval <query>
  python main.py serve
  python main.py chat
  python main.py ingest <path>
"""


def cmd_ui() -> None:
    """Start the Streamlit UI."""
    print("Starting Adaptive RAG Streamlit UI: http://localhost:8501")
    from streamlit.web import cli as stcli

    os.environ["STREAMLIT_SUPPRESS_ONBOARDING_EMAIL"] = "1"
    ui_path = Path(__file__).resolve().parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(ui_path), "--server.headless", "true"]
    stcli.main()


def cmd_ask(query: str) -> None:
    """Run one Adaptive RAG query from the CLI."""

    async def _ask() -> None:
        from src.graph.state import GraphState
        from src.graph.workflow import build_adaptive_rag_graph
        from src.utils.observability import langfuse_trace_context, with_langfuse_config

        print(f"Question: {query}")
        print("=" * 60)

        app = build_adaptive_rag_graph()
        config = with_langfuse_config(
            {"configurable": {"thread_id": "cli"}},
            trace_name="adaptive-rag.cli.ask",
            session_id="cli",
            metadata={"entrypoint": "cli", "query": query},
            tags=["cli"],
        )
        with langfuse_trace_context(
            trace_name="adaptive-rag.cli.ask",
            session_id="cli",
            metadata={"entrypoint": "cli", "query": query},
            tags=["cli"],
        ):
            result = await app.ainvoke(
                GraphState(query=query, session_id="cli"),
                config,
            )

        print(f"\nComplexity: {result.get('complexity', 'N/A')}")
        print(f"Strategy: {result.get('selected_strategy', 'N/A')}")
        print(f"Search count: {result.get('search_count', 0)}")
        print(f"Quality: {result.get('quality_score', 0):.2f}")
        print(f"\nAnswer:\n{result.get('generated_answer', '(no answer)')}")

    asyncio.run(_ask())


def cmd_eval(query: str) -> None:
    """Run the three-way comparison evaluation."""

    async def _eval() -> None:
        from rich.console import Console
        from rich.table import Table
        from src.evaluation.compare import run_comparison

        console = Console()
        console.print(f"[bold]Comparison[/bold]: {query}")
        console.print("=" * 60)

        result = await run_comparison(query)
        direct = result.direct_answer
        rag = result.standard_rag
        adaptive = result.adaptive_rag

        table = Table(title="Comparison Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Direct", style="yellow")
        table.add_column("Standard RAG", style="yellow")
        table.add_column("Adaptive RAG", style="green")
        table.add_row("Time", f"{direct['time_ms']:.0f}ms", f"{rag['time_ms']:.0f}ms", f"{adaptive['time_ms']:.0f}ms")
        table.add_row("Tokens", str(direct['tokens_est']), str(rag['tokens_est']), str(adaptive['tokens_est']))

        rag_scores = rag.get("scores", {}) or {}
        adaptive_scores = adaptive.get("scores", {}) or {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            table.add_row(
                metric,
                "N/A",
                f"{rag_scores.get(metric, 0):.3f}",
                f"{adaptive_scores.get(metric, 0):.3f}",
            )

        console.print(table)
        console.print(f"\n[bold]Conclusion[/bold]\n{result.conclusion}")

    asyncio.run(_eval())


def cmd_serve() -> None:
    """Start the FastAPI service."""
    import uvicorn
    from api.routes import app

    print("Starting FastAPI service: http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def cmd_chat() -> None:
    """Run interactive CLI chat."""

    async def _chat() -> None:
        from rich.console import Console
        from rich.markdown import Markdown
        from src.graph.state import GraphState
        from src.graph.workflow import build_adaptive_rag_graph
        from src.utils.observability import langfuse_trace_context, with_langfuse_config

        console = Console()
        console.print("[bold green]Adaptive RAG interactive mode[/bold green]")
        console.print("Type a question, or type [bold]quit[/bold] to exit.\n")

        app = build_adaptive_rag_graph()
        session_id = "cli_chat_session"

        while True:
            try:
                query = console.input("[cyan]You> [/cyan]")
                if query.lower() in ("quit", "exit", "q"):
                    console.print("Bye.")
                    break
                if not query.strip():
                    continue

                with console.status("[bold green]Thinking...[/bold green]"):
                    config = with_langfuse_config(
                        {"configurable": {"thread_id": session_id}},
                        trace_name="adaptive-rag.cli.chat",
                        session_id=session_id,
                        metadata={"entrypoint": "cli.chat", "query": query},
                        tags=["cli", "chat"],
                    )
                    with langfuse_trace_context(
                        trace_name="adaptive-rag.cli.chat",
                        session_id=session_id,
                        metadata={"entrypoint": "cli.chat", "query": query},
                        tags=["cli", "chat"],
                    ):
                        result = await app.ainvoke(
                            GraphState(query=query, session_id=session_id),
                            config,
                        )

                console.print(
                    f"\n[dim]Complexity: {result.get('complexity', 'N/A')} | "
                    f"Strategy: {result.get('selected_strategy', 'N/A')} | "
                    f"Docs: {result.get('search_count', 0)} | "
                    f"Quality: {result.get('quality_score', 0):.2f}[/dim]"
                )
                console.print(Markdown(result.get("generated_answer", "(no answer)")))
                console.print()
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    asyncio.run(_chat())


def cmd_ingest(filepath: str) -> None:
    """Ingest a document into the local vector index."""

    async def _ingest() -> None:
        from src.ingestion.chunker import ChunkingStrategy, auto_chunk, chunk_documents
        from src.ingestion.indexer import DocumentIndexer
        from src.ingestion.loader import load_document

        raw_docs = await load_document(filepath)
        try:
            chunks = auto_chunk(filepath)
        except Exception:
            chunks = chunk_documents(raw_docs, strategy=ChunkingStrategy.RECURSIVE)

        indexer = DocumentIndexer()
        count = await indexer.index_documents(chunks)
        print(f"Loaded segments: {len(raw_docs)}")
        print(f"Chunks: {len(chunks)}")
        print(f"Indexed chunks: {count}")

    asyncio.run(_ingest())


def main() -> None:
    """Dispatch CLI commands."""
    if len(sys.argv) < 2:
        print(HELP)
        return

    cmd = sys.argv[1].lower()
    if cmd == "ui":
        cmd_ui()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('Usage: python main.py ask "your question"')
            sys.exit(1)
        cmd_ask(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        if len(sys.argv) < 3:
            print('Usage: python main.py eval "your question"')
            sys.exit(1)
        cmd_eval(" ".join(sys.argv[2:]))
    elif cmd == "serve":
        cmd_serve()
    elif cmd == "chat":
        cmd_chat()
    elif cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <path>")
            sys.exit(1)
        cmd_ingest(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        print(HELP)
        sys.exit(1)


if __name__ == "__main__":
    main()