#!/usr/bin/env python3
"""arXiv Digest & QA Agent — command-line interface.

Usage:
    python main.py digest "<topic or arXiv ID/URL>"   # run the full pipeline + QA
    python main.py digest "<...>" --no-qa              # briefing only, skip QA loop
    python main.py qa <arxiv_id>                       # resume QA on a saved paper
    python main.py list                                # list papers with saved sessions

Examples:
    python main.py digest 2401.12345
    python main.py digest "recent work on KV-cache compression for LLMs"
    python main.py qa 2401.12345
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from src.briefing import render_markdown
from src.config import config
from src.graph import build_digest_graph, build_qa_graph
from src.llm import LLMError
from src.session import list_sessions, load_session, save_session
from src.state import AgentState, new_state

console = Console()


# --------------------------------------------------------------------------- #
# Digest command
# --------------------------------------------------------------------------- #
def cmd_digest(user_input: str, run_qa: bool) -> int:
    console.print(Panel.fit(f"[bold]arXiv Digest Agent[/bold]\nLLM: {config.describe_llm()}"))
    console.print(f"[cyan]Input:[/cyan] {user_input}\n")

    graph = build_digest_graph()
    with console.status("[bold green]Running pipeline (understand → retrieve → parse → embed → summarise)…"):
        final: AgentState = graph.invoke(new_state(user_input))

    if final.get("error"):
        console.print(Panel(final["error"], title="[red]Could not complete", border_style="red"))
        return 1

    _print_selection(final)
    _print_briefing(final)

    session_path = save_session(final)
    if session_path:
        console.print(f"\n[dim]Briefing saved to {final.get('briefing_path')}[/dim]")
        console.print(f"[dim]Session saved to {session_path}[/dim]")

    for w in final.get("warnings", []):
        console.print(f"[yellow]⚠ {w}[/yellow]")

    if run_qa and final.get("collection_name"):
        _qa_loop(final)
    elif run_qa:
        console.print("\n[yellow]QA unavailable: no text could be embedded for this paper.[/yellow]")
    return 0


# --------------------------------------------------------------------------- #
# QA command (resume a saved session)
# --------------------------------------------------------------------------- #
def cmd_qa(arxiv_id: str) -> int:
    state = load_session(arxiv_id)
    if state is None:
        console.print(f"[red]No saved session for '{arxiv_id}'.[/red] Run `digest` on it first.")
        ids = list_sessions()
        if ids:
            console.print("Saved sessions: " + ", ".join(ids))
        return 1
    console.print(Panel.fit(f"[bold]QA — {state['selected']['title']}[/bold]\n{arxiv_id}"))
    if not state.get("collection_name"):
        console.print("[red]This paper has no embedded text; QA is unavailable.[/red]")
        return 1
    _qa_loop(state)
    return 0


def cmd_list() -> int:
    ids = list_sessions()
    if not ids:
        console.print("No saved sessions yet. Run `digest` first.")
        return 0
    console.print("[bold]Saved sessions:[/bold]")
    for i in ids:
        s = load_session(i)
        title = (s or {}).get("selected", {}).get("title", "?")
        console.print(f"  • {i} — {title}")
    return 0


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _print_selection(state: AgentState) -> None:
    sel = state["selected"]
    console.print(Rule("Selected paper"))
    console.print(f"[bold]{sel['title']}[/bold]  ([green]{sel['arxiv_id']}[/green])")
    console.print(f"[dim]{', '.join(sel['authors'][:6])}"
                  f"{' et al.' if len(sel['authors']) > 6 else ''} · {sel.get('published')}[/dim]")
    if state.get("selection_reason"):
        console.print(f"[dim]Why: {state['selection_reason']}[/dim]")
    alts = state.get("alternatives") or []
    if alts:
        console.print("\n[dim]Other candidates:[/dim]")
        for a in alts:
            console.print(f"  [dim]- {a['title']} ({a['arxiv_id']})[/dim]")
    console.print()


def _print_briefing(state: AgentState) -> None:
    console.print(Rule("Executive Briefing"))
    md = render_markdown(state["briefing"])
    console.print(Markdown(md))


def _qa_loop(state: AgentState) -> None:
    console.print(Rule("QA mode"))
    suggestions = state.get("briefing", {}).get("follow_up_questions", [])
    if suggestions:
        console.print("[dim]Try one of the suggested questions, or ask your own:[/dim]")
        for q in suggestions:
            console.print(f"  [dim]• {q}[/dim]")
    console.print("[dim]Type your question. Commands: 'exit' to quit.[/dim]\n")

    qa_graph = build_qa_graph()
    while True:
        try:
            question = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Leaving QA.[/dim]")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break

        state["question"] = question
        try:
            result = qa_graph.invoke(state)
        except LLMError as exc:
            console.print(f"[red]{exc}[/red]")
            continue

        # Merge QA outputs back into the persistent state.
        state["conversation"] = result.get("conversation", state.get("conversation", []))
        answer = result.get("answer", "")
        sources = result.get("retrieved", [])

        console.print(f"\n[bold green]agent ›[/bold green] {answer}")
        strong = [s for s in sources if s.get("similarity", 0) >= 0.15]
        if strong:
            secs = ", ".join(sorted({s["section"] for s in strong}))
            console.print(f"[dim]  grounded in {len(strong)} chunk(s) · sections: {secs}[/dim]\n")
        else:
            console.print()

        save_session(state)  # persist conversation after each turn


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous arXiv Paper Digest & QA Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("digest", help="Fetch, summarise, and QA a paper or topic")
    d.add_argument("input", help="arXiv ID/URL or a natural-language topic")
    d.add_argument("--no-qa", action="store_true", help="Skip the interactive QA loop")

    q = sub.add_parser("qa", help="Resume QA on a previously digested paper")
    q.add_argument("arxiv_id", help="arXiv ID of a saved session")

    sub.add_parser("list", help="List saved sessions")

    args = parser.parse_args()

    try:
        if args.command == "digest":
            return cmd_digest(args.input, run_qa=not args.no_qa)
        if args.command == "qa":
            return cmd_qa(args.arxiv_id)
        if args.command == "list":
            return cmd_list()
    except LLMError as exc:
        console.print(Panel(str(exc), title="[red]LLM configuration error", border_style="red"))
        return 2
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
