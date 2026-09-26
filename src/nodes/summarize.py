"""Node 6 — Summarise.

Produces the structured executive briefing (JSON) and writes a Markdown copy to
disk. The prompt forces explicit limitations and grounds content in the extracted
paper text.
"""
from __future__ import annotations

from ..briefing import (
    SUMMARY_SYSTEM,
    build_summary_prompt,
    normalise_briefing,
    render_markdown,
)
from ..config import config
from ..llm import LLMError, get_llm
from ..state import AgentState


def summarize(state: AgentState) -> AgentState:
    meta = state["selected"]
    parsed = state.get("parsed") or {}
    warnings = list(state.get("warnings", []))

    prompt = build_summary_prompt(meta, parsed)
    try:
        raw = get_llm().chat_json(SUMMARY_SYSTEM, prompt, max_tokens=2200)
    except LLMError as exc:
        return {"error": f"Summarisation failed: {exc}"}

    briefing = normalise_briefing(raw, meta)

    # Persist a Markdown copy alongside the session.
    md = render_markdown(briefing, warnings)
    path = config.briefing_dir / f"{meta['arxiv_id'].replace('/', '_')}.md"
    path.write_text(md)

    return {"briefing": briefing, "briefing_path": str(path), "warnings": warnings}
