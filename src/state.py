"""Shared state that flows through the agent graph.

This is the single source of truth that persists across nodes. Each node reads
from it and returns a partial update; LangGraph merges updates back in. The same
object is serialised to disk (see ``session.py``) so the QA stage can resume a
previous run without re-parsing the PDF.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ---- Input & query understanding ----
    raw_input: str                 # exactly what the user typed
    intent: str                    # "paper_lookup" | "topic_search"
    arxiv_query: str               # cleaned query used against the arXiv API
    target_arxiv_id: Optional[str] # set when the input was an ID/URL

    # ---- Retrieval & selection ----
    candidates: list[dict[str, Any]]   # paper metadata dicts from arXiv
    selected: dict[str, Any]           # the chosen paper's metadata
    alternatives: list[dict[str, Any]] # other strong candidates (topic search)
    selection_reason: str

    # ---- Fetch & parse ----
    pdf_path: Optional[str]
    parsed: dict[str, Any]         # {full_text, sections, abstract, references, num_pages, quality}

    # ---- Chunk & embed ----
    collection_name: Optional[str] # Chroma collection holding this paper's chunks
    num_chunks: int

    # ---- Summarise ----
    briefing: dict[str, Any]       # structured executive briefing
    briefing_path: Optional[str]

    # ---- QA (single turn; the loop lives in the CLI) ----
    question: str
    retrieved: list[dict[str, Any]]
    answer: str

    # ---- Cross-cutting ----
    conversation: list[dict[str, str]]  # [{role, content}] QA history
    warnings: list[str]
    error: Optional[str]           # set => pipeline halts gracefully


def new_state(raw_input: str) -> AgentState:
    """Create a fresh state for a new pipeline run."""
    return AgentState(
        raw_input=raw_input,
        warnings=[],
        conversation=[],
        error=None,
    )
