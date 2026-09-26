"""Node 1 — Query Understanding.

Decide whether the input is a specific paper (ID/URL) or a topic search, and
produce a clean arXiv query string. ID detection is deterministic (regex); topic
refinement uses the LLM but falls back to the raw text if the LLM is unavailable.
"""
from __future__ import annotations

from ..arxiv_client import extract_arxiv_id
from ..llm import LLMError, get_llm
from ..state import AgentState

_REFINE_SYSTEM = (
    "You convert a user's research interest into a concise arXiv search query. "
    "Return only the query string: 3-8 keywords, no punctuation, no explanation."
)


def query_understanding(state: AgentState) -> AgentState:
    raw = (state.get("raw_input") or "").strip()
    if not raw:
        return {"error": "Empty input. Provide an arXiv ID/URL or a research topic."}

    arxiv_id = extract_arxiv_id(raw)
    if arxiv_id:
        return {
            "intent": "paper_lookup",
            "target_arxiv_id": arxiv_id,
            "arxiv_query": arxiv_id,
        }

    # Topic search: try to refine into good keywords, but never hard-fail here.
    query = raw
    try:
        refined = get_llm().chat(_REFINE_SYSTEM, raw, temperature=0.0, max_tokens=60)
        refined = refined.strip().strip('"').splitlines()[0]
        if refined:
            query = refined
    except LLMError:
        pass  # keep raw query

    return {"intent": "topic_search", "target_arxiv_id": None, "arxiv_query": query}
