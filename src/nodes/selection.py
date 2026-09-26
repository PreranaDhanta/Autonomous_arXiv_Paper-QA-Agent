"""Node 3 — Selection / Ranking.

For a paper lookup there is one candidate. For a topic search we ask the LLM to
pick the single most relevant paper given the user's query and the candidate
titles+abstracts, and we keep the runners-up as `alternatives`. If the LLM is
unavailable we fall back to arXiv's own relevance ordering (candidate 0).
"""
from __future__ import annotations

from ..llm import LLMError, get_llm
from ..state import AgentState

_SELECT_SYSTEM = (
    "You are helping choose the single most relevant arXiv paper for a user's "
    "research interest. Return JSON: {\"choice\": <int index>, \"reason\": <short string>}."
)


def selection(state: AgentState) -> AgentState:
    candidates = state.get("candidates", [])
    if not candidates:
        return {"error": "No candidates to select from."}

    if state.get("intent") == "paper_lookup" or len(candidates) == 1:
        return {
            "selected": candidates[0],
            "alternatives": [],
            "selection_reason": "Directly requested / only candidate.",
        }

    # Build a compact menu for the LLM.
    menu = "\n".join(
        f"[{i}] {c['title']} ({c['arxiv_id']})\n    {c['abstract'][:280]}"
        for i, c in enumerate(candidates)
    )
    user = f"User interest: {state.get('raw_input')}\n\nCandidates:\n{menu}"

    choice = 0
    reason = "Top arXiv relevance result (LLM ranking unavailable)."
    try:
        out = get_llm().chat_json(_SELECT_SYSTEM, user, max_tokens=200)
        idx = int(out.get("choice", 0))
        if 0 <= idx < len(candidates):
            choice = idx
            reason = str(out.get("reason", "")).strip() or reason
    except (LLMError, ValueError, TypeError):
        pass

    selected = candidates[choice]
    alternatives = [c for i, c in enumerate(candidates) if i != choice][:4]
    return {"selected": selected, "alternatives": alternatives, "selection_reason": reason}
