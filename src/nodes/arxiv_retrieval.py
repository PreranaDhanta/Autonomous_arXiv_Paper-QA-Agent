"""Node 2 — arXiv Retrieval.

Calls the official arXiv API. Handles the two realistic failure modes explicitly:
  * paper lookup returns nothing  -> graceful error
  * topic search returns zero      -> graceful error with a hint to broaden
"""
from __future__ import annotations

from ..arxiv_client import fetch_by_id, search_topic
from ..config import config
from ..state import AgentState


def arxiv_retrieval(state: AgentState) -> AgentState:
    intent = state.get("intent")
    try:
        if intent == "paper_lookup":
            paper = fetch_by_id(state["target_arxiv_id"])
            if not paper:
                return {
                    "error": f"No arXiv paper found for ID '{state['target_arxiv_id']}'. "
                    "Double-check the ID (e.g. 2401.12345)."
                }
            return {"candidates": [paper]}

        # Topic search
        results = search_topic(state["arxiv_query"], max_results=config.topic_max_results)
        if not results:
            return {
                "error": f"arXiv returned no papers for '{state['arxiv_query']}'. "
                "Try broader or different keywords."
            }
        return {"candidates": results}

    except Exception as exc:  # network / API errors
        return {"error": f"arXiv retrieval failed: {exc}"}
