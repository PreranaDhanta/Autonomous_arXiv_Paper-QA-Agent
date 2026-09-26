"""Explicit agentic state graph (nodes + edges + shared state) via LangGraph.

Two graphs are built:

  DIGEST graph (topic/ID -> executive briefing):

      query_understanding
              |
        arxiv_retrieval --(error / zero results)--> END
              |
          selection
              |
         fetch_parse ----(fatal: nothing to work with)--> END
              |
         chunk_embed
              |
          summarize --> END

  QA graph (one grounded question -> answer):

      retrieve_chunks --> generate_answer --> END

The shared ``AgentState`` object is threaded through every node. Conditional
edges implement graceful failure: any node that sets ``state['error']`` routes
the run straight to END with a helpful message instead of crashing downstream.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    arxiv_retrieval,
    chunk_embed,
    fetch_parse,
    generate_answer,
    query_understanding,
    retrieve_chunks,
    selection,
    summarize,
)
from .state import AgentState


def _has_error(state: AgentState) -> str:
    return "error" if state.get("error") else "ok"


def _parse_ok(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    parsed = state.get("parsed") or {}
    # Nothing at all to embed or summarise -> stop gracefully.
    if not parsed.get("full_text") and not parsed.get("abstract"):
        return "error"
    return "ok"


def build_digest_graph():
    """Compile the topic/ID -> briefing pipeline."""
    g = StateGraph(AgentState)

    g.add_node("query_understanding", query_understanding)
    g.add_node("arxiv_retrieval", arxiv_retrieval)
    g.add_node("selection", selection)
    g.add_node("fetch_parse", fetch_parse)
    g.add_node("chunk_embed", chunk_embed)
    g.add_node("summarize", summarize)

    g.set_entry_point("query_understanding")

    # query_understanding may set an error (empty input).
    g.add_conditional_edges(
        "query_understanding", _has_error, {"ok": "arxiv_retrieval", "error": END}
    )
    g.add_conditional_edges(
        "arxiv_retrieval", _has_error, {"ok": "selection", "error": END}
    )
    g.add_conditional_edges(
        "selection", _has_error, {"ok": "fetch_parse", "error": END}
    )
    g.add_conditional_edges(
        "fetch_parse", _parse_ok, {"ok": "chunk_embed", "error": END}
    )
    g.add_edge("chunk_embed", "summarize")
    g.add_edge("summarize", END)

    return g.compile()


def build_qa_graph():
    """Compile the single-turn RAG QA graph."""
    g = StateGraph(AgentState)
    g.add_node("retrieve_chunks", retrieve_chunks)
    g.add_node("generate_answer", generate_answer)
    g.set_entry_point("retrieve_chunks")
    g.add_edge("retrieve_chunks", "generate_answer")
    g.add_edge("generate_answer", END)
    return g.compile()
