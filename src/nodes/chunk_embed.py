"""Node 5 — Chunk & Embed.

Section-aware chunking, then store embeddings in the persistent Chroma DB. The
collection name is recorded in state so the QA stage can reattach to it later.
"""
from __future__ import annotations

from ..config import config
from ..state import AgentState
from ..vector_store import chunk_text, get_store


def chunk_embed(state: AgentState) -> AgentState:
    parsed = state.get("parsed") or {}
    meta = state["selected"]
    warnings = list(state.get("warnings", []))

    chunks = chunk_text(parsed, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    if not chunks:
        warnings.append("No text available to embed; QA will be unavailable for this paper.")
        return {"collection_name": None, "num_chunks": 0, "warnings": warnings}

    try:
        name = get_store().build(meta["arxiv_id"], chunks)
    except Exception as exc:
        warnings.append(f"Vector store build failed ({exc}); QA disabled.")
        return {"collection_name": None, "num_chunks": 0, "warnings": warnings}

    return {"collection_name": name, "num_chunks": len(chunks), "warnings": warnings}
