"""Chunking + persistent Chroma vector store.

Design choices:
  * Section-aware chunking: we chunk each parsed section separately so a chunk
    never straddles a section boundary, and we attach the section name as
    metadata. This improves retrieval precision and lets QA answers cite where
    the evidence came from.
  * Character windows with overlap: simple, deterministic, and framework-free.
  * Embeddings: Chroma's built-in local model (all-MiniLM-L6-v2 via ONNX). It is
    free, runs offline after a one-time download, and needs no API key.
  * Persistence: a PersistentClient on disk, so embeddings survive between the
    summarisation run and a later QA session.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from .config import config


def chunk_text(
    parsed: dict[str, Any],
    chunk_size: int = 1100,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    """Turn parsed PDF output into a list of {text, section, index} chunks."""
    sections: dict[str, str] = parsed.get("sections") or {}
    chunks: list[dict[str, Any]] = []

    # If we failed to detect sections, fall back to chunking the full text.
    source_map = sections if sections else {"body": parsed.get("full_text", "")}
    # Always ensure the abstract is present as its own chunk (high-signal).
    if parsed.get("abstract") and "abstract" not in source_map:
        source_map = {"abstract": parsed["abstract"], **source_map}

    for section, body in source_map.items():
        for piece in _window(body, chunk_size, overlap):
            piece = piece.strip()
            if len(piece) < 40:  # skip near-empty fragments
                continue
            chunks.append({"text": piece, "section": section, "index": len(chunks)})
    return chunks


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Sliding character window that prefers to break on sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        window = text[start:end]
        # Try to end on a sentence boundary for cleaner chunks.
        if end < len(text):
            m = list(re.finditer(r"[.!?]\s", window))
            if m and m[-1].end() > size * 0.5:
                window = window[: m[-1].end()]
                end = start + m[-1].end()
        out.append(window)
        start += max(step, len(window) - overlap)
    return out


class VectorStore:
    """A per-paper Chroma collection with add + similarity-search helpers."""

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(config.chroma_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

    @staticmethod
    def collection_name(arxiv_id: str) -> str:
        # Chroma names must be alphanumeric/._- and 3-63 chars.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", arxiv_id)
        return f"paper_{safe}"

    def build(self, arxiv_id: str, chunks: list[dict[str, Any]]) -> str:
        """(Re)create a collection for a paper and add its chunks. Returns the name."""
        name = self.collection_name(arxiv_id)
        # Rebuild cleanly to avoid duplicate documents on re-run.
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        collection = self.client.create_collection(
            name=name, metadata={"arxiv_id": arxiv_id, "hnsw:space": "cosine"}
        )
        collection.add(
            ids=[f"{arxiv_id}-{c['index']}" for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"section": c["section"], "index": c["index"]} for c in chunks],
        )
        return name

    def query(self, collection_name: str, question: str, top_k: int = 6) -> list[dict[str, Any]]:
        """Return the top_k most similar chunks with their cosine distances."""
        collection = self.client.get_collection(collection_name)
        res = collection.query(query_texts=[question], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: list[dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "text": doc,
                    "section": (meta or {}).get("section", "?"),
                    "index": (meta or {}).get("index"),
                    "distance": dist,          # cosine distance: lower = more similar
                    "similarity": 1.0 - dist,  # convenience
                }
            )
        return out

    def exists(self, collection_name: str) -> bool:
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
