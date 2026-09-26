"""Thin wrapper around the official arXiv API (via the `arxiv` package).

Responsible only for (a) normalising user input into an arXiv ID when possible,
and (b) turning API results into plain metadata dicts the rest of the graph uses.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import arxiv

# Modern IDs: 2401.12345 / 2401.12345v2 ; legacy: hep-th/9901001
_NEW_ID = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_OLD_ID = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")


def extract_arxiv_id(text: str) -> Optional[str]:
    """Return a bare arXiv ID if the input is an ID or an arxiv.org URL, else None."""
    text = text.strip()
    # URL forms: /abs/<id>, /pdf/<id>, /pdf/<id>.pdf
    url = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", text, re.IGNORECASE)
    if url:
        candidate = url.group(1).replace(".pdf", "")
        text = candidate
    m = _NEW_ID.search(text)
    if m:
        return m.group(1)  # drop version suffix for a stable collection key
    m = _OLD_ID.search(text)
    if m:
        return m.group(1)
    return None


def _to_dict(result: "arxiv.Result") -> dict[str, Any]:
    """Flatten an arxiv.Result into a JSON-serialisable metadata dict."""
    short_id = result.get_short_id()
    return {
        "arxiv_id": short_id.split("v")[0],
        "version_id": short_id,
        "title": (result.title or "").strip().replace("\n", " "),
        "authors": [a.name for a in result.authors],
        "abstract": (result.summary or "").strip().replace("\n", " "),
        "pdf_url": result.pdf_url,
        "abs_url": result.entry_id,
        "categories": result.categories,
        "primary_category": result.primary_category,
        "published": result.published.date().isoformat() if result.published else None,
        "updated": result.updated.date().isoformat() if result.updated else None,
    }


def fetch_by_id(arxiv_id: str) -> Optional[dict[str, Any]]:
    """Fetch a single paper's metadata by ID. Returns None if not found."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    for result in client.results(search):
        return _to_dict(result)
    return None


def search_topic(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    """Search arXiv by relevance for a natural-language topic."""
    client = arxiv.Client(page_size=max_results, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [_to_dict(r) for r in client.results(search)]
