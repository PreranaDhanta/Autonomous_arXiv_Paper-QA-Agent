"""Node 4 — Fetch & Parse.

Downloads the selected paper's PDF and extracts structured text. Handles the
"PDF fails to parse cleanly" failure case: on an unreadable/scanned/empty PDF we
record a warning and continue with the arXiv abstract as a fallback body, rather
than aborting the whole run.
"""
from __future__ import annotations

from pathlib import Path

from ..config import config
from ..pdf_parser import download_pdf, parse_pdf
from ..state import AgentState


def fetch_parse(state: AgentState) -> AgentState:
    meta = state["selected"]
    arxiv_id = meta["arxiv_id"]
    warnings = list(state.get("warnings", []))

    dest = config.pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    try:
        download_pdf(meta["pdf_url"], dest)
    except Exception as exc:
        # Download failure is not fatal — fall back to abstract-only.
        warnings.append(f"PDF download failed ({exc}); using abstract only.")
        parsed = _abstract_only(meta)
        return {"pdf_path": None, "parsed": parsed, "warnings": warnings}

    parsed = parse_pdf(Path(dest), max_pages=config.max_pdf_pages)
    for note in parsed.get("notes", []):
        warnings.append(note)

    if parsed.get("quality") in {"poor", "unreadable"} or not parsed.get("full_text"):
        warnings.append(
            "PDF text extraction was poor; briefing will rely on the abstract. "
            "QA answers may be limited."
        )
        # Seed the store with the abstract so QA still has something grounded.
        fallback = _abstract_only(meta)
        # Keep any partial text we did manage to extract.
        if parsed.get("full_text"):
            fallback["full_text"] = parsed["full_text"]
            fallback["sections"] = parsed.get("sections", {})
        parsed = fallback

    return {"pdf_path": str(dest), "parsed": parsed, "warnings": warnings}


def _abstract_only(meta: dict) -> dict:
    abstract = meta.get("abstract", "")
    return {
        "full_text": abstract,
        "sections": {"abstract": abstract} if abstract else {},
        "abstract": abstract,
        "references": "",
        "num_pages": 0,
        "quality": "abstract_only",
        "notes": [],
    }
