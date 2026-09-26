"""Download and parse arXiv PDFs with PyMuPDF.

Extracts, at minimum: abstract, sections (heading -> text), a references block,
and the full text. Designed to degrade gracefully:
  * huge PDFs      -> parse only the first ``max_pages`` pages (configurable)
  * scanned/empty  -> flags quality="poor"; caller falls back to the abstract
  * broken layout  -> best-effort; never raises to the caller for parse issues
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests

# Common section headings in ML/CS papers. Used as anchors to split the body.
_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\d*\s+)?"
    r"(abstract|introduction|related work|background|preliminaries|method(?:s|ology)?|"
    r"approach|model|architecture|experiment(?:s|al setup)?|results?|evaluation|"
    r"discussion|analysis|ablation(?:s| study)?|conclusion(?:s)?|"
    r"limitations?|future work|references|acknowledg(?:e?ments?))\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def download_pdf(pdf_url: str, dest: Path) -> Path:
    """Download a PDF to ``dest`` (skips if already present)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    resp = requests.get(pdf_url, timeout=120, headers={"User-Agent": "arxiv-digest-agent/1.0"})
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def parse_pdf(path: Path, max_pages: int = 80) -> dict[str, Any]:
    """Parse a PDF into a structured dict. Never raises on content problems."""
    result: dict[str, Any] = {
        "full_text": "",
        "sections": {},
        "abstract": "",
        "references": "",
        "num_pages": 0,
        "quality": "good",
        "notes": [],
    }
    try:
        doc = fitz.open(path)
    except Exception as exc:  # corrupt / unreadable file
        result["quality"] = "unreadable"
        result["notes"].append(f"Could not open PDF: {exc}")
        return result

    total = doc.page_count
    result["num_pages"] = total
    limit = total if max_pages in (0, None) else min(total, max_pages)
    if limit < total:
        result["notes"].append(f"Large PDF: parsed first {limit} of {total} pages.")

    pages_text: list[str] = []
    for i in range(limit):
        try:
            pages_text.append(doc[i].get_text("text"))
        except Exception:
            continue
    doc.close()

    full_text = _clean("\n".join(pages_text))
    result["full_text"] = full_text

    # Scanned / image-only PDFs yield almost no extractable text.
    alpha = sum(c.isalpha() for c in full_text)
    if alpha < 500:
        result["quality"] = "poor"
        result["notes"].append(
            "Very little machine-readable text extracted (likely scanned/image-based)."
        )
        return result

    result["sections"] = _split_sections(full_text)
    result["abstract"] = _find_abstract(full_text, result["sections"])
    result["references"] = result["sections"].get("references", "")
    return result


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sections(text: str) -> dict[str, str]:
    """Split body text at recognised headings into {normalised_heading: text}."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for idx, m in enumerate(matches):
        name = m.group(1).lower().strip()
        # Normalise a few aliases so downstream lookups are predictable.
        name = {
            "methods": "method",
            "methodology": "method",
            "conclusions": "conclusion",
            "acknowledgements": "acknowledgments",
            "acknowledgement": "acknowledgments",
        }.get(name, name)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            # Keep the first occurrence of each heading (front matter wins).
            sections.setdefault(name, body)
    return sections


def _find_abstract(text: str, sections: dict[str, str]) -> str:
    if sections.get("abstract"):
        # Trim at the next heading if the abstract ran long.
        abs_text = sections["abstract"]
        return abs_text[:3000].strip()
    # Fallback: text between "Abstract" and "Introduction".
    m = re.search(r"abstract(.*?)(introduction|1\s+introduction)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:3000]
    return ""
