"""Executive-briefing prompt, schema normalisation, and Markdown rendering."""
from __future__ import annotations

from typing import Any

# Fields required by the assessment spec.
BRIEFING_FIELDS = [
    "plain_english_summary",
    "problem_statement",
    "method",              # list[str]
    "key_results",         # list[str]
    "limitations",         # list[str]
    "follow_up_questions", # list[str]
]

SUMMARY_SYSTEM = """You are a meticulous research analyst. You write concise, \
accurate executive briefings of arXiv papers for busy engineers and researchers.

Rules:
- Ground every statement in the provided paper text. Do NOT invent results, \
numbers, or citations.
- If the text does not support a field, say so briefly (e.g. "Not clearly stated \
in the extracted text") rather than fabricating.
- Limitations must be explicit and specific — never leave them empty. If the \
authors state none, infer reasonable ones from the method and mark them as \
"(reviewer-inferred)".
- Keep the plain-English summary to a single tight paragraph."""


def build_summary_prompt(meta: dict[str, Any], parsed: dict[str, Any]) -> str:
    """Assemble the user prompt: metadata + the most informative sections."""
    sections = parsed.get("sections") or {}
    # Prioritise high-signal sections; truncate to stay within context limits.
    wanted = [
        "abstract", "introduction", "method", "approach", "model",
        "results", "experiment", "evaluation", "discussion",
        "conclusion", "limitations",
    ]
    parts: list[str] = []
    budget = 14000  # characters of body context
    used = 0

    abstract = parsed.get("abstract") or meta.get("abstract", "")
    if abstract:
        parts.append(f"## ABSTRACT\n{abstract[:2500]}")
        used += min(len(abstract), 2500)

    for name in wanted:
        if name == "abstract":
            continue
        body = sections.get(name)
        if not body:
            continue
        take = min(len(body), max(0, budget - used))
        if take <= 0:
            break
        parts.append(f"## {name.upper()}\n{body[:take]}")
        used += take

    if used < 200:
        # Poor parse — fall back to the arXiv abstract only.
        parts = [f"## ABSTRACT (metadata fallback)\n{meta.get('abstract', '')}"]

    body_text = "\n\n".join(parts)
    return f"""Paper metadata:
- Title: {meta.get('title')}
- Authors: {', '.join(meta.get('authors', [])[:12])}
- arXiv ID: {meta.get('arxiv_id')}
- Published: {meta.get('published')}
- Categories: {', '.join(meta.get('categories', []))}

Paper content (extracted):
{body_text}

Produce a JSON object with EXACTLY these keys:
- "plain_english_summary": string (one paragraph: why this paper matters)
- "problem_statement": string
- "method": array of short bullet strings
- "key_results": array of short bullet strings (include concrete numbers when present)
- "limitations": array of short bullet strings (never empty)
- "follow_up_questions": array of 3-5 questions a reader might ask
"""


def normalise_briefing(raw: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce the LLM output into a stable shape and attach metadata."""
    def as_list(v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    def as_str(v: Any) -> str:
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
        return str(v).strip() if v else ""

    briefing = {
        "title": meta.get("title"),
        "authors": meta.get("authors", []),
        "arxiv_id": meta.get("arxiv_id"),
        "published": meta.get("published"),
        "categories": meta.get("categories", []),
        "link": meta.get("abs_url"),
        "pdf_url": meta.get("pdf_url"),
        "plain_english_summary": as_str(raw.get("plain_english_summary")),
        "problem_statement": as_str(raw.get("problem_statement")),
        "method": as_list(raw.get("method")),
        "key_results": as_list(raw.get("key_results")),
        "limitations": as_list(raw.get("limitations")) or ["Not clearly stated in the extracted text."],
        "follow_up_questions": as_list(raw.get("follow_up_questions")),
    }
    return briefing


def render_markdown(b: dict[str, Any], warnings: list[str] | None = None) -> str:
    """Render the briefing dict as a readable Markdown document."""
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {i}" for i in items) if items else "- _None provided_"

    authors = ", ".join(b.get("authors", []))
    lines = [
        f"# {b.get('title')}",
        "",
        f"**Authors:** {authors}  ",
        f"**arXiv ID:** {b.get('arxiv_id')}  ",
        f"**Published:** {b.get('published')}  ",
        f"**Categories:** {', '.join(b.get('categories', []))}  ",
        f"**Link:** {b.get('link')}  ",
        f"**PDF:** {b.get('pdf_url')}",
        "",
        "## Why this paper matters",
        b.get("plain_english_summary", ""),
        "",
        "## Problem statement",
        b.get("problem_statement", ""),
        "",
        "## Method / approach",
        bullets(b.get("method", [])),
        "",
        "## Key results / claims",
        bullets(b.get("key_results", [])),
        "",
        "## Limitations",
        bullets(b.get("limitations", [])),
        "",
        "## Suggested follow-up questions",
        bullets(b.get("follow_up_questions", [])),
    ]
    if warnings:
        lines += ["", "## ⚠️ Processing notes", bullets(warnings)]
    return "\n".join(lines) + "\n"
