"""Tests for briefing normalisation — especially the 'limitations never empty' rule."""
from src.briefing import normalise_briefing, render_markdown

_META = {
    "title": "A Paper",
    "authors": ["Ada Lovelace", "Alan Turing"],
    "arxiv_id": "2401.00001",
    "published": "2024-01-01",
    "categories": ["cs.LG"],
    "abs_url": "http://arxiv.org/abs/2401.00001",
    "pdf_url": "http://arxiv.org/pdf/2401.00001",
}


def test_metadata_is_attached():
    b = normalise_briefing({}, _META)
    assert b["title"] == "A Paper"
    assert b["arxiv_id"] == "2401.00001"
    assert b["link"] == "http://arxiv.org/abs/2401.00001"


def test_limitations_never_empty():
    # The spec requires limitations to always be present.
    b = normalise_briefing({"limitations": []}, _META)
    assert b["limitations"]
    assert isinstance(b["limitations"], list)


def test_list_fields_coerced_from_string():
    b = normalise_briefing({"method": "a single string method"}, _META)
    assert b["method"] == ["a single string method"]


def test_string_field_coerced_from_list():
    b = normalise_briefing({"problem_statement": ["part one", "part two"]}, _META)
    assert b["problem_statement"] == "part one part two"


def test_blank_items_filtered_out():
    b = normalise_briefing({"key_results": ["  ", "real result", ""]}, _META)
    assert b["key_results"] == ["real result"]


def test_render_markdown_contains_all_sections():
    b = normalise_briefing(
        {
            "plain_english_summary": "why it matters",
            "problem_statement": "the problem",
            "method": ["step 1"],
            "key_results": ["result 1"],
            "limitations": ["limit 1"],
            "follow_up_questions": ["q1?"],
        },
        _META,
    )
    md = render_markdown(b)
    for heading in (
        "# A Paper",
        "## Why this paper matters",
        "## Problem statement",
        "## Method / approach",
        "## Key results / claims",
        "## Limitations",
        "## Suggested follow-up questions",
    ):
        assert heading in md


def test_render_markdown_includes_warnings():
    b = normalise_briefing({}, _META)
    md = render_markdown(b, warnings=["PDF was scanned"])
    assert "Processing notes" in md
    assert "PDF was scanned" in md
