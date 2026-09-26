"""Tests for the text-only helpers in the PDF parser (no real PDF needed)."""
from src.pdf_parser import _clean, _find_abstract, _split_sections


def test_clean_collapses_whitespace_and_nulls():
    raw = "hello\x00   world\n\n\n\nnext"
    cleaned = _clean(raw)
    assert "\x00" not in cleaned
    assert "  " not in cleaned          # runs of spaces collapsed
    assert "\n\n\n" not in cleaned      # 3+ newlines collapsed


def test_split_sections_basic():
    text = (
        "Abstract\nWe study things.\n"
        "1 Introduction\nMotivation here.\n"
        "2 Method\nWe do X.\n"
        "Conclusion\nWe conclude.\n"
    )
    sections = _split_sections(text)
    assert "abstract" in sections
    assert "introduction" in sections
    assert "method" in sections
    assert "conclusion" in sections
    assert "We do X." in sections["method"]


def test_split_sections_normalises_aliases():
    text = "Methods\nour approach\nConclusions\ndone\n"
    sections = _split_sections(text)
    # "Methods" -> "method", "Conclusions" -> "conclusion"
    assert "method" in sections
    assert "conclusion" in sections


def test_split_sections_no_headings_returns_empty():
    assert _split_sections("just some body text with no headings") == {}


def test_find_abstract_from_sections():
    sections = {"abstract": "this is the abstract"}
    assert _find_abstract("", sections) == "this is the abstract"


def test_find_abstract_fallback_between_markers():
    text = "Title\nAbstract This is inline abstract text. 1 Introduction begins"
    got = _find_abstract(text, {})
    assert "inline abstract text" in got
