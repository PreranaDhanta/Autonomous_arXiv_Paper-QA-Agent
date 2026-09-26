"""Tests for the graph's conditional-edge routing helpers."""
from src.graph import _has_error, _parse_ok


def test_has_error_ok():
    assert _has_error({}) == "ok"
    assert _has_error({"error": None}) == "ok"


def test_has_error_set():
    assert _has_error({"error": "boom"}) == "error"


def test_parse_ok_with_full_text():
    assert _parse_ok({"parsed": {"full_text": "lots of text"}}) == "ok"


def test_parse_ok_with_only_abstract():
    assert _parse_ok({"parsed": {"full_text": "", "abstract": "an abstract"}}) == "ok"


def test_parse_ok_error_when_nothing():
    assert _parse_ok({"parsed": {"full_text": "", "abstract": ""}}) == "error"


def test_parse_ok_error_propagates_upstream_error():
    assert _parse_ok({"error": "download failed", "parsed": {"full_text": "x"}}) == "error"
