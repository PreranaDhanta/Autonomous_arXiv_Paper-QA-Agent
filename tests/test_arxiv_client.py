"""Tests for arXiv ID extraction — pure regex logic, no network."""
from src.arxiv_client import extract_arxiv_id


def test_bare_modern_id():
    assert extract_arxiv_id("2401.12345") == "2401.12345"


def test_modern_id_with_version_is_stripped():
    # Version suffix is dropped so the collection key is stable across versions.
    assert extract_arxiv_id("2401.12345v3") == "2401.12345"


def test_abs_url():
    assert extract_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"


def test_pdf_url_with_extension():
    assert extract_arxiv_id("https://arxiv.org/pdf/1706.03762.pdf") == "1706.03762"


def test_pdf_url_with_version():
    assert extract_arxiv_id("http://arxiv.org/pdf/2009.08553v4") == "2009.08553"


def test_legacy_id():
    assert extract_arxiv_id("hep-th/9901001") == "hep-th/9901001"


def test_id_embedded_in_text():
    assert extract_arxiv_id("please digest 2401.12345 for me") == "2401.12345"


def test_topic_returns_none():
    assert extract_arxiv_id("recent work on KV-cache compression") is None


def test_empty_returns_none():
    assert extract_arxiv_id("") is None
