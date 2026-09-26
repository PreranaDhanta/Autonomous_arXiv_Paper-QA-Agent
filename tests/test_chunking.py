"""Tests for section-aware chunking and the sliding window."""
from src.vector_store import _window, chunk_text


def test_window_short_text_single_chunk():
    assert _window("hello world", size=100, overlap=10) == ["hello world"]


def test_window_empty():
    assert _window("", size=100, overlap=10) == []


def test_window_splits_long_text_with_overlap():
    text = "a. " * 400  # ~1200 chars
    chunks = _window(text, size=300, overlap=50)
    assert len(chunks) > 1
    # Every chunk respects the size bound (allowing sentence-boundary trimming).
    assert all(len(c) <= 300 for c in chunks)


def test_window_covers_all_text():
    text = "word " * 200
    chunks = _window(text, size=200, overlap=40)
    # Reassembled coverage: last chunk should reach the end.
    assert chunks[-1].strip().endswith("word")


def test_chunk_text_attaches_section_metadata():
    parsed = {
        "sections": {"introduction": "x" * 500, "method": "y" * 500},
        "abstract": "an abstract that is long enough to survive the min length filter",
        "full_text": "",
    }
    chunks = chunk_text(parsed, chunk_size=200, overlap=20)
    sections = {c["section"] for c in chunks}
    assert "introduction" in sections
    assert "method" in sections
    # Abstract is injected as its own high-signal chunk.
    assert "abstract" in sections
    # Indices are unique and sequential.
    assert [c["index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_text_falls_back_to_full_text_without_sections():
    parsed = {"sections": {}, "abstract": "", "full_text": "z" * 400}
    chunks = chunk_text(parsed, chunk_size=150, overlap=20)
    assert chunks
    assert all(c["section"] == "body" for c in chunks)


def test_chunk_text_skips_tiny_fragments():
    parsed = {"sections": {"tiny": "short"}, "abstract": "", "full_text": ""}
    # "short" is < 40 chars, so it is dropped.
    assert chunk_text(parsed) == []
