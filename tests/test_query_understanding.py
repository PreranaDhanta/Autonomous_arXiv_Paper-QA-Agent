"""Tests for query understanding.

The paper-lookup path is deterministic (regex, no LLM), so it is tested directly.
The topic path calls the LLM, so we stub it to keep the test offline.
"""
import importlib

# Load the submodule explicitly: src.nodes.__init__ re-exports the function under
# the same name, which would otherwise shadow the module on attribute access.
qu = importlib.import_module("src.nodes.query_understanding")


def test_empty_input_sets_error():
    out = qu.query_understanding({"raw_input": "   "})
    assert "error" in out


def test_paper_lookup_by_id_no_llm():
    out = qu.query_understanding({"raw_input": "2401.12345"})
    assert out["intent"] == "paper_lookup"
    assert out["target_arxiv_id"] == "2401.12345"
    assert out["arxiv_query"] == "2401.12345"


def test_paper_lookup_by_url():
    out = qu.query_understanding({"raw_input": "https://arxiv.org/abs/1706.03762"})
    assert out["intent"] == "paper_lookup"
    assert out["target_arxiv_id"] == "1706.03762"


def test_topic_search_uses_refined_query(monkeypatch):
    class _StubLLM:
        def chat(self, *a, **k):
            return "kv cache compression llm"

    monkeypatch.setattr(qu, "get_llm", lambda: _StubLLM())
    out = qu.query_understanding({"raw_input": "recent work on KV-cache compression"})
    assert out["intent"] == "topic_search"
    assert out["arxiv_query"] == "kv cache compression llm"
    assert out["target_arxiv_id"] is None


def test_topic_search_falls_back_on_empty_llm_output(monkeypatch):
    # Regression test for the IndexError bug: empty LLM output must not crash.
    class _EmptyLLM:
        def chat(self, *a, **k):
            return "\n   \n"

    monkeypatch.setattr(qu, "get_llm", lambda: _EmptyLLM())
    raw = "some vague topic"
    out = qu.query_understanding({"raw_input": raw})
    assert out["intent"] == "topic_search"
    assert out["arxiv_query"] == raw  # fell back to the raw query, no crash
