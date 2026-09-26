"""Tests for the tolerant JSON extraction used by chat_json()."""
import pytest

from src.llm import LLMError, _extract_json


def test_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json_block():
    text = 'Here you go:\n```json\n{"choice": 2, "reason": "best"}\n```\nthanks'
    assert _extract_json(text) == {"choice": 2, "reason": "best"}


def test_fenced_without_lang():
    text = "```\n{\"x\": true}\n```"
    assert _extract_json(text) == {"x": True}


def test_json_with_surrounding_prose():
    text = 'The answer is {"k": [1, 2, 3]} — hope that helps.'
    assert _extract_json(text) == {"k": [1, 2, 3]}


def test_no_json_raises():
    with pytest.raises(LLMError):
        _extract_json("there is no object here")


def test_malformed_json_raises():
    with pytest.raises(LLMError):
        _extract_json('{"a": 1,,,}')
