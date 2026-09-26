"""A tiny, dependency-light LLM abstraction over three FREE options.

We deliberately avoid provider SDKs and just speak HTTP with ``requests``. That
keeps the dependency tree small and version conflicts unlikely, and makes the
"how do I add a provider" story obvious.

Providers:
  * groq   — free tier, OpenAI-compatible chat endpoint
  * gemini — Google AI Studio free tier
  * ollama — fully local, no API key

All providers expose the same ``chat()`` and ``chat_json()`` interface.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import requests

from .config import config


class LLMError(RuntimeError):
    """Raised when the LLM cannot be reached or returns an unusable response."""


class LLMClient:
    """Uniform chat interface across Groq / Gemini / Ollama."""

    def __init__(self) -> None:
        self.provider = config.llm_provider
        self._validate()

    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        if self.provider == "groq" and not config.groq_api_key:
            raise LLMError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty. "
                "Get a free key at https://console.groq.com/keys or switch "
                "LLM_PROVIDER to `ollama` for a fully local, no-key setup."
            )
        if self.provider == "gemini" and not config.gemini_api_key:
            raise LLMError(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is empty. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        if self.provider not in {"groq", "gemini", "ollama"}:
            raise LLMError(f"Unknown LLM_PROVIDER: {self.provider!r}")

    # ------------------------------------------------------------------ #
    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1600,
        retries: int = 3,
    ) -> str:
        """Return the model's text response. Retries with backoff on transient errors."""
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if self.provider == "groq":
                    return self._groq(system, user, temperature, max_tokens)
                if self.provider == "gemini":
                    return self._gemini(system, user, temperature, max_tokens)
                return self._ollama(system, user, temperature, max_tokens)
            except (requests.RequestException, LLMError) as exc:
                last_err = exc
                # Exponential backoff; helps with free-tier rate limits.
                sleep = 2 ** attempt
                time.sleep(sleep)
        raise LLMError(f"LLM call failed after {retries} attempts: {last_err}")

    def chat_json(self, system: str, user: str, *, max_tokens: int = 2200) -> dict:
        """Chat and parse a JSON object out of the response, tolerant of fences/prose."""
        system_json = system + "\n\nRespond with a single valid JSON object and nothing else."
        raw = self.chat(system_json, user, temperature=0.1, max_tokens=max_tokens)
        return _extract_json(raw)

    # ------------------------- providers ------------------------------ #
    def _groq(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.groq_api_key}"},
            json={
                "model": config.groq_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=90,
        )
        if resp.status_code == 429:
            raise LLMError("Groq rate limit (429). Wait a moment and retry.")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _gemini(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{config.gemini_model}:generateContent?key={config.gemini_api_key}"
        )
        resp = requests.post(
            url,
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
            timeout=90,
        )
        if resp.status_code == 429:
            raise LLMError("Gemini rate limit (429). Wait a moment and retry.")
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response: {data}") from exc

    def _ollama(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        try:
            resp = requests.post(
                f"{config.ollama_base_url}/api/chat",
                json={
                    "model": config.ollama_model,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=180,
            )
        except requests.ConnectionError as exc:
            raise LLMError(
                "Could not reach Ollama. Is it running? Start it with `ollama serve` "
                f"and pull a model, e.g. `ollama pull {config.ollama_model}`."
            ) from exc
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    # 1) Fenced ```json ... ``` block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    # 2) Otherwise the first {...} spanning the string
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        raise LLMError(f"No JSON object found in LLM response:\n{text[:400]}")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Malformed JSON from LLM: {exc}\n{candidate[:400]}") from exc


# Lazily-created shared client so importing this module never triggers a network
# call or a hard failure when keys are absent (useful for tests / --help).
_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
