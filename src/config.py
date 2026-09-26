"""Central configuration, loaded once from environment / .env.

Keeping all tunables in one typed object makes the rest of the codebase easy to
read and makes the "which knobs exist" question answerable at a glance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    """All runtime settings for the agent."""

    # LLM provider selection
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1").strip()

    # Storage
    chroma_dir: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_DIR", "./data/chroma")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./data/output")))

    # Retrieval / parsing
    chunk_size: int = _int("CHUNK_SIZE", 1100)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)
    qa_top_k: int = _int("QA_TOP_K", 6)
    max_pdf_pages: int = _int("MAX_PDF_PAGES", 80)
    topic_max_results: int = _int("TOPIC_MAX_RESULTS", 8)

    def __post_init__(self) -> None:
        self.chroma_dir = Path(self.chroma_dir)
        self.output_dir = Path(self.output_dir)
        self.pdf_dir = self.output_dir / "pdfs"
        self.briefing_dir = self.output_dir / "briefings"
        self.session_dir = self.output_dir / "sessions"
        for d in (self.chroma_dir, self.pdf_dir, self.briefing_dir, self.session_dir):
            d.mkdir(parents=True, exist_ok=True)

    def describe_llm(self) -> str:
        if self.llm_provider == "groq":
            return f"groq / {self.groq_model}"
        if self.llm_provider == "gemini":
            return f"gemini / {self.gemini_model}"
        if self.llm_provider == "ollama":
            return f"ollama / {self.ollama_model}"
        return self.llm_provider


# Single shared instance.
config = Config()
