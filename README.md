# Autonomous arXiv Paper Digest & QA Agent

An autonomous agent that takes an **arXiv ID/URL or a research topic**, finds the
right paper, reads it, writes an **executive briefing**, and then answers
**grounded follow-up questions** about it — all runnable on a free LLM tier or
fully locally.

It is built as an **explicit agentic state graph** (nodes + edges + shared state)
using [LangGraph](https://github.com/langchain-ai/langgraph), with graceful
failure handling at every step.

---

## What it does

```
topic / arXiv ID  ─▶  Executive Briefing  ─▶  Interactive grounded QA
```

1. **Understands** the input — an arXiv ID/URL (deterministic regex) or a
   natural-language topic (LLM-refined keywords).
2. **Retrieves** candidates from the official arXiv API.
3. **Selects** the most relevant paper (LLM ranking for topic searches; keeps
   runners-up as alternatives).
4. **Fetches & parses** the PDF into structured sections with PyMuPDF, degrading
   gracefully on scanned/broken PDFs (falls back to the abstract).
5. **Chunks & embeds** the text section-aware into a persistent local Chroma
   vector store (free on-device `all-MiniLM-L6-v2` embeddings — no API key).
6. **Summarises** into a structured briefing: plain-English summary, problem,
   method, key results, explicit limitations, and follow-up questions.
7. **Answers questions** via RAG — retrieval + an answer grounded *only* in the
   retrieved chunks, with section citations and a "not in the paper" guardrail.

## Architecture

```
DIGEST graph (topic/ID ─▶ briefing)          QA graph (question ─▶ answer)

  query_understanding                          retrieve_chunks
        │                                             │
   arxiv_retrieval ──(no results)──▶ END         generate_answer
        │                                             │
     selection                                       END
        │
    fetch_parse ──(nothing to work with)──▶ END
        │
    chunk_embed
        │
     summarize ──▶ END
```

A single shared [`AgentState`](src/state.py) is threaded through every node; each
node returns a partial update that LangGraph merges back in. Any node that sets
`state["error"]` routes the run straight to `END` with a helpful message instead
of crashing downstream.

**State persistence** between the summarisation and QA stages lives in two places:
- **Embeddings/chunks** → persisted on disk by Chroma (`data/chroma`).
- **Everything else** (metadata, briefing, parsed sections, QA history) → a small
  JSON session file keyed by arXiv ID (`data/output/sessions/<id>.json`).

A later `qa` command reloads the JSON and reattaches to the existing Chroma
collection — no re-parsing needed.

| Path | Module |
|------|--------|
| CLI entry point | [main.py](main.py) |
| Graph wiring | [src/graph.py](src/graph.py) |
| Shared state | [src/state.py](src/state.py) |
| Nodes | [src/nodes/](src/nodes/) |
| arXiv client | [src/arxiv_client.py](src/arxiv_client.py) |
| PDF parsing | [src/pdf_parser.py](src/pdf_parser.py) |
| Chunking + vector store | [src/vector_store.py](src/vector_store.py) |
| Briefing prompt + rendering | [src/briefing.py](src/briefing.py) |
| Session persistence | [src/session.py](src/session.py) |
| Config | [src/config.py](src/config.py) |
| LLM abstraction (Groq/Gemini/Ollama) | [src/llm.py](src/llm.py) |

## Quick start (TL;DR)

```bash
# 1. Clone
git clone https://github.com/PreranaDhanta/Autonomous_arXiv_Paper-QA-Agent.git
cd Autonomous_arXiv_Paper-QA-Agent

# 2. Set up (creates .venv, installs deps, seeds .env from the template)
./setup.sh

# 3. Add ONE free LLM key — edit .env and set, e.g.:
#      LLM_PROVIDER=groq
#      GROQ_API_KEY=gsk_your_key_here     # free: https://console.groq.com/keys

# 4. Activate the environment
source .venv/bin/activate

# 5. Run it
python main.py digest 1706.03762
```

That's it. `digest` fetches the paper, prints an executive briefing, and drops
you into an interactive QA chat (type a question, press Enter, `exit` to quit).

### Everyday commands

```bash
source .venv/bin/activate     # activate the venv first (once per terminal)

# Digest a specific paper by ID or URL, then interactive QA
python main.py digest 1706.03762
python main.py digest https://arxiv.org/abs/1706.03762

# Digest by topic — the agent finds and picks the most relevant paper, then QA
python main.py digest "retrieval augmented generation for open domain question answering"

# Briefing only, skip the QA chat
python main.py digest 2009.08553 --no-qa

# Resume QA on a paper you already digested (no re-parsing, reuses saved state)
python main.py qa 2009.08553

# List saved sessions
python main.py list
```

One-liner without activating the venv first:

```bash
.venv/bin/python main.py digest 1706.03762
```

## Setup

One command creates the virtual environment, installs dependencies, and seeds
`.env`:

```bash
./setup.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # or requirements.lock.txt for pinned versions
cp .env.example .env
```

Then edit `.env` and configure **one** LLM provider (retrieval, PDF parsing and
embeddings need no key — only the summary and QA steps use the LLM):

| Provider | Cost | How |
|----------|------|-----|
| **Groq** | Free tier, fast | Set `LLM_PROVIDER=groq` and `GROQ_API_KEY` from https://console.groq.com/keys |
| **Gemini** | Free tier | Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` from https://aistudio.google.com/app/apikey |
| **Ollama** | Local, no key | Set `LLM_PROVIDER=ollama`, run `ollama serve`, `ollama pull llama3.1` |

> **Corporate networks:** this project depends on
> [`truststore`](https://pypi.org/project/truststore/) and injects it on import
> ([src/__init__.py](src/__init__.py)) so Python uses the OS-native trust store
> (macOS Keychain / Windows / Linux). This lets HTTPS work behind corporate
> TLS-inspection proxies whose root CA `certifi` does not ship.

Briefings are also written to `data/output/briefings/<id>.md`.

## Configuration knobs

All tunables live in `.env` (see [.env.example](.env.example)): chunk size and
overlap, QA top-k, max PDF pages, topic result count, and storage directories.

## Design notes

- **No provider SDKs** — the LLM layer speaks plain HTTP via `requests`, keeping
  the dependency tree small and adding a provider trivial ([src/llm.py](src/llm.py)).
- **Hallucination guardrails** — QA answers are grounded only in retrieved
  chunks; if nothing clears a minimum similarity threshold, the agent returns a
  "not in the paper" answer *without* calling the LLM.
- **Graceful degradation** — download failures, scanned PDFs, poor extraction,
  and LLM outages each have an explicit fallback rather than a crash.
