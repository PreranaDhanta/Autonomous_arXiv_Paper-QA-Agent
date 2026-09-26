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

### Rate limits (free tiers)

- **Groq** free tier is generous but rate-limited per minute/day (requests and
  tokens). The LLM layer retries with exponential backoff on `429`
  ([src/llm.py](src/llm.py)); if you hit a limit during testing, wait a moment
  and re-run — saved sessions mean you won't re-parse the PDF.
- **Gemini** free tier similarly caps requests per minute. Same backoff applies.
- **Ollama** is fully local with no limits (needs a running `ollama serve`).
- Model note: Groq occasionally retires model IDs; the default is
  `openai/gpt-oss-120b`. If you get a `404`, check the live list at
  https://console.groq.com/docs/models and update `GROQ_MODEL` in `.env`.

## Example run

Input — a natural-language topic:

```
$ python main.py digest "retrieval augmented generation for open domain question answering"
```

Briefing output (abridged):

```
──────────────── Selected paper ────────────────
Generation-Augmented Retrieval for Open-domain Question Answering  (2009.08553)
Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han et al. · 2020-09-17
Why: Directly proposes Generation-Augmented Retrieval for open-domain QA, matching the query focus.

Other candidates:
  - MUST-RAG: MUSical Text QA with Retrieval Augmented Generation (2507.23334)
  - An Exploration of Data Augmentation ... for Domain-Agnostic QA (1912.02145)
  - Tree of Reviews: ... Multi-hop Question Answering (2404.14464)
  - Generating Answer Candidates for Quizzes ... (2108.12898)

────────────── Executive Briefing ──────────────
# Generation-Augmented Retrieval for Open-domain Question Answering
arXiv ID: 2009.08553 · Published: 2020-09-17 · Categories: cs.CL, cs.IR
Link: http://arxiv.org/abs/2009.08553v4

## Why this paper matters
GAR expands open-domain QA queries with automatically generated contexts (answers,
surrounding sentences, titles) from a pretrained LM, letting sparse BM25 retrieval
match or surpass dense retrieval while staying computationally cheap.

## Problem statement
Sparse retrieval relies on lexical overlap and misses semantics; dense retrieval is
semantically rich but expensive and loses exact-match guarantees.

## Method / approach
- Train a seq2seq generator (BART-large) to produce relevant contexts from the question
- Append generated contexts to the original question (generation-augmented query)
- Retrieve passages with BM25 using the augmented query
- Optionally fuse BM25 + dense DPR retrieval (GAR+)
- Read with an extractive BERT reader or a generative reader

## Key results / claims
- GAR + BM25 beats plain BM25 and unsupervised query expansion (RM3)
- Matches or exceeds SOTA dense DPR retrieval on Natural Questions and TriviaQA
- End-to-end EM competitive with DPR; GAR+ fusion improves further

## Limitations
- Depends on generator quality; poor generated contexts can hurt retrieval (reviewer-inferred)
- Generation adds an extra inference step / latency vs. pure BM25 (reviewer-inferred)
- Evaluated only on two English QA benchmarks; cross-domain/language generalization untested (reviewer-inferred)

## Suggested follow-up questions
- How does GAR perform with a smaller/faster generator than BART-large?
- Can generation and retrieval be jointly trained?
- What is the impact of sampling multiple contexts vs. greedy decoding?
```

Sample QA exchanges (grounded RAG):

```
you   › What contexts does GAR generate to augment the query?
agent › GAR appends text the LM generates from its internal knowledge — e.g. the title
        of a relevant passage [chunk 0], the answer for Trivia questions or multiple
        answers/titles joined with [SEP] [chunk 2], or any task-specific snippet [chunk 5].
        grounded in 6 chunks · sections: approach, conclusion, introduction, related work

you   › Does the paper report results on languages other than English?
agent › The paper does not appear to cover this based on the retrieved text.
        (correctly refuses instead of hallucinating)

you   › How many attention heads does the model use?      # asked on 1706.03762
agent › The model uses 8 parallel attention heads (h = 8) [chunk 0], so it can attend to
        different representation subspaces at once while keeping cost comparable to a
        single head.
```

## Configuration knobs

All tunables live in `.env` (see [.env.example](.env.example)): chunk size and
overlap, QA top-k, max PDF pages, topic result count, and storage directories.

## Design decisions & tradeoffs

**Orchestration — LangGraph state graph, not a prompt chain.** Each stage is an
isolated node (`AgentState -> partial update`) with conditional edges, so the
data flow and every failure exit are visible in one place
([src/graph.py](src/graph.py)). This made the "one realistic failure case"
requirement fall out naturally — any node can set `state["error"]` and the graph
routes straight to `END`.

**State & persistence — split by size.** Embeddings live in a persistent Chroma
collection on disk; everything else (metadata, parsed sections, briefing, QA
history) is a small JSON session keyed by arXiv ID
([src/session.py](src/session.py)). A later `qa` run reloads the JSON and
reattaches to the existing collection — no re-download, no re-parse. The raw full
text is dropped from the JSON (sections are enough for QA context) to keep it small.

**LLM layer — plain HTTP, no provider SDKs.** Groq/Gemini/Ollama share one
`chat()`/`chat_json()` interface over `requests` ([src/llm.py](src/llm.py)).
Smaller dependency tree, fewer version conflicts, and adding a provider is one
method. Backoff on `429` handles free-tier limits.

**Retrieval — section-aware chunking + local embeddings.** Chunks never straddle
a section boundary and carry the section name as metadata, so QA answers can cite
*where* evidence came from. Embeddings use Chroma's built-in on-device
`all-MiniLM-L6-v2` — free, offline, no key.

**Grounding — cheap hallucination guard.** If no retrieved chunk clears a minimum
cosine similarity, QA returns "not in the paper" *without* calling the LLM
([src/nodes/qa.py](src/nodes/qa.py)). The prompt also forbids outside knowledge
and requires section citations.

**Graceful degradation.** Download failure → abstract-only briefing; scanned/empty
PDF → quality flag + abstract fallback; huge PDF → page cap; empty LLM output in
query refinement → fall back to the raw query.

### What I'd do with more time
- **Better PDF structure**: heading detection is regex-based; a layout-aware
  parser (e.g. `unstructured`) would handle two-column and figure-heavy papers better.
- **Multi-paper digests** for a topic (compare 2–3 papers), instead of picking one.
- **Reranking** the retrieved chunks (cross-encoder) before answering.
- **Streaming** QA answers and a small eval harness (grounding/faithfulness checks).
- **Unit tests** for `arxiv_client`, chunking, and the JSON extraction in `llm.py`.

### Known limitations
- Section splitting relies on common ML/CS headings; unusual templates degrade to
  full-text chunking.
- Topic selection trusts arXiv relevance + a single LLM judgment; no dedup across
  versions/preprints beyond the ID normalisation.
- Free-tier LLMs vary in quality; briefing accuracy tracks the chosen model.
