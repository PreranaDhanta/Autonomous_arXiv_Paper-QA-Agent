"""Node 7 — QA Loop (RAG).

Two nodes form a single QA turn:
  * retrieve_chunks  — cosine similarity search over the paper's chunks
  * generate_answer  — answer grounded ONLY in retrieved chunks, with citations

Grounding guardrails:
  * The prompt instructs the model to answer solely from the provided context and
    to explicitly say the paper doesn't cover it otherwise.
  * If no chunks clear a minimum similarity threshold, we short-circuit and return
    a "not in the paper" answer without calling the LLM — cheap and hallucination-proof.
"""
from __future__ import annotations

from ..config import config
from ..llm import LLMError, get_llm
from ..state import AgentState
from ..vector_store import get_store

# Cosine similarity below this => treat as "no real evidence in the paper".
_MIN_SIMILARITY = 0.15

_QA_SYSTEM = """You answer questions about a single research paper using ONLY the \
provided context chunks.

Rules:
- Use only the context below. Do not use outside knowledge.
- If the answer is not contained in the context, reply exactly: \
"The paper does not appear to cover this based on the retrieved text."
- Be concise and specific. Cite the section(s) you used in brackets, e.g. [method]."""


def retrieve_chunks(state: AgentState) -> AgentState:
    collection = state.get("collection_name")
    question = state.get("question", "")
    if not collection or not get_store().exists(collection):
        return {"retrieved": []}
    try:
        hits = get_store().query(collection, question, top_k=config.qa_top_k)
    except Exception:
        hits = []
    return {"retrieved": hits}


def generate_answer(state: AgentState) -> AgentState:
    question = state.get("question", "")
    hits = state.get("retrieved", [])
    conversation = list(state.get("conversation", []))

    strong = [h for h in hits if h.get("similarity", 0) >= _MIN_SIMILARITY]
    if not strong:
        answer = "The paper does not appear to cover this based on the retrieved text."
        conversation += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {"answer": answer, "conversation": conversation}

    context = "\n\n".join(
        f"[chunk {i} | section: {h['section']} | similarity: {h['similarity']:.2f}]\n{h['text']}"
        for i, h in enumerate(strong)
    )
    # Include brief recent history so follow-ups keep context.
    history = ""
    if conversation:
        recent = conversation[-4:]
        history = "Recent conversation:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in recent
        ) + "\n\n"

    user = f"{history}Context chunks from the paper:\n{context}\n\nQuestion: {question}"
    try:
        answer = get_llm().chat(_QA_SYSTEM, user, temperature=0.1, max_tokens=700)
    except LLMError as exc:
        answer = f"(LLM error while answering: {exc})"

    conversation += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return {"answer": answer, "conversation": conversation}
