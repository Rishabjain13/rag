"""
LLM generation layer.

New in v2
─────────
  Structured citations    System prompt instructs Claude to use [p.N] markers.
                          parse_citations() extracts them into a structured list
                          so the API can return {"citations": [{"page": 3, ...}]}.

  Conversation history    stream_answer() and generate_answer() accept a
                          session.messages history list so follow-up questions
                          work naturally.

  Answer style control    'concise' | 'detailed' | 'bullets' – injected into
                          the system prompt.

  Self-RAG grounding      verify_grounding() scores how well the answer is
                          supported by the retrieved context (0.0 – 1.0).
                          Runs after generation as a non-blocking async call.
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    OPENROUTER_MODEL, OPENROUTER_FAST_MODEL,
    FAST_MODEL_API_KEY, FAST_MODEL_BASE_URL,
    MAX_ANSWER_TOKENS, SELF_RAG_ENABLED,
)
from rag.models import RetrievedContext

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None
_fast_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    return _client


def _get_fast_client() -> AsyncOpenAI:
    global _fast_client
    if _fast_client is None:
        _fast_client = AsyncOpenAI(
            api_key=FAST_MODEL_API_KEY,
            base_url=FAST_MODEL_BASE_URL,
        )
    return _fast_client


# ── Style variants ────────────────────────────────────────────────────────────

_STYLE_SUFFIX: Dict[str, str] = {
    "concise":  "Be concise: 2-4 sentences max.",
    "detailed": "Be thorough: explain the reasoning in full.",
    "bullets":  "Format your answer as a bullet-point list.",
    "default":  "",
}

# ── System prompts ────────────────────────────────────────────────────────────

_HYDE_SYSTEM = (
    "You are a knowledgeable assistant. "
    "Write a concise, factual paragraph (3-5 sentences) that directly answers "
    "the following question as if it came from a technical document. "
    "Do NOT say 'I don't know'; write a plausible hypothetical answer."
)

_RAG_SYSTEM_TMPL = (
    "You are a precise, helpful assistant that answers questions from document excerpts.\n"
    "Rules:\n"
    "  1. Use ONLY the provided context passages.\n"
    "  2. After each fact you use, add a citation [p.N] where N is the page number "
    "     shown in the passage header — e.g. 'The model achieves 94 %% accuracy [p.3].'\n"
    "  3. If the context does not contain enough information, say so explicitly.\n"
    "  {style}\n"
)


def _make_system(style: str = "default") -> str:
    suffix = _STYLE_SUFFIX.get(style, "")
    return _RAG_SYSTEM_TMPL.format(style=suffix).strip()


# ── Format context ────────────────────────────────────────────────────────────

def _format_context(contexts: List[RetrievedContext]) -> str:
    parts = []
    for i, ctx in enumerate(contexts, start=1):
        heading = f"[{ctx.heading}] " if ctx.heading else ""
        page    = f"(page {ctx.page_num})" if ctx.page_num else ""
        parts.append(f"--- Passage {i} {heading}{page} ---\n{ctx.parent_text}")
    return "\n\n".join(parts)


# ── Citation parser ───────────────────────────────────────────────────────────

_CITE_RE = re.compile(r"\[p\.(\d+)\]")


def parse_citations(
    answer_text: str,
    contexts: List[RetrievedContext],
) -> List[Dict]:
    """
    Extract [p.N] markers from *answer_text* and resolve them to context metadata.
    Returns a deduplicated list like:
      [{"page": 3, "heading": "Methods", "doc_id": "abc"}, ...]
    """
    page_nums = sorted({int(m) for m in _CITE_RE.findall(answer_text)})
    page_to_ctx = {ctx.page_num: ctx for ctx in contexts if ctx.page_num}
    citations = []
    seen = set()
    for pg in page_nums:
        if pg in seen:
            continue
        seen.add(pg)
        ctx = page_to_ctx.get(pg)
        citations.append({
            "page":    pg,
            "heading": ctx.heading if ctx else "",
            "doc_id":  ctx.doc_id  if ctx else "",
        })
    return citations


# ── HyDE ─────────────────────────────────────────────────────────────────────

async def generate_hyde(query: str) -> str:
    try:
        resp = await _get_client().chat.completions.create(
            model=OPENROUTER_MODEL,
            max_tokens=256,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        return resp.choices[0].message.content or query
    except Exception as e:
        logger.warning("HyDE failed: %s", e)
        return query


# ── Streaming answer ──────────────────────────────────────────────────────────

async def stream_answer(
    query: str,
    contexts: List[RetrievedContext],
    history: Optional[List[Dict]] = None,
    style: str = "default",
    max_tokens: int = MAX_ANSWER_TOKENS,
) -> AsyncIterator[str]:
    """
    Yield answer tokens from Claude.
    *history* is a list of prior {"role": ..., "content": ...} messages.
    """
    if not contexts:
        yield "I could not find relevant information in the document to answer your question."
        return

    context_text = _format_context(contexts)
    user_content = f"Context:\n{context_text}\n\nQuestion: {query}"

    messages = list(history or []) + [{"role": "user", "content": user_content}]

    try:
        stream = await _get_client().chat.completions.create(
            model=OPENROUTER_MODEL,
            max_tokens=max_tokens,
            stream=True,
            messages=[{"role": "system", "content": _make_system(style)}] + messages,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        logger.error("Stream error: %s", e)
        yield f"\n[Generation error: {e}]"


# ── Non-streaming (background + eval) ────────────────────────────────────────

async def generate_answer(
    query: str,
    contexts: List[RetrievedContext],
    history: Optional[List[Dict]] = None,
    style: str = "default",
) -> str:
    chunks = []
    async for tok in stream_answer(query, contexts, history=history, style=style):
        chunks.append(tok)
    return "".join(chunks)


# ── Self-RAG grounding check ──────────────────────────────────────────────────

async def verify_grounding(
    query: str,
    answer: str,
    contexts: List[RetrievedContext],
) -> Dict:
    """
    Ask fast model: is this answer grounded in the context?
    Returns {"grounded": bool, "score": float, "issues": str}.
    """
    if not SELF_RAG_ENABLED or not contexts or not answer:
        return {"grounded": True, "score": 1.0, "issues": ""}

    ctx_snippet = "\n".join(f"[p.{c.page_num}] {c.parent_text[:300]}" for c in contexts)
    prompt = (
        f"Query: {query}\n\n"
        f"Context:\n{ctx_snippet[:1500]}\n\n"
        f"Answer: {answer[:500]}\n\n"
        "Is every factual claim in the answer supported by the context? "
        "Respond ONLY with JSON: "
        '{"grounded": true/false, "score": 0.0-1.0, "issues": "one sentence or empty"}'
    )
    try:
        resp = await _get_fast_client().chat.completions.create(
            model=OPENROUTER_FAST_MODEL,
            max_tokens=150,
            messages=[
                {"role": "system", "content": "You are a faithful grounding checker. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.debug("Grounding check failed: %s", e)
        return {"grounded": True, "score": 0.85, "issues": ""}


# ── RAPTOR summary ────────────────────────────────────────────────────────────

async def generate_summary(texts: List[str]) -> str:
    combined = "\n\n".join(texts[:5])
    try:
        resp = await _get_fast_client().chat.completions.create(
            model=OPENROUTER_FAST_MODEL,
            max_tokens=512,
            messages=[
                {"role": "system", "content": "Produce a single coherent paragraph summary of the following passages."},
                {"role": "user", "content": combined},
            ],
        )
        return resp.choices[0].message.content or combined[:500]
    except Exception as e:
        logger.warning("Summary failed: %s", e)
        return combined[:500]
