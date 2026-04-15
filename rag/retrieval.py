"""
Retrieval layer.

New in v2
─────────
  Multi-query expansion   Generate 3 query variants via fast model; run all through
                          hybrid search in parallel; mega-RRF the combined pool.
                          Improves recall on ambiguous / under-specified queries.

  MMR                     After parent expansion, reorder contexts by Maximal
                          Marginal Relevance (relevance × diversity) so the LLM
                          sees 4 distinct angles, not 4 near-duplicate passages.

  Contextual compression  After parent expansion, use fast model to extract only the
                          sentences relevant to the query from each 1024-tok parent.
                          Cuts LLM context size ~60 % and reduces hallucination.

  Retrieval tracing       Every call returns a RetrievalTrace so the API can emit
                          diagnostic SSE events (which indexes fired, hit counts, …).

  Per-doc filter          All search methods accept optional doc_id to scope results
                          to a single document in multi-doc mode.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    TOP_K_SEARCH, TOP_K_RERANK, TOP_K_PARENTS, RRF_K,
    MULTI_QUERY_ENABLED, MULTI_QUERY_COUNT, MULTI_QUERY_MIN_HITS,
    MMR_ENABLED, MMR_LAMBDA,
    COMPRESSION_ENABLED,
    FAST_MODEL_API_KEY, FAST_MODEL_BASE_URL, OPENROUTER_FAST_MODEL,
)
from rag.models import (
    QueryRoute, RetrievedContext, RetrievalTrace, SearchHit,
)
from rag.embeddings import embed_query, embed_texts
from rag.stores import RAGStores
from rag.cache import SemanticCache

logger = logging.getLogger(__name__)

# ── Lazy fast client (cheap LLM for expansion + compression) ──────────────────
_fast_client = None


def _get_fast_client():
    global _fast_client
    if _fast_client is None:
        from openai import AsyncOpenAI
        _fast_client = AsyncOpenAI(
            api_key=FAST_MODEL_API_KEY,
            base_url=FAST_MODEL_BASE_URL,
        )
    return _fast_client


# ── Adaptive Router ───────────────────────────────────────────────────────────

_RAPTOR_PAT = re.compile(
    r"\b(summari[sz]e|overview|main (idea|point|topic)|entire|whole|throughout|"
    r"overall|abstract|introduction|conclusion|all chapters?|full document)\b",
    re.IGNORECASE,
)
_HYDE_PAT = re.compile(
    r"\b(explain|how does|why (does|is|are|do)|compare|contrast|analys[iz]e|"
    r"implication|relationship|difference|similar|unlike|advantage|disadvantage)\b",
    re.IGNORECASE,
)


async def route_query(
    query: str,
    cache: SemanticCache,
    raptor_ready: bool,
    doc_id: Optional[str] = None,
) -> Tuple[QueryRoute, Optional[str]]:
    cached = await cache.lookup(query, doc_id=doc_id)
    if cached:
        return QueryRoute.CACHED, cached
    if raptor_ready and _RAPTOR_PAT.search(query):
        return QueryRoute.RAPTOR, None
    if _HYDE_PAT.search(query):
        return QueryRoute.HYDE, None
    return QueryRoute.DIRECT, None


# ── Multi-query expansion ─────────────────────────────────────────────────────

async def expand_query(query: str) -> List[str]:
    """Return original query + up to MULTI_QUERY_COUNT LLM-generated variants."""
    if not MULTI_QUERY_ENABLED:
        return [query]
    try:
        resp = await _get_fast_client().chat.completions.create(
            model=OPENROUTER_FAST_MODEL,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Generate {MULTI_QUERY_COUNT} alternative search queries that "
                        "retrieve the same information as the input. "
                        "Return one query per line, no numbering, no explanations."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        # Reasoning models leak chain-of-thought before actual queries.
        # Real search queries are short; CoT is long. Filter on length
        # and skip lines that start with CoT indicators.
        _COT = ("okay", "first,", "let me", "i need", "we need", "so,",
                "i'll", "to answer", "note:", "step ", "the user", "this is")
        variants = [
            q.strip()
            for q in (resp.choices[0].message.content or "").strip().splitlines()
            if q.strip()
            and q.strip().lower() != query.lower()
            and len(q.strip()) <= 150
            and not any(q.strip().lower().startswith(p) for p in _COT)
        ][:MULTI_QUERY_COUNT]
        all_queries = [query] + variants
        logger.info("Query expansion: %d variants", len(all_queries))
        return all_queries
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)
        return [query]


# ── RRF Fusion ────────────────────────────────────────────────────────────────

def rrf_fusion(ranked_lists: List[List[SearchHit]], k: int = RRF_K) -> List[SearchHit]:
    scores: Dict[str, float]    = {}
    best:   Dict[str, SearchHit] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            cid = hit.child_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in best or hit.score > best[cid].score:
                best[cid] = hit

    merged = sorted(best.values(), key=lambda h: scores[h.child_id], reverse=True)
    for hit in merged:
        hit.score = scores[hit.child_id]
    return merged


# ── Triple Hybrid Search ──────────────────────────────────────────────────────

async def _dense_search(vec, stores, k, doc_id):
    return await stores.qdrant.search_children(vec, k=k, doc_id=doc_id)


async def _colbert_search(query, stores, k):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, stores.colbert.search, query, k)


async def _bm25_search(query, stores, k, doc_id):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, stores.bm25.search, query, k, doc_id)


def _fill_parent_ids(colbert_hits: List[SearchHit], dense_hits: List[SearchHit]):
    t2p = {h.text: h.parent_id for h in dense_hits}
    for h in colbert_hits:
        if not h.parent_id:
            h.parent_id = t2p.get(h.text, "")


async def hybrid_search(
    query: str,
    query_vec: List[float],
    stores: RAGStores,
    k: int = TOP_K_SEARCH,
    doc_id: Optional[str] = None,
) -> Tuple[List[SearchHit], int, int, int]:
    """Returns (fused_hits, n_dense, n_colbert, n_bm25)."""
    dense, colbert, bm25 = await asyncio.gather(
        _dense_search(query_vec, stores, k, doc_id),
        _colbert_search(query, stores, k),
        _bm25_search(query, stores, k, doc_id),
    )
    _fill_parent_ids(colbert, dense)
    fused = rrf_fusion([dense, colbert, bm25])[:k]
    return fused, len(dense), len(colbert), len(bm25)


# ── Rerank ────────────────────────────────────────────────────────────────────

async def rerank(query: str, candidates: List[SearchHit],
                 stores: RAGStores, k: int = TOP_K_RERANK) -> List[SearchHit]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, stores.colbert.rerank, query, candidates, k)


# ── MMR ───────────────────────────────────────────────────────────────────────

async def apply_mmr(
    query_vec: List[float],
    contexts: List[RetrievedContext],
    k: int = TOP_K_PARENTS,
    lam: float = MMR_LAMBDA,
) -> List[RetrievedContext]:
    """
    Maximal Marginal Relevance over parent contexts.
    score_i = lam * sim(query, ctx_i) - (1-lam) * max_j_in_selected sim(ctx_i, ctx_j)
    """
    if len(contexts) <= k or not MMR_ENABLED:
        return contexts[:k]

    texts = [c.parent_text for c in contexts]
    embs = np.array(await embed_texts(texts), dtype=np.float32)
    q    = np.array(query_vec, dtype=np.float32)

    # Normalise
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    q_n    = q    / (np.linalg.norm(q) + 1e-8)

    selected: List[int] = []
    remaining = list(range(len(contexts)))

    while len(selected) < k and remaining:
        best_idx, best_score = None, float("-inf")
        for i in remaining:
            rel = float(q_n @ embs_n[i])
            div = max((float(embs_n[i] @ embs_n[j]) for j in selected), default=0.0)
            score = lam * rel - (1 - lam) * div
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [contexts[i] for i in selected]


# ── Contextual compression ────────────────────────────────────────────────────

# Prefixes that indicate a reasoning model's chain-of-thought preamble.
_COT_STARTS = (
    "okay", "alright", "let me", "i need", "we need", "first,", "so,",
    "i'll", "to answer", "the question is", "to extract", "the user",
    "i should", "looking at", "reading", "analyzing", "note:", "step ",
    "i am", "i will", "my task", "let's", "let us", "here is", "here are",
    "in this", "the text", "the passage", "from the",
)


def _strip_cot(text: str) -> str:
    """
    Strip chain-of-thought preamble that reasoning models emit before actual output.
    Strategy: split on blank lines; if the first block looks like CoT, drop it.
    """
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        first_lower = parts[0].lower()
        if any(first_lower.startswith(p) for p in _COT_STARTS):
            return "\n\n".join(parts[1:]).strip()
    return text


async def compress_context(query: str, parent_text: str) -> str:
    """
    Ask fast model to keep only the sentences relevant to *query*.
    Falls back to the original if compression is too aggressive or fails.
    """
    if not COMPRESSION_ENABLED:
        return parent_text
    try:
        resp = await _get_fast_client().chat.completions.create(
            model=OPENROUTER_FAST_MODEL,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract ONLY the sentences from the text that directly help answer "
                        "the question. Return verbatim excerpts – do not paraphrase. "
                        "If the entire text is relevant, return it unchanged."
                    ),
                },
                {"role": "user", "content": f"Question: {query}\n\nText:\n{parent_text}"},
            ],
        )
        compressed = _strip_cot((resp.choices[0].message.content or "").strip())
        # Reject if too short (over-compressed) or longer than original
        if len(compressed) < 80 or len(compressed) >= len(parent_text):
            return parent_text
        return compressed
    except Exception as e:
        logger.debug("Compression failed: %s", e)
        return parent_text


async def compress_all(query: str, contexts: List[RetrievedContext]) -> Tuple[List[RetrievedContext], bool]:
    """Compress all contexts concurrently. Returns (contexts, compression_applied)."""
    if not COMPRESSION_ENABLED:
        return contexts, False
    tasks = [compress_context(query, c.parent_text) for c in contexts]
    compressed_texts = await asyncio.gather(*tasks)
    applied = any(t != c.parent_text for t, c in zip(compressed_texts, contexts))
    for ctx, text in zip(contexts, compressed_texts):
        ctx.parent_text = text
    return contexts, applied


# ── Parent Expansion ──────────────────────────────────────────────────────────

def expand_to_parents(
    hits: List[SearchHit],
    stores: RAGStores,
    max_parents: int = TOP_K_PARENTS,
) -> List[RetrievedContext]:
    seen:      Dict[str, float]       = {}
    p_children: Dict[str, List[str]]  = {}

    for hit in hits:
        pid = hit.parent_id
        if not pid:
            continue
        if pid not in seen or hit.score > seen[pid]:
            seen[pid] = hit.score
        p_children.setdefault(pid, []).append(hit.child_id)

    sorted_pids = sorted(seen, key=lambda p: seen[p], reverse=True)
    contexts: List[RetrievedContext] = []

    for pid in sorted_pids[:max_parents]:
        parent = stores.parents.get(pid)
        if parent is None:
            continue
        contexts.append(RetrievedContext(
            parent_id=pid,
            parent_text=parent.text,
            page_num=parent.page_num,
            heading=parent.heading,
            doc_id=parent.doc_id,
            rrf_score=seen[pid],
            child_ids=p_children[pid],
        ))

    logger.info("Parent expansion: %d hits → %d contexts", len(hits), len(contexts))
    return contexts


# ── RAPTOR ────────────────────────────────────────────────────────────────────

async def raptor_retrieve(
    query: str, stores: RAGStores, k: int = 3, doc_id: Optional[str] = None,
) -> List[str]:
    q_vec = await embed_query(query)
    return await stores.qdrant.search_raptor(q_vec, k=k, doc_id=doc_id)


# ── Full pipeline ─────────────────────────────────────────────────────────────

async def retrieve(
    query: str,
    stores: RAGStores,
    cache: SemanticCache,
    raptor_ready: bool,
    doc_id: Optional[str] = None,
    hyde_fn=None,
) -> Tuple[QueryRoute, List[RetrievedContext], Optional[str], RetrievalTrace]:
    """
    Main retrieval entry.

    Returns (route, contexts, cached_answer, trace).
    """
    trace = RetrievalTrace(route="")

    route, cached = await route_query(query, cache, raptor_ready, doc_id)
    trace.route = route.value

    # ── Cached ────────────────────────────────────────────────────────────────
    if route == QueryRoute.CACHED:
        return route, [], cached, trace

    # ── RAPTOR ────────────────────────────────────────────────────────────────
    if route == QueryRoute.RAPTOR:
        summaries = await raptor_retrieve(query, stores, doc_id=doc_id)
        contexts = [
            RetrievedContext(
                parent_id=f"raptor_{i}", parent_text=s,
                page_num=0, heading="RAPTOR Summary", doc_id=doc_id or "",
                rrf_score=1.0,
            )
            for i, s in enumerate(summaries)
        ]
        trace.parents_returned = len(contexts)
        return route, contexts, None, trace

    # ── Embed query (HyDE uses a hypothetical answer as the query vector) ────────
    if route == QueryRoute.HYDE and hyde_fn is not None:
        hyp = await hyde_fn(query)
        q_vec = await embed_query(hyp)
        logger.info("HyDE: %s…", hyp[:80])
    else:
        q_vec = await embed_query(query)

    # ── First-pass search (original query, full k) ────────────────────────────
    first_hits, nd, nc, nb = await hybrid_search(query, q_vec, stores, TOP_K_SEARCH, doc_id)
    all_ranked: List[List[SearchHit]] = [first_hits]
    total_dense, total_colbert, total_bm25 = nd, nc, nb
    trace.queries = [query]

    # ── Conditional multi-query expansion ─────────────────────────────────────
    # Only call the LLM for expansion when the first pass is sparse.
    # Good first-pass results (dense+bm25 >= MULTI_QUERY_MIN_HITS) don't need it.
    first_pass_hits = nd + nb
    if MULTI_QUERY_ENABLED and first_pass_hits < MULTI_QUERY_MIN_HITS:
        logger.info("First pass sparse (%d hits) — expanding query", first_pass_hits)
        variants = [q for q in await expand_query(query) if q.lower() != query.lower()]
        if variants:
            trace.queries = [query] + variants
            # Embed all variant queries in parallel, then launch hybrid searches in parallel
            variant_vecs = await asyncio.gather(*[embed_query(q) for q in variants])
            extra_tasks = [
                hybrid_search(q, vec, stores, TOP_K_SEARCH, doc_id)
                for q, vec in zip(variants, variant_vecs)
            ]
            for hits, nd2, nc2, nb2 in await asyncio.gather(*extra_tasks):
                all_ranked.append(hits)
                total_dense   += nd2
                total_colbert += nc2
                total_bm25    += nb2

    fused = rrf_fusion(all_ranked)[:TOP_K_SEARCH]

    # ── Zero-hit fallback: individual BM25 term search ────────────────────────
    # Fires when ALL three indexes return nothing — typically caused by vocabulary
    # mismatch (query uses different phrasing than the document).
    if not fused:
        logger.info("Zero fused hits — trying BM25 term fallback")
        fallback = stores.bm25.search_fallback(query, TOP_K_SEARCH, doc_id)
        if fallback:
            fused = fallback
            total_bm25 += len(fallback)
            logger.info("BM25 term fallback: %d hits", len(fallback))

    trace.dense_hits   = total_dense
    trace.colbert_hits = total_colbert
    trace.bm25_hits    = total_bm25
    trace.fused_hits   = len(fused)

    # ── Rerank ────────────────────────────────────────────────────────────────
    reranked = await rerank(query, fused, stores, k=TOP_K_RERANK)
    trace.reranked_hits = len(reranked)

    # ── Parent expansion ──────────────────────────────────────────────────────
    contexts = expand_to_parents(reranked, stores)

    # ── MMR ───────────────────────────────────────────────────────────────────
    if MMR_ENABLED and len(contexts) > 1:
        contexts = await apply_mmr(q_vec, contexts)
        trace.mmr_applied = True

    # ── Contextual compression ────────────────────────────────────────────────
    if COMPRESSION_ENABLED and contexts:
        contexts, applied = await compress_all(query, contexts)
        trace.compression_applied = applied

    trace.parents_returned = len(contexts)
    return route, contexts, None, trace
