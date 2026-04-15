"""
Non-blocking background upgrades that run after ingestion.

  +2 min  QA Cache builder
  ─────────────────────────
  1. Extract unique headings from all ingested parents
  2. For each heading, ask Claude to generate 3 likely questions
  3. Answer each question via the full RAG pipeline
  4. Embed each question and store in Qdrant cache collection
  → Common / repeat queries now return in ~50 ms

  +5 min  RAPTOR Tree builder
  ─────────────────────────────
  Layer 0 : all child chunks  (leaf nodes)
  Layer 1 : cluster summaries (KMeans, cluster_size ≈ 5)
  Layer 2 : section summaries (cluster the layer-1 summaries)
  ...
  Layer N : single document root summary
  → Broad / summary queries get full-document context
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

import numpy as np
from sklearn.cluster import KMeans

# Limit concurrent LLM calls to the fast model (Groq free tier: ~30 req/min).
# Created lazily inside the event loop to avoid "no running loop" on import.
_LLM_SEM: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _LLM_SEM
    if _LLM_SEM is None:
        _LLM_SEM = asyncio.Semaphore(5)
    return _LLM_SEM

from config import RAPTOR_CLUSTER_SIZE, RAPTOR_MAX_LEVELS
from rag.models import ParentChunk, ChildChunk, RaptorNode, CachedQA
from rag.embeddings import embed_texts
from rag.generation import generate_summary, generate_answer
from rag.stores import RAGStores
from rag.cache import SemanticCache

logger = logging.getLogger(__name__)


# ── QA Cache ──────────────────────────────────────────────────────────────────

async def _generate_questions_for_heading(heading: str, llm_client) -> List[str]:
    """Ask the fast model for 3 plausible questions about a heading."""
    from config import OPENROUTER_FAST_MODEL
    try:
        async with _get_sem():
            response = await llm_client.chat.completions.create(
                model=OPENROUTER_FAST_MODEL,
                max_tokens=200,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate exactly 3 concise questions that a reader might ask about the "
                            "following document section heading. Return one question per line, "
                            "no numbering, no extra text."
                        ),
                    },
                    {"role": "user", "content": heading},
                ],
            )
        text = response.choices[0].message.content or ""
        questions = [
            q.strip() for q in text.strip().splitlines()
            if q.strip().endswith("?") and len(q.strip()) < 200
        ]
        return questions[:3]
    except Exception as e:
        logger.warning("Question generation failed for '%s': %s", heading, e)
        return [f"What is {heading}?"]


async def build_qa_cache(
    parents: List[ParentChunk],
    stores: RAGStores,
    cache: SemanticCache,
    retrieve_fn,   # async (query) → List[RetrievedContext]
):
    """
    Build the semantic QA cache.
    Runs in the background; errors are swallowed so they don't block the server.
    """
    from openai import AsyncOpenAI
    from config import FAST_MODEL_API_KEY, FAST_MODEL_BASE_URL
    llm = AsyncOpenAI(api_key=FAST_MODEL_API_KEY, base_url=FAST_MODEL_BASE_URL)

    # Collect unique headings (non-empty)
    headings = list({p.heading for p in parents if p.heading.strip()})[:20]
    if not headings:
        logger.info("QA Cache: no headings found – skipping")
        return

    logger.info("QA Cache: generating questions for %d headings", len(headings))
    total = 0

    for heading in headings:
        questions = await _generate_questions_for_heading(heading, llm)
        for question in questions:
            try:
                contexts = await retrieve_fn(question)
                if not contexts:
                    continue
                async with _get_sem():
                    answer = await generate_answer(question, contexts)
                qa = CachedQA.create(question=question, answer=answer, doc_id=parents[0].doc_id)
                await cache.store(qa.id, qa.question, qa.answer, qa.doc_id)
                total += 1
            except Exception as e:
                logger.warning("QA Cache: failed for '%s': %s", question, e)

    logger.info("QA Cache: stored %d pre-computed answers", total)


# ── RAPTOR Tree ───────────────────────────────────────────────────────────────

async def _summarise_cluster(cluster_texts: List[str], doc_id: str, level: int) -> RaptorNode:
    """Summarise one cluster under a concurrency semaphore."""
    async with _get_sem():
        summary = await generate_summary(cluster_texts)
    return RaptorNode.create(doc_id=doc_id, text=summary, level=level, child_ids=[])


async def _cluster_and_summarise(
    texts: List[str],
    embeddings: List[List[float]],
    doc_id: str,
    level: int,
) -> List[RaptorNode]:
    """
    Cluster *texts* via KMeans, summarise each cluster with the fast model.
    All cluster summaries at a given level run in parallel (capped by _LLM_SEM).
    """
    n = len(texts)
    if n <= 1:
        summary = texts[0] if texts else ""
        return [RaptorNode.create(doc_id=doc_id, text=summary, level=level, child_ids=[])]

    n_clusters = max(1, min(n // RAPTOR_CLUSTER_SIZE, 20))
    emb_arr = np.array(embeddings, dtype=np.float32)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(emb_arr)

    tasks = []
    for cluster_id in range(n_clusters):
        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        tasks.append(_summarise_cluster([texts[i] for i in indices], doc_id, level))

    return list(await asyncio.gather(*tasks))


async def build_raptor_tree(
    children: List[ChildChunk],
    doc_id: str,
    stores: RAGStores,
):
    """
    Recursively build RAPTOR layers and store all nodes in Qdrant.

    Layer 0 : leaf nodes = child chunk texts
    Layer 1 : cluster summaries of layer-0
    ...
    Layer N : single document root
    """
    if not children:
        logger.info("RAPTOR: no children – skipping")
        return

    logger.info("RAPTOR: building tree from %d children", len(children))
    all_nodes: List[RaptorNode] = []

    # ── Layer 0: embed children ──────────────────────────────────────────────
    texts_0 = [c.text for c in children]
    embs_0 = [c.embedding for c in children if c.embedding]
    if len(embs_0) != len(texts_0):
        # Re-embed if needed
        embs_0 = await embed_texts(texts_0)

    current_texts = texts_0
    current_embs = embs_0

    for level in range(1, RAPTOR_MAX_LEVELS + 1):
        nodes = await _cluster_and_summarise(current_texts, current_embs, doc_id, level)
        all_nodes.extend(nodes)
        logger.info("RAPTOR level %d: %d nodes", level, len(nodes))

        if len(nodes) <= 1:
            break

        # Embed summaries for next level
        node_texts = [n.text for n in nodes]
        node_embs = await embed_texts(node_texts)
        for node, emb in zip(nodes, node_embs):
            node.embedding = emb

        current_texts = node_texts
        current_embs = node_embs

    # Determine dim from first node that has an embedding
    dim = next((len(n.embedding) for n in all_nodes if n.embedding), None)
    if dim:
        await stores.qdrant.add_raptor_nodes(all_nodes, dim)
    logger.info("RAPTOR: tree complete – %d total nodes stored", len(all_nodes))


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_background_jobs(
    parents: List[ParentChunk],
    children: List[ChildChunk],
    doc_id: str,
    stores: RAGStores,
    cache: SemanticCache,
    retrieve_fn,               # async (query) → List[RetrievedContext]
    raptor_ready_flag: dict,   # {"ready": bool} – mutated when RAPTOR done
):
    """
    Runs both background jobs sequentially (low priority).
    Designed to be launched with asyncio.create_task().
    """
    # +2 min: QA cache
    await asyncio.sleep(120)
    logger.info("Background: starting QA cache build")
    try:
        await build_qa_cache(parents, stores, cache, retrieve_fn)
    except Exception as e:
        logger.error("QA cache build failed: %s", e)

    # +3 min: RAPTOR tree
    await asyncio.sleep(180)
    logger.info("Background: starting RAPTOR tree build")
    try:
        await build_raptor_tree(children, doc_id, stores)
        raptor_ready_flag["ready"] = True
        logger.info("Background: RAPTOR tree ready")
    except Exception as e:
        logger.error("RAPTOR tree build failed: %s", e)
