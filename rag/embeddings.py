"""
Embedding layer — sentence-transformers (local, no external API).

Model is lazy-loaded on first use (thread-safe singleton).
All encode() calls run in a thread-pool executor so the asyncio event loop
is never blocked. Internal batching at batch_size=64 for throughput.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import List

from config import EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


# ── Lazy-loaded local model (thread-safe singleton) ───────────────────────────

_st_model      = None
_st_model_lock = threading.Lock()


def _local_model():
    global _st_model
    if _st_model is not None:        # fast path — no lock needed once loaded
        return _st_model
    with _st_model_lock:             # only one thread loads
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model: %s", EMBEDDING_MODEL)
            _st_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _st_model


# ── Sync workers (run inside thread-pool executor) ────────────────────────────

def _embed_local(texts: List[str]) -> List[List[float]]:
    vecs = _local_model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    return vecs.tolist()


def _embed_query_local(query: str) -> List[float]:
    model = _local_model()
    try:
        vec = model.encode(
            [query],
            normalize_embeddings=True,
            prompt_name="query",
            show_progress_bar=False,
        )
    except TypeError:
        vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return vec[0].tolist()


# ── Public async interface ────────────────────────────────────────────────────

async def embed_children(parent_text: str, child_texts: List[str]) -> List[List[float]]:
    """Embed child chunks. parent_text kept for API compatibility."""
    if not child_texts:
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_local, child_texts)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed arbitrary texts (RAPTOR, QA cache, etc.)."""
    if not texts:
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_local, texts)


async def embed_query(query: str) -> List[float]:
    """Embed a search query."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_query_local, query)


def get_embedding_dim() -> int:
    try:
        return _local_model().get_sentence_embedding_dimension() or EMBEDDING_DIM
    except Exception:
        return EMBEDDING_DIM
