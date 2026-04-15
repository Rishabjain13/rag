"""Semantic cache with per-doc filtering support."""
from __future__ import annotations

import logging
from typing import Optional

from config import CACHE_THRESHOLD
from rag.embeddings import embed_query, embed_texts
from rag.stores import QdrantStore

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self, qdrant: QdrantStore):
        self._qdrant = qdrant

    async def lookup(self, query: str, doc_id: Optional[str] = None) -> Optional[str]:
        """Return cached answer if query is similar enough, else None."""
        try:
            q_vec = await embed_query(query)
            return await self._qdrant.search_cache(q_vec, threshold=CACHE_THRESHOLD, doc_id=doc_id)
        except Exception as e:
            logger.warning("Cache lookup error: %s", e)
            return None

    async def store(self, entry_id: str, question: str, answer: str, doc_id: str):
        try:
            vecs = await embed_texts([question])
            if vecs:
                await self._qdrant.add_cache_entry(
                    entry_id=entry_id, question=question,
                    answer=answer, doc_id=doc_id, embedding=vecs[0],
                )
        except Exception as e:
            logger.warning("Cache store error: %s", e)
