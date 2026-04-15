"""
Three search indexes + parent store + document registry.

Improvements over v1
────────────────────
  DocRegistry      tracks every ingested document; enables multi-doc queries
  Per-doc filter   Qdrant Filter(must=[doc_id==X]) on all searches
  BM25 multi-doc   stores doc_id per entry; filters post-scoring
  Incremental BM25 rebuilds index from all children on each new ingest
                   (simple & correct; fast enough up to ~100k chunks)
"""
from __future__ import annotations

import asyncio
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import (
    QDRANT_URL, QDRANT_COLLECTION,
    QDRANT_CACHE_COLLECTION, QDRANT_RAPTOR_COLLECTION,
    COLBERT_MODEL, COLBERT_INDEX_PATH,
    TOP_K_SEARCH, EMBEDDING_DIM,
    PERSIST_DIR,
)
from rag.models import ChildChunk, DocRecord, ParentChunk, SearchHit

logger = logging.getLogger(__name__)


# ── ColBERT availability ──────────────────────────────────────────────────────

COLBERT_AVAILABLE = False
try:
    from ragatouille import RAGPretrainedModel as _RagModel  # noqa: F401
    COLBERT_AVAILABLE = True
    logger.info("RAGatouille ColBERT available")
except Exception:
    logger.warning("RAGatouille unavailable – cross-encoder rerank fallback active")

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ── Qdrant store ──────────────────────────────────────────────────────────────

class QdrantStore:
    def __init__(self):
        from qdrant_client import AsyncQdrantClient
        if QDRANT_URL:
            self.client = AsyncQdrantClient(url=QDRANT_URL)
            logger.info("Qdrant → %s (async)", QDRANT_URL)
        else:
            self.client = AsyncQdrantClient(":memory:")
            logger.info("Qdrant in-memory (async)")

    async def _ensure_collection(self, name: str, dim: int):
        from qdrant_client.models import Distance, VectorParams
        existing = {c.name for c in (await self.client.get_collections()).collections}
        if name not in existing:
            await self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def add_children(self, children: List[ChildChunk], dim: int):
        from qdrant_client.models import PointStruct
        await self._ensure_collection(QDRANT_COLLECTION, dim)
        points = [
            PointStruct(
                id=c.id,
                vector=c.embedding,
                payload={
                    "parent_id": c.parent_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "page_num": c.page_num,
                    "index_in_parent": c.index_in_parent,
                },
            )
            for c in children if c.embedding is not None
        ]
        if points:
            await self.client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            logger.info("Qdrant: upserted %d child vectors", len(points))

    async def search_children(
        self,
        query_vec: List[float],
        k: int = TOP_K_SEARCH,
        doc_id: Optional[str] = None,
    ) -> List[SearchHit]:
        query_filter = None
        if doc_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        results = await self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vec,
            limit=k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [
            SearchHit(
                child_id=r.id,
                parent_id=r.payload["parent_id"],
                text=r.payload["text"],
                score=r.score,
                source="dense",
            )
            for r in results.points
        ]

    # ── Cache ──────────────────────────────────────────────────────────────────

    async def add_cache_entry(self, entry_id: str, question: str, answer: str,
                              doc_id: str, embedding: List[float]):
        from qdrant_client.models import PointStruct
        await self._ensure_collection(QDRANT_CACHE_COLLECTION, len(embedding))
        await self.client.upsert(
            collection_name=QDRANT_CACHE_COLLECTION,
            points=[PointStruct(
                id=entry_id, vector=embedding,
                payload={"question": question, "answer": answer, "doc_id": doc_id},
            )],
        )

    async def search_cache(
        self,
        query_vec: List[float],
        threshold: float,
        doc_id: Optional[str] = None,
    ) -> Optional[str]:
        try:
            query_filter = None
            if doc_id:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                query_filter = Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            results = await self.client.query_points(
                collection_name=QDRANT_CACHE_COLLECTION,
                query=query_vec,
                limit=1,
                with_payload=True,
                score_threshold=threshold,
                query_filter=query_filter,
            )
            if results.points:
                return results.points[0].payload["answer"]
        except Exception:
            pass
        return None

    # ── RAPTOR ────────────────────────────────────────────────────────────────

    async def add_raptor_nodes(self, nodes: list, dim: int):
        from qdrant_client.models import PointStruct
        await self._ensure_collection(QDRANT_RAPTOR_COLLECTION, dim)
        points = [
            PointStruct(
                id=n.id, vector=n.embedding,
                payload={"text": n.text, "level": n.level, "doc_id": n.doc_id},
            )
            for n in nodes if n.embedding is not None
        ]
        if points:
            await self.client.upsert(collection_name=QDRANT_RAPTOR_COLLECTION, points=points)
            logger.info("RAPTOR: upserted %d nodes", len(points))

    async def search_raptor(
        self,
        query_vec: List[float],
        k: int = 3,
        doc_id: Optional[str] = None,
    ) -> List[str]:
        try:
            query_filter = None
            if doc_id:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                query_filter = Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            results = await self.client.query_points(
                collection_name=QDRANT_RAPTOR_COLLECTION,
                query=query_vec,
                limit=k,
                with_payload=True,
                query_filter=query_filter,
            )
            return [r.payload["text"] for r in results.points]
        except Exception:
            return []

    async def clear_all(self):
        for name in [QDRANT_COLLECTION, QDRANT_CACHE_COLLECTION, QDRANT_RAPTOR_COLLECTION]:
            try:
                await self.client.delete_collection(name)
            except Exception:
                pass


# ── BM25 store ────────────────────────────────────────────────────────────────

class BM25Store:
    def __init__(self):
        self._index     = None
        self._id_map:     List[str] = []
        self._parent_map: List[str] = []
        self._doc_map:    List[str] = []   # NEW: position → doc_id
        self._text_map:   List[str] = []

    def _rebuild(self, children: List[ChildChunk]):
        from rank_bm25 import BM25Okapi
        corpus = [c.text.lower().split() for c in children]
        self._id_map     = [c.id         for c in children]
        self._parent_map = [c.parent_id  for c in children]
        self._doc_map    = [c.doc_id     for c in children]
        self._text_map   = [c.text       for c in children]
        self._index = BM25Okapi(corpus)

    def rebuild(self, all_children: List[ChildChunk]):
        """Rebuild index from scratch (call after every new ingest)."""
        self._rebuild(all_children)
        logger.info("BM25: index rebuilt with %d documents", len(all_children))

    def search(
        self,
        query: str,
        k: int = TOP_K_SEARCH,
        doc_id: Optional[str] = None,
    ) -> List[SearchHit]:
        if self._index is None:
            return []
        scores = self._index.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1]

        hits: List[SearchHit] = []
        for idx in top_indices:
            if len(hits) >= k:
                break
            if scores[idx] <= 0:
                break
            if doc_id and self._doc_map[idx] != doc_id:
                continue
            hits.append(SearchHit(
                child_id=self._id_map[idx],
                parent_id=self._parent_map[idx],
                text=self._text_map[idx],
                score=float(scores[idx]),
                source="bm25",
            ))
        return hits


# ── ColBERT store ─────────────────────────────────────────────────────────────

class ColBERTStore:
    def __init__(self):
        self._rag      = None
        self._fallback = False

    def build_index(self, children: List[ChildChunk]):
        if not COLBERT_AVAILABLE:
            self._fallback = True
            return
        try:
            from ragatouille import RAGPretrainedModel
            os.makedirs(COLBERT_INDEX_PATH, exist_ok=True)
            rag = RAGPretrainedModel.from_pretrained(COLBERT_MODEL)
            rag.index(
                collection=[c.text for c in children],
                document_ids=[c.id for c in children],
                index_name="rag_index",
                max_document_length=256,
                split_documents=False,
                overwrite_index=True,
            )
            self._rag = rag
            logger.info("ColBERT index built (%d docs)", len(children))
        except Exception as e:
            logger.warning("ColBERT index failed: %s – cross-encoder fallback", e)
            self._fallback = True

    def search(self, query: str, k: int = TOP_K_SEARCH) -> List[SearchHit]:
        if self._rag is None or self._fallback:
            return []
        try:
            return [
                SearchHit(
                    child_id=r.get("document_id", ""),
                    parent_id="",
                    text=r["content"],
                    score=r["score"],
                    source="colbert",
                )
                for r in self._rag.search(query=query, k=k)
            ]
        except Exception as e:
            logger.warning("ColBERT search: %s", e)
            return []

    def rerank(self, query: str, candidates: List[SearchHit], k: int) -> List[SearchHit]:
        if not candidates:
            return candidates
        if self._rag is not None and not self._fallback:
            return self._colbert_rerank(query, candidates, k)
        return self._ce_rerank(query, candidates, k)

    def _colbert_rerank(self, query: str, candidates: List[SearchHit], k: int) -> List[SearchHit]:
        try:
            texts = [c.text for c in candidates]
            results = self._rag.rerank(query=query, documents=texts, k=k)
            text_to_hit = {c.text: c for c in candidates}
            reranked = []
            for r in results:
                orig = text_to_hit.get(r["content"], candidates[0])
                reranked.append(SearchHit(
                    child_id=orig.child_id, parent_id=orig.parent_id,
                    text=orig.text, score=r["score"], source=orig.source,
                ))
            return reranked
        except Exception as e:
            logger.warning("ColBERT rerank error: %s", e)
            return self._ce_rerank(query, candidates, k)

    def _ce_rerank(self, query: str, candidates: List[SearchHit], k: int) -> List[SearchHit]:
        ce = _get_cross_encoder()
        scores = ce.predict([[query, c.text] for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [
            SearchHit(child_id=h.child_id, parent_id=h.parent_id,
                      text=h.text, score=float(s), source=h.source)
            for s, h in ranked[:k]
        ]


# ── Parent store ──────────────────────────────────────────────────────────────

class ParentStore:
    def __init__(self):
        self._store: Dict[str, ParentChunk] = {}

    def add(self, parents: List[ParentChunk]):
        for p in parents:
            self._store[p.id] = p

    def get(self, parent_id: str) -> Optional[ParentChunk]:
        return self._store.get(parent_id)

    def by_doc(self, doc_id: str) -> List[ParentChunk]:
        return [p for p in self._store.values() if p.doc_id == doc_id]

    def clear(self):
        self._store.clear()

    def remove_doc(self, doc_id: str):
        to_del = [k for k, v in self._store.items() if v.doc_id == doc_id]
        for k in to_del:
            del self._store[k]


# ── Document registry ─────────────────────────────────────────────────────────

class DocRegistry:
    def __init__(self):
        self._docs: Dict[str, DocRecord] = {}
        self._fp_to_id: Dict[str, str]   = {}   # fingerprint → doc_id

    def register(self, record: DocRecord):
        self._docs[record.doc_id] = record
        self._fp_to_id[record.fingerprint] = record.doc_id

    def find_by_fingerprint(self, fp: str) -> Optional[str]:
        return self._fp_to_id.get(fp)

    def get(self, doc_id: str) -> Optional[DocRecord]:
        return self._docs.get(doc_id)

    def all_docs(self) -> List[DocRecord]:
        return list(self._docs.values())

    def remove(self, doc_id: str):
        rec = self._docs.pop(doc_id, None)
        if rec:
            self._fp_to_id.pop(rec.fingerprint, None)


# ── Unified facade ────────────────────────────────────────────────────────────

class RAGStores:
    def __init__(self):
        self.qdrant   = QdrantStore()
        self.bm25     = BM25Store()
        self.colbert  = ColBERTStore()
        self.parents  = ParentStore()
        self.registry = DocRegistry()
        self._all_children: List[ChildChunk] = []   # for BM25 incremental rebuild
        self._ready = False

    async def add_documents(
        self,
        parents: List[ParentChunk],
        children: List[ChildChunk],
        doc_id: str,
    ):
        if not children:
            return

        dim = len(children[0].embedding) if children[0].embedding else EMBEDDING_DIM

        self.parents.add(parents)
        self._all_children.extend(children)

        # Qdrant (dense, supports per-doc filter natively)
        await self.qdrant.add_children(children, dim)

        # BM25 – rebuild from all children so per-doc filter works
        self.bm25.rebuild(self._all_children)

        # ColBERT – rebuild from all children
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.colbert.build_index, self._all_children)

        self._ready = True
        logger.info("All indexes ready (doc: %s)", doc_id)

        # Persist in-memory state whenever Qdrant is also persistent
        if QDRANT_URL:
            self.save_local()

    def is_ready(self) -> bool:
        return self._ready

    async def clear(self):
        await self.qdrant.clear_all()
        self.bm25     = BM25Store()
        self.colbert  = ColBERTStore()
        self.parents.clear()
        self.registry = DocRegistry()
        self._all_children.clear()
        self._ready = False

    def remove_doc(self, doc_id: str):
        """Remove one document from all stores."""
        self._all_children = [c for c in self._all_children if c.doc_id != doc_id]
        self.parents.remove_doc(doc_id)
        self.registry.remove(doc_id)
        if self._all_children:
            self.bm25.rebuild(self._all_children)
        else:
            self.bm25 = BM25Store()
        self._ready = bool(self._all_children)
        if QDRANT_URL:
            self.save_local()

    # ── Persistence (used with Docker / persistent Qdrant) ────────────────────
    #
    # Qdrant persists vectors on its own.  BM25, ParentStore, DocRegistry, and
    # _all_children are in-memory and would be lost on restart.  When QDRANT_URL
    # is set we pickle them to PERSIST_DIR so the app recovers without re-ingesting.

    def _persist_path(self, name: str) -> Path:
        p = Path(PERSIST_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p / name

    def save_local(self):
        """Pickle the in-memory state to disk."""
        try:
            with open(self._persist_path("bm25.pkl"), "wb") as f:
                pickle.dump((self.bm25._index, self.bm25._id_map,
                             self.bm25._parent_map, self.bm25._doc_map,
                             self.bm25._text_map), f)
            with open(self._persist_path("parents.pkl"), "wb") as f:
                pickle.dump(self.parents._store, f)
            with open(self._persist_path("registry.pkl"), "wb") as f:
                pickle.dump((self.registry._docs, self.registry._fp_to_id), f)
            with open(self._persist_path("children.pkl"), "wb") as f:
                pickle.dump(self._all_children, f)
            logger.info("Local state persisted to %s", PERSIST_DIR)
        except Exception as e:
            logger.warning("Failed to persist local state: %s", e)

    def load_local(self) -> bool:
        """
        Load previously persisted state.
        Returns True if everything was loaded successfully.
        """
        try:
            bm25_path     = self._persist_path("bm25.pkl")
            parents_path  = self._persist_path("parents.pkl")
            registry_path = self._persist_path("registry.pkl")
            children_path = self._persist_path("children.pkl")

            if not all(p.exists() for p in [bm25_path, parents_path, registry_path, children_path]):
                logger.info("No persisted local state found in %s", PERSIST_DIR)
                return False

            with open(bm25_path, "rb") as f:
                (self.bm25._index, self.bm25._id_map, self.bm25._parent_map,
                 self.bm25._doc_map, self.bm25._text_map) = pickle.load(f)

            with open(parents_path, "rb") as f:
                self.parents._store = pickle.load(f)

            with open(registry_path, "rb") as f:
                self.registry._docs, self.registry._fp_to_id = pickle.load(f)

            with open(children_path, "rb") as f:
                self._all_children = pickle.load(f)

            self._ready = bool(self._all_children)
            n_docs = len(self.registry.all_docs())
            logger.info("Loaded persisted state: %d docs, %d parents, %d children",
                        n_docs, len(self.parents._store), len(self._all_children))
            return True
        except Exception as e:
            logger.warning("Failed to load persisted state: %s", e)
            return False
