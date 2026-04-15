"""Core data classes shared across the entire pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid


class QueryRoute(str, Enum):
    CACHED = "cached"
    RAPTOR  = "raptor"
    HYDE    = "hyde"
    DIRECT  = "direct"


# ── Ingestion ─────────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    page_num: int
    text: str
    headings: List[str] = field(default_factory=list)
    tables: List[str]   = field(default_factory=list)   # markdown-formatted tables


@dataclass
class ParentChunk:
    id: str
    doc_id: str
    text: str
    page_num: int
    heading: str
    section: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, doc_id: str, text: str, page_num: int,
               heading: str = "", section: str = "", token_count: int = 0) -> "ParentChunk":
        return cls(id=str(uuid.uuid4()), doc_id=doc_id, text=text,
                   page_num=page_num, heading=heading, section=section, token_count=token_count)


@dataclass
class ChildChunk:
    id: str
    parent_id: str
    doc_id: str
    text: str
    page_num: int
    index_in_parent: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, parent_id: str, doc_id: str, text: str,
               page_num: int, index_in_parent: int) -> "ChildChunk":
        return cls(id=str(uuid.uuid4()), parent_id=parent_id, doc_id=doc_id,
                   text=text, page_num=page_num, index_in_parent=index_in_parent)


# ── Document registry ─────────────────────────────────────────────────────────

@dataclass
class DocRecord:
    doc_id: str
    filename: str
    fingerprint: str   # MD5 of file bytes – used for dedup
    pages: int
    parents: int
    children: int
    ingested_at: str
    status: str = "ready"   # "ingesting" | "ready" | "failed"


# ── Background models ─────────────────────────────────────────────────────────

@dataclass
class RaptorNode:
    id: str
    doc_id: str
    text: str
    level: int
    child_ids: List[str]        = field(default_factory=list)
    embedding: Optional[List[float]] = None

    @classmethod
    def create(cls, doc_id: str, text: str, level: int, child_ids: List[str]) -> "RaptorNode":
        return cls(id=str(uuid.uuid4()), doc_id=doc_id, text=text, level=level, child_ids=child_ids)


@dataclass
class CachedQA:
    id: str
    question: str
    answer: str
    doc_id: str
    embedding: Optional[List[float]] = None

    @classmethod
    def create(cls, question: str, answer: str, doc_id: str) -> "CachedQA":
        return cls(id=str(uuid.uuid4()), question=question, answer=answer, doc_id=doc_id)


# ── Retrieval models ──────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    child_id: str
    parent_id: str
    text: str
    score: float
    source: str       # "dense" | "colbert" | "bm25"


@dataclass
class RetrievedContext:
    parent_id: str
    parent_text: str
    page_num: int
    heading: str
    doc_id: str
    rrf_score: float
    child_ids: List[str]  = field(default_factory=list)
    sources: List[str]    = field(default_factory=list)


@dataclass
class RetrievalTrace:
    """Diagnostic information emitted alongside every answer."""
    route: str
    queries: List[str]        = field(default_factory=list)
    dense_hits: int           = 0
    colbert_hits: int         = 0
    bm25_hits: int            = 0
    fused_hits: int           = 0
    reranked_hits: int        = 0
    parents_returned: int     = 0
    compression_applied: bool = False
    mmr_applied: bool         = False


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    query: str
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevance: float
    answer: str = ""
    error: str  = ""

    @property
    def mean_score(self) -> float:
        return (self.context_recall + self.context_precision +
                self.faithfulness + self.answer_relevance) / 4
