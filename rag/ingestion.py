"""
Ingestion pipeline: document → pages → parent/child chunks → indexes.

Parallelism notes
─────────────────
  parse_pdf()      CPU-bound (pdfplumber) → run_in_executor so the event loop
                   is never blocked during parsing.

  build_chunks()   CPU-bound (tiktoken) → run_in_executor.

  _embed_children() One batch encode() call for ALL children together
                   via run_in_executor. batch_size=64 handled internally.

  add_documents()  Qdrant upsert (async) + BM25 rebuild (executor) run after
                   embedding. ColBERT builds as a background asyncio task.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import AsyncIterator, List, Tuple

import tiktoken

from config import (
    PARENT_CHUNK_TOKENS,
    CHILD_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)
from rag.models import PageContent, ParentChunk, ChildChunk
from rag.embeddings import _embed_local

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


# ── Utilities ─────────────────────────────────────────────────────────────────

def fingerprint_file(path: str | Path) -> str:
    """Return MD5 hex digest of file bytes for duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split_by_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    tokens = _enc.encode(text)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


# ── Heading detection ─────────────────────────────────────────────────────────

_HEADING_RE = re.compile(
    r"^(?:"
    r"[A-Z][A-Z\s]{2,}(?:[:\.])?(?=\s|$)"
    r"|\d+(?:\.\d+)*\s+[A-Z][^\n]{0,80}"
    r"|(?:Abstract|Introduction|Conclusion|References"
    r"|Background|Methodology|Method|Result|Discussion"
    r"|Related Work|Future Work)[^\n]{0,60}"
    r")$",
    re.MULTILINE,
)


def _extract_headings(text: str) -> List[str]:
    return _HEADING_RE.findall(text)[:10]


def _table_to_markdown(table: List[List]) -> str:
    if not table:
        return ""
    rows = [[str(cell or "").strip() for cell in row] for row in table]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    header = rows[0]
    sep = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)


# ── PDF parsing ───────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str | Path) -> List[PageContent]:
    """
    Extract text + tables from every page using pdfplumber.
    Falls back to pypdf if pdfplumber fails on a page.
    Runs synchronously — call via run_in_executor to avoid blocking.
    """
    pages: List[PageContent] = []

    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                table_strings: List[str] = []
                try:
                    for table in page.extract_tables():
                        md = _table_to_markdown(table)
                        if md:
                            table_strings.append(md)
                except Exception:
                    pass

                if not text and not table_strings:
                    continue

                if table_strings:
                    text = text + "\n\n" + "\n\n".join(table_strings)

                pages.append(PageContent(
                    page_num=i + 1,
                    text=text,
                    headings=_extract_headings(text),
                    tables=table_strings,
                ))

    except Exception as e:
        logger.warning("pdfplumber failed (%s), falling back to pypdf", e)
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pages.append(PageContent(
                page_num=i + 1,
                text=text,
                headings=_extract_headings(text),
            ))

    logger.info("Parsed %d non-empty pages", len(pages))
    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def _detect_section_boundaries(text: str) -> List[Tuple[int, str]]:
    boundaries: List[Tuple[int, str]] = []
    for m in _HEADING_RE.finditer(text):
        boundaries.append((m.start(), m.group().strip()))
    return boundaries


def _make_parents(pages: List[PageContent], doc_id: str) -> List[ParentChunk]:
    parents: List[ParentChunk] = []
    for page in pages:
        full_text = page.text
        boundaries = _detect_section_boundaries(full_text)

        if len(boundaries) > 1:
            sections: List[Tuple[str, str]] = []
            for idx, (start, heading) in enumerate(boundaries):
                end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(full_text)
                section_text = full_text[start:end].strip()
                if section_text:
                    sections.append((section_text, heading))
        else:
            heading = page.headings[0] if page.headings else ""
            sections = [(full_text, heading)]

        buffer_text = ""
        buffer_heading = ""
        MIN_TOKENS = PARENT_CHUNK_TOKENS // 4

        for section_text, heading in sections:
            tok_count = _count_tokens(section_text)

            if tok_count > PARENT_CHUNK_TOKENS:
                if buffer_text:
                    _add_parent(parents, buffer_text, buffer_heading, doc_id, page.page_num)
                    buffer_text = buffer_heading = ""
                for sub in _split_by_tokens(section_text, PARENT_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS):
                    _add_parent(parents, sub, heading, doc_id, page.page_num)
            elif tok_count < MIN_TOKENS:
                buffer_text = (buffer_text + " " + section_text).strip()
                buffer_heading = buffer_heading or heading
                if _count_tokens(buffer_text) >= MIN_TOKENS:
                    _add_parent(parents, buffer_text, buffer_heading, doc_id, page.page_num)
                    buffer_text = buffer_heading = ""
            else:
                if buffer_text:
                    _add_parent(parents, buffer_text, buffer_heading, doc_id, page.page_num)
                    buffer_text = buffer_heading = ""
                _add_parent(parents, section_text, heading, doc_id, page.page_num)

        if buffer_text:
            _add_parent(parents, buffer_text, buffer_heading, doc_id, page.page_num)

    logger.info("Section-aware chunking: %d parent chunks", len(parents))
    return parents


def _add_parent(parents: List[ParentChunk], text: str, heading: str, doc_id: str, page_num: int):
    parents.append(ParentChunk.create(
        doc_id=doc_id, text=text, page_num=page_num,
        heading=heading, section=heading, token_count=_count_tokens(text),
    ))


def _make_children(parent: ParentChunk) -> List[ChildChunk]:
    segments = _split_by_tokens(parent.text, CHILD_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
    return [
        ChildChunk.create(
            parent_id=parent.id, doc_id=parent.doc_id, text=seg,
            page_num=parent.page_num, index_in_parent=i,
        )
        for i, seg in enumerate(segments)
    ]


def build_chunks(pages: List[PageContent], doc_id: str) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """
    Build parent + child chunks from parsed pages.
    Synchronous — call via run_in_executor to avoid blocking.
    """
    parents = _make_parents(pages, doc_id)
    children: List[ChildChunk] = []
    for p in parents:
        children.extend(_make_children(p))
    logger.info("Small-to-big: %d parents → %d children", len(parents), len(children))
    return parents, children


# ── Embedding ─────────────────────────────────────────────────────────────────

async def _embed_children(children: List[ChildChunk]) -> List[ChildChunk]:
    """
    Batch-embed all children in a single executor call.
    sentence-transformers splits internally at batch_size=64.
    """
    if not children:
        return children
    logger.info("Embedding %d children (batch, local)…", len(children))
    loop = asyncio.get_event_loop()
    texts = [c.text for c in children]
    embeddings = await loop.run_in_executor(None, _embed_local, texts)
    for child, emb in zip(children, embeddings):
        child.embedding = emb
    logger.info("Embedding complete: %d vectors", len(embeddings))
    return children


# ── Standard ingestion ────────────────────────────────────────────────────────

async def ingest_pdf(
    pdf_path: str | Path,
    doc_id: str,
    stores,
) -> Tuple[List[ParentChunk], List[ChildChunk], int]:
    """
    End-to-end ingestion. Returns (parents, children, n_pages).
    parse + chunk run in thread-pool so the event loop stays free.
    """
    loop = asyncio.get_event_loop()

    pages = await loop.run_in_executor(None, parse_pdf, pdf_path)
    parents, children = await loop.run_in_executor(None, build_chunks, pages, doc_id)
    children = await _embed_children(children)
    await stores.add_documents(parents, children, doc_id)

    return parents, children, len(pages)


# ── Streaming ingestion ───────────────────────────────────────────────────────

async def ingest_pdf_streaming(
    pdf_path: str | Path,
    doc_id: str,
    stores,
) -> AsyncIterator[dict]:
    """
    Same as ingest_pdf but yields SSE progress events.
    """
    loop = asyncio.get_event_loop()

    yield {"stage": "parsing", "progress": 0.05, "message": "Parsing PDF pages…"}
    pages = await loop.run_in_executor(None, parse_pdf, pdf_path)
    n_tables = sum(len(p.tables) for p in pages)
    yield {
        "stage": "parsing", "progress": 0.20,
        "message": f"Parsed {len(pages)} pages, {n_tables} tables",
    }

    yield {"stage": "chunking", "progress": 0.25, "message": "Building parent/child chunks…"}
    parents, children = await loop.run_in_executor(None, build_chunks, pages, doc_id)
    yield {
        "stage": "chunking", "progress": 0.40,
        "message": f"{len(parents)} parents, {len(children)} children",
    }

    yield {"stage": "embedding", "progress": 0.42,
           "message": f"Embedding {len(children)} chunks…"}
    children = await _embed_children(children)
    yield {"stage": "embedding", "progress": 0.80,
           "message": f"Embedded {len(children)} chunks"}

    yield {"stage": "indexing", "progress": 0.82,
           "message": "Building Qdrant + BM25 indexes…"}
    await stores.add_documents(parents, children, doc_id)
    yield {"stage": "indexing", "progress": 1.0, "message": "All indexes ready"}

    yield {
        "stage": "done", "progress": 1.0,
        "pages": len(pages), "parents": len(parents),
        "children": len(children), "doc_id": doc_id,
    }
