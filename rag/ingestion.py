"""
Ingestion pipeline: PDF → pages → parent/child chunks → all indexes.

Improvements over v1
────────────────────
  pdfplumber parser   handles complex layouts, columns, headers far better
                      than pypdf; extracts tables as markdown grids
  Table extraction    tables become | col | col | rows preserved as text
  Section-boundary    split at detected headings first, then overflow by tokens
  chunking            preserves semantic units; less wasted overlap
  Doc fingerprint     MD5 hash of file bytes → instant duplicate detection
  Progress generator  ingest_pdf_streaming() yields progress events for SSE
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import AsyncIterator, Dict, List, Tuple

import tiktoken

from config import (
    PARENT_CHUNK_TOKENS,
    CHILD_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)
from config import JINA_API_KEY, JINA_FOR_INGEST, JINA_CONCURRENCY
from rag.models import PageContent, ParentChunk, ChildChunk
from rag.embeddings import embed_children, _embed_local

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
    r"[A-Z][A-Z\s]{2,}(?:[:\.])?(?=\s|$)"          # ALL CAPS
    r"|\d+(?:\.\d+)*\s+[A-Z][^\n]{0,80}"             # 1.2 Title
    r"|(?:Abstract|Introduction|Conclusion|References"
    r"|Background|Methodology|Method|Result|Discussion"
    r"|Related Work|Future Work)[^\n]{0,60}"          # common section names
    r")$",
    re.MULTILINE,
)


def _extract_headings(text: str) -> List[str]:
    return _HEADING_RE.findall(text)[:10]


def _table_to_markdown(table: List[List]) -> str:
    """Convert a pdfplumber table (list of rows) to a markdown table string."""
    if not table:
        return ""
    rows = [[str(cell or "").strip() for cell in row] for row in table]
    # Remove completely empty rows
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
        # Pad/truncate row to header width
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)


# ── PDF parsing ───────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str | Path) -> List[PageContent]:
    """
    Extract text + tables from every page using pdfplumber.
    Falls back to pypdf if pdfplumber fails on a page.
    """
    pages: List[PageContent] = []

    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text() or ""
                text = text.strip()

                # Extract tables as markdown
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

                # Merge tables into text (append after page text)
                if table_strings:
                    text = text + "\n\n" + "\n\n".join(table_strings)

                headings = _extract_headings(text)
                pages.append(PageContent(
                    page_num=i + 1,
                    text=text,
                    headings=headings,
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

    logger.info("Parsed %d non-empty pages from %s", len(pages), pdf_path)
    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def _detect_section_boundaries(text: str) -> List[Tuple[int, str]]:
    """
    Return list of (char_offset, heading_text) for each section boundary found.
    Used to split at semantically meaningful points before falling back to
    pure token-count splits.
    """
    boundaries: List[Tuple[int, str]] = []
    for m in _HEADING_RE.finditer(text):
        boundaries.append((m.start(), m.group().strip()))
    return boundaries


def _make_parents(pages: List[PageContent], doc_id: str) -> List[ParentChunk]:
    """
    Section-boundary aware parent chunking:
      1. Split page text at detected headings
      2. If a section is larger than PARENT_CHUNK_TOKENS, further split by tokens
      3. If a section is tiny, merge with the next section
    """
    parents: List[ParentChunk] = []

    for page in pages:
        full_text = page.text
        boundaries = _detect_section_boundaries(full_text)

        if len(boundaries) > 1:
            # Split at heading boundaries
            sections: List[Tuple[str, str]] = []   # (section_text, heading)
            for idx, (start, heading) in enumerate(boundaries):
                end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(full_text)
                section_text = full_text[start:end].strip()
                if section_text:
                    sections.append((section_text, heading))
        else:
            heading = page.headings[0] if page.headings else ""
            sections = [(full_text, heading)]

        # For each section, token-split if too large; merge if too small
        buffer_text = ""
        buffer_heading = ""
        MIN_TOKENS = PARENT_CHUNK_TOKENS // 4

        for section_text, heading in sections:
            tok_count = _count_tokens(section_text)

            if tok_count > PARENT_CHUNK_TOKENS:
                # Flush buffer first
                if buffer_text:
                    _add_parent(parents, buffer_text, buffer_heading, doc_id, page.page_num)
                    buffer_text = buffer_heading = ""
                # Split large section
                for sub in _split_by_tokens(section_text, PARENT_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS):
                    _add_parent(parents, sub, heading, doc_id, page.page_num)

            elif tok_count < MIN_TOKENS:
                # Too small – accumulate into buffer
                buffer_text = (buffer_text + " " + section_text).strip()
                buffer_heading = buffer_heading or heading

                # If buffer is now large enough, flush
                if _count_tokens(buffer_text) >= MIN_TOKENS:
                    _add_parent(parents, buffer_text, buffer_heading, doc_id, page.page_num)
                    buffer_text = buffer_heading = ""
            else:
                # Flush buffer then add this section
                if buffer_text:
                    _add_parent(parents, buffer_text, buffer_heading, doc_id, page.page_num)
                    buffer_text = buffer_heading = ""
                _add_parent(parents, section_text, heading, doc_id, page.page_num)

        # Flush any remaining buffer
        if buffer_text:
            _add_parent(parents, buffer_text, buffer_heading, doc_id, page.page_num)

    logger.info("Section-aware chunking: %d parent chunks", len(parents))
    return parents


def _add_parent(parents: List[ParentChunk], text: str, heading: str, doc_id: str, page_num: int):
    tok = _count_tokens(text)
    parents.append(ParentChunk.create(
        doc_id=doc_id, text=text, page_num=page_num,
        heading=heading, section=heading, token_count=tok,
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
    parents = _make_parents(pages, doc_id)
    children: List[ChildChunk] = []
    for p in parents:
        children.extend(_make_children(p))
    logger.info("Small-to-big: %d parents → %d children", len(parents), len(children))
    return parents, children


# ── Embedding helpers ─────────────────────────────────────────────────────────

async def _embed_all_children(
    parents: List[ParentChunk],
    children: List[ChildChunk],
) -> List[ChildChunk]:
    """
    Embed all children.

    Local path (JINA_FOR_INGEST=false or no JINA_API_KEY):
      ONE batch encode call for all children → thread-safe, no concurrency
      issues, sentence-transformers handles internal batching (batch_size=32).
      For 1621 children: ~51 batches × ~100 ms = ~5 s on CPU.

    Jina path (JINA_API_KEY set and JINA_FOR_INGEST=true):
      asyncio.gather → per-parent concurrent API calls, rate-limited by
      token-bucket (JINA_RPM) and semaphore (JINA_CONCURRENCY).
    """
    # ── Fast local path: single batch encode, fully thread-safe ──────────────
    use_jina = bool(JINA_API_KEY) and JINA_FOR_INGEST
    if not use_jina:
        logger.info("Local batch embed: %d children in one pass…", len(children))
        loop = asyncio.get_event_loop()
        texts    = [c.text for c in children]
        all_embs = await loop.run_in_executor(None, _embed_local, texts)
        for child, emb in zip(children, all_embs):
            child.embedding = emb
        logger.info("Local embed done: %d embeddings", len(all_embs))
        return children

    # ── Jina path: per-parent concurrent late-chunking ────────────────────────
    parent_map: Dict[str, ParentChunk] = {p.id: p for p in parents}
    by_parent: Dict[str, List[ChildChunk]] = defaultdict(list)
    for c in children:
        by_parent[c.parent_id].append(c)

    parent_ids   = list(by_parent.keys())
    kids_per_pid = [by_parent[pid] for pid in parent_ids]

    async def _embed_one(pid: str, kids: List[ChildChunk]) -> List[ChildChunk]:
        texts = [k.text for k in kids]
        embs  = await embed_children(parent_map[pid].text, texts)
        for kid, emb in zip(kids, embs):
            kid.embedding = emb
        return kids

    logger.info("Jina: embedding %d parent batches concurrently (JINA_CONCURRENCY=%d)…",
                len(parent_ids), JINA_CONCURRENCY)

    results = await asyncio.gather(
        *[_embed_one(pid, kids) for pid, kids in zip(parent_ids, kids_per_pid)]
    )

    embedded: List[ChildChunk] = [kid for batch in results for kid in batch]
    logger.info("Jina: embedded %d children", len(embedded))
    return embedded


# ── Standard ingestion (returns when done) ────────────────────────────────────

async def ingest_pdf(
    pdf_path: str | Path,
    doc_id: str,
    stores,
) -> Tuple[List[ParentChunk], List[ChildChunk], int]:
    """
    End-to-end ingestion. Returns (parents, children, n_pages).
    """
    pages = parse_pdf(pdf_path)
    parents, children = build_chunks(pages, doc_id)
    embedded = await _embed_all_children(parents, children)
    await stores.add_documents(parents, embedded, doc_id)
    return parents, embedded, len(pages)


# ── Streaming ingestion (yields progress events for SSE) ─────────────────────

async def ingest_pdf_streaming(
    pdf_path: str | Path,
    doc_id: str,
    stores,
) -> AsyncIterator[dict]:
    """
    Same as ingest_pdf but yields progress dicts so the API can stream them.
    Usage: async for event in ingest_pdf_streaming(...): ...
    """
    yield {"stage": "parsing", "progress": 0.05, "message": "Parsing PDF pages…"}
    pages = parse_pdf(pdf_path)
    n_tables = sum(len(p.tables) for p in pages)
    yield {
        "stage": "parsing", "progress": 0.2,
        "message": f"Parsed {len(pages)} pages, {n_tables} tables extracted",
    }

    yield {"stage": "chunking", "progress": 0.25, "message": "Building parent/child chunks…"}
    parents, children = build_chunks(pages, doc_id)
    yield {
        "stage": "chunking", "progress": 0.40,
        "message": f"{len(parents)} parents, {len(children)} children created",
    }

    yield {"stage": "embedding", "progress": 0.42, "message": "Embedding children…"}

    use_jina = bool(JINA_API_KEY) and JINA_FOR_INGEST

    if not use_jina:
        # ── Local batch path: one encode call, no threading issues ────────────
        loop = asyncio.get_event_loop()
        texts    = [c.text for c in children]
        all_embs = await loop.run_in_executor(None, _embed_local, texts)
        for child, emb in zip(children, all_embs):
            child.embedding = emb
        embedded = list(children)
        yield {"stage": "embedding", "progress": 0.80,
               "message": f"Embedded {len(embedded)} children (local batch)"}
    else:
        # ── Jina path: per-parent with progress reporting ─────────────────────
        parent_map: Dict[str, ParentChunk] = {p.id: p for p in parents}
        by_parent: Dict[str, List[ChildChunk]] = defaultdict(list)
        for c in children:
            by_parent[c.parent_id].append(c)

        embedded: List[ChildChunk] = []
        total = len(by_parent)
        for i, (pid, kids) in enumerate(by_parent.items(), start=1):
            child_texts = [k.text for k in kids]
            embs = await embed_children(parent_map[pid].text, child_texts)
            for kid, emb in zip(kids, embs):
                kid.embedding = emb
                embedded.append(kid)
            if i % max(1, total // 10) == 0:
                progress = 0.42 + 0.38 * (i / total)
                yield {"stage": "embedding", "progress": round(progress, 2),
                       "message": f"Embedded {i}/{total} parent batches"}

    yield {"stage": "indexing", "progress": 0.82, "message": "Building Qdrant + BM25 + ColBERT indexes…"}
    await stores.add_documents(parents, embedded, doc_id)
    yield {"stage": "indexing", "progress": 1.0, "message": "All indexes ready"}

    # Attach result to last event
    yield {
        "stage": "done",
        "progress": 1.0,
        "pages": len(pages),
        "parents": len(parents),
        "children": len(embedded),
        "doc_id": doc_id,
    }
    # Make parents + children accessible to caller via generator send() – not possible
    # easily, so store them on the generator object via a mutable dict trick
    # Caller should call ingest_pdf() for the return value if needed.
