"""
FastAPI routes – v2

Endpoints
─────────
  POST   /ingest           Upload PDF; ?stream=true for SSE progress
  POST   /query            Query with streaming SSE
  GET    /status           System + per-doc status
  GET    /docs             List all ingested documents
  DELETE /docs/{doc_id}    Remove a single document
  DELETE /collection       Wipe everything
  POST   /evaluate         RAGAS-style offline evaluation
  GET    /sessions         List active session IDs

New SSE events on /query
────────────────────────
  {"type": "queries",   "queries": [...]}          multi-query variants
  {"type": "trace",     ...hit counts...}           retrieval diagnostics
  {"type": "context",   "page":N, "heading":"..."}  per-context metadata
  {"type": "token",     "text": "..."}              streamed answer token
  {"type": "citations", "citations": [...]}          structured [p.N] refs
  {"type": "grounding", "score": 0.92, ...}          self-RAG score
  {"type": "done"}

Security
────────
  API_KEY env var → X-API-Key header required on every request
  File size        → MAX_FILE_SIZE_MB limit on /ingest
  Rate limiting    → via slowapi (configured in main.py)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from config import UPLOAD_DIR, API_KEY, MAX_FILE_SIZE_MB
from rag.ingestion import ingest_pdf, ingest_pdf_streaming, fingerprint_file
from rag.retrieval import retrieve
from rag.generation import stream_answer, generate_hyde, parse_citations, verify_grounding, generate_answer
from rag.background import run_background_jobs
from rag.models import DocRecord
from rag.evaluation import evaluate_batch

logger = logging.getLogger(__name__)
router = APIRouter()

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Auth ──────────────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Depends(_api_key_header)):
    if not API_KEY:
        return    # no auth configured
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state(request: Request):
    return request.app.state


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── POST /ingest ──────────────────────────────────────────────────────────────

@router.post("/ingest", dependencies=[Depends(verify_api_key)])
async def ingest_endpoint(
    request: Request,
    file: UploadFile = File(...),
    doc_id: str = Form(default=""),
    stream: bool = Form(default=False),
):
    """
    Upload a PDF.  Pass stream=true (as a form field) for SSE progress events.
    Non-streaming returns JSON immediately when indexing is done.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    # ── File size guard ───────────────────────────────────────────────────────
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit ({len(contents) // 1024 // 1024} MB received).",
        )

    doc_id  = doc_id or str(uuid.uuid4())
    state   = _state(request)

    # ── Duplicate detection ───────────────────────────────────────────────────
    save_path = Path(UPLOAD_DIR) / f"{doc_id}.pdf"
    with open(save_path, "wb") as f:
        f.write(contents)

    fp = fingerprint_file(save_path)
    existing_id = state.stores.registry.find_by_fingerprint(fp)
    if existing_id:
        os.unlink(save_path)
        return {
            "status": "duplicate",
            "doc_id": existing_id,
            "message": f"This file was already ingested as doc_id={existing_id}.",
        }

    # Register with status "ingesting"
    state.stores.registry.register(DocRecord(
        doc_id=doc_id, filename=file.filename, fingerprint=fp,
        pages=0, parents=0, children=0,
        ingested_at=datetime.utcnow().isoformat(),
        status="ingesting",
    ))

    # ── Streaming ingest ──────────────────────────────────────────────────────
    if stream:
        async def sse_ingest():
            try:
                parents = children = None
                async for event in ingest_pdf_streaming(save_path, doc_id, state.stores):
                    if event.get("stage") == "done":
                        pages_count    = event.get("pages", 0)
                        parents_count  = event.get("parents", 0)
                        children_count = event.get("children", 0)
                        _finish_ingest(state, doc_id, file.filename, fp,
                                       pages_count, parents_count, children_count, save_path)
                    yield _sse(event)
            except Exception as e:
                rec = state.stores.registry.get(doc_id)
                if rec:
                    rec.status = "failed"
                yield _sse({"stage": "error", "message": str(e)})

        return StreamingResponse(sse_ingest(), media_type="text/event-stream")

    # ── Non-streaming ingest ──────────────────────────────────────────────────
    try:
        parents, children, n_pages = await ingest_pdf(str(save_path), doc_id, state.stores)
    except Exception as e:
        rec = state.stores.registry.get(doc_id)
        if rec:
            rec.status = "failed"
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    _finish_ingest(state, doc_id, file.filename, fp, n_pages, len(parents), len(children), save_path)

    return {
        "status": "ingested",
        "doc_id": doc_id,
        "parents": len(parents),
        "children": len(children),
        "message": "QA cache and RAPTOR tree building in background.",
    }


def _finish_ingest(state, doc_id, filename, fp, n_pages, n_parents, n_children, save_path):
    """Update registry, kick off background jobs."""
    rec = state.stores.registry.get(doc_id)
    if rec:
        rec.status   = "ready"
        rec.pages    = n_pages
        rec.parents  = n_parents
        rec.children = n_children

    # Store last-ingested for convenience
    state.last_doc_id = doc_id

    parents  = state.stores.parents.by_doc(doc_id)
    children = [c for c in state.stores._all_children if c.doc_id == doc_id]

    async def _retrieve_for_cache(query: str):
        _, contexts, _, _ = await retrieve(
            query, state.stores, state.cache,
            state.raptor_ready.get("ready", False), doc_id=doc_id,
        )
        return contexts

    asyncio.create_task(run_background_jobs(
        parents=parents, children=children, doc_id=doc_id,
        stores=state.stores, cache=state.cache,
        retrieve_fn=_retrieve_for_cache,
        raptor_ready_flag=state.raptor_ready,
    ))


# ── POST /query ───────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str]    = None    # scope to a specific document
    session_id: Optional[str] = None   # for conversation history
    style: str               = "default"   # concise | detailed | bullets | default
    max_tokens: int          = 1024
    stream: bool             = True


@router.post("/query", dependencies=[Depends(verify_api_key)])
async def query_endpoint(body: QueryRequest, request: Request):
    """
    Query the RAG system with full streaming SSE.
    """
    state = _state(request)

    if not state.stores.is_ready():
        raise HTTPException(status_code=503, detail="No document ingested yet. POST /ingest first.")

    # ── Session ───────────────────────────────────────────────────────────────
    session = None
    if body.session_id:
        session = state.sessions.get_or_create(body.session_id, doc_id=body.doc_id)
        session.add_turn("user", body.query)
    history = session.history_for_llm()[:-1] if session else []   # exclude last user msg (sent separately)

    doc_id = body.doc_id or (session.doc_id if session else None)

    async def event_generator():
        full_answer = []
        contexts    = []
        try:
            # 1. Retrieve
            route, contexts, cached, trace = await retrieve(
                query=body.query,
                stores=state.stores,
                cache=state.cache,
                raptor_ready=state.raptor_ready.get("ready", False),
                doc_id=doc_id,
                hyde_fn=generate_hyde,
            )

            yield _sse({"type": "route", "route": route.value})

            # Cached – return instantly
            if route.value == "cached" and cached:
                yield _sse({"type": "token", "text": cached})
                if session:
                    session.add_turn("assistant", cached)
                yield _sse({"type": "done"})
                return

            # Emit query variants
            if trace.queries and len(trace.queries) > 1:
                yield _sse({"type": "queries", "queries": trace.queries})

            # Emit retrieval trace
            yield _sse({
                "type":        "trace",
                "dense_hits":  trace.dense_hits,
                "colbert_hits":trace.colbert_hits,
                "bm25_hits":   trace.bm25_hits,
                "fused":       trace.fused_hits,
                "reranked":    trace.reranked_hits,
                "parents":     trace.parents_returned,
                "mmr":         trace.mmr_applied,
                "compressed":  trace.compression_applied,
            })

            # Context previews
            for ctx in contexts:
                yield _sse({
                    "type":    "context",
                    "page":    ctx.page_num,
                    "heading": ctx.heading,
                    "doc_id":  ctx.doc_id,
                    "preview": ctx.parent_text[:120] + "…",
                })

            # 2. Stream answer
            async for token in stream_answer(
                body.query, contexts,
                history=history,
                style=body.style,
                max_tokens=body.max_tokens,
            ):
                full_answer.append(token)
                yield _sse({"type": "token", "text": token})

            answer_text = "".join(full_answer)

            # 3. Citations
            citations = parse_citations(answer_text, contexts)
            if citations:
                yield _sse({"type": "citations", "citations": citations})

            # 4. Self-RAG grounding (non-blocking – runs and emits after done)
            yield _sse({"type": "done"})

            # Update session history
            if session:
                session.add_turn("assistant", answer_text)

            # 5. Grounding score (sent after done so client can display answer first)
            grounding = await verify_grounding(body.query, answer_text, contexts)
            yield _sse({"type": "grounding", **grounding})

        except Exception as e:
            logger.error("Query error: %s", e)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── GET /status ───────────────────────────────────────────────────────────────

@router.get("/status", dependencies=[Depends(verify_api_key)])
async def status_endpoint(request: Request):
    state = _state(request)
    return {
        "ready":        state.stores.is_ready(),
        "raptor_ready": state.raptor_ready.get("ready", False),
        "docs":         len(state.stores.registry.all_docs()),
        "total_parents":  sum(r.parents  for r in state.stores.registry.all_docs()),
        "total_children": sum(r.children for r in state.stores.registry.all_docs()),
        "active_sessions": len(state.sessions.all_ids()),
    }


# ── GET /docs ─────────────────────────────────────────────────────────────────

@router.get("/docs", dependencies=[Depends(verify_api_key)])
async def list_docs(request: Request):
    state = _state(request)
    return {
        "documents": [
            {
                "doc_id":      r.doc_id,
                "filename":    r.filename,
                "pages":       r.pages,
                "parents":     r.parents,
                "children":    r.children,
                "status":      r.status,
                "ingested_at": r.ingested_at,
            }
            for r in state.stores.registry.all_docs()
        ]
    }


# ── DELETE /docs/{doc_id} ─────────────────────────────────────────────────────

@router.delete("/docs/{doc_id}", dependencies=[Depends(verify_api_key)])
async def delete_doc(doc_id: str, request: Request):
    state = _state(request)
    if not state.stores.registry.get(doc_id):
        raise HTTPException(status_code=404, detail=f"doc_id={doc_id} not found.")
    await state.stores.remove_doc(doc_id)
    pdf = Path(UPLOAD_DIR) / f"{doc_id}.pdf"
    if pdf.exists():
        pdf.unlink()
    return {"status": "deleted", "doc_id": doc_id}


# ── DELETE /collection ────────────────────────────────────────────────────────

@router.delete("/collection", dependencies=[Depends(verify_api_key)])
async def delete_collection(request: Request):
    state = _state(request)
    await state.stores.clear()
    state.raptor_ready["ready"] = False
    return {"status": "cleared"}


# ── POST /evaluate ────────────────────────────────────────────────────────────

class EvalCase(BaseModel):
    query: str
    expected: str
    doc_id: Optional[str] = None


class EvalRequest(BaseModel):
    test_cases: List[EvalCase]


@router.post("/evaluate", dependencies=[Depends(verify_api_key)])
async def evaluate_endpoint(body: EvalRequest, request: Request):
    """
    RAGAS-style evaluation.

    For each test case, the system:
      1. Runs full retrieval (no cache, direct route)
      2. Generates an answer
      3. Scores it on 4 metrics via the fast OpenRouter model
    """
    state = _state(request)

    if not state.stores.is_ready():
        raise HTTPException(status_code=503, detail="No document ingested yet.")

    prepared = []
    for tc in body.test_cases:
        _, contexts, _, _ = await retrieve(
            query=tc.query,
            stores=state.stores,
            cache=state.cache,
            raptor_ready=False,   # skip cache + RAPTOR for fair eval
            doc_id=tc.doc_id,
            hyde_fn=None,
        )
        answer = await generate_answer(tc.query, contexts)
        prepared.append({
            "query":    tc.query,
            "expected": tc.expected,
            "contexts": contexts,
            "answer":   answer,
        })

    return await evaluate_batch(prepared)


# ── GET /sessions ─────────────────────────────────────────────────────────────

@router.get("/sessions", dependencies=[Depends(verify_api_key)])
async def list_sessions(request: Request):
    state = _state(request)
    return {"session_ids": state.sessions.all_ids()}


@router.delete("/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_session(session_id: str, request: Request):
    state = _state(request)
    state.sessions.delete(session_id)
    return {"status": "deleted", "session_id": session_id}
