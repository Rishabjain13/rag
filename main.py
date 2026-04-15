"""
RAG PDF – FastAPI application entry point  (v2)

Start:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST   /ingest               Upload PDF (streaming or batch)
    POST   /query                Query with streaming SSE
    GET    /status               System status
    GET    /docs                 List all ingested documents
    DELETE /docs/{doc_id}        Remove a document
    DELETE /collection           Wipe everything
    POST   /evaluate             RAGAS-style evaluation
    GET    /sessions             Active session IDs
    DELETE /sessions/{id}        Delete a session
    GET    /docs (Swagger)       http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

# Set PyTorch CPU thread count before any imports that touch torch.
# Must happen in the main process — setting inside an executor thread is unreliable.
try:
    import torch
    _n = os.cpu_count() or 4
    torch.set_num_threads(_n)
    torch.set_num_interop_threads(max(1, _n // 2))
except Exception:
    pass

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.insert(0, os.path.dirname(__file__))

from api.routes import router
from config import RATE_LIMIT_QUERY, RATE_LIMIT_INGEST, QDRANT_URL, PERSIST_DIR
from rag.stores import RAGStores, _get_cross_encoder
from rag.cache import SemanticCache
from rag.sessions import SessionStore
from rag.embeddings import _local_model

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG PDF v2",
    description=(
        "Production RAG: late chunking · triple hybrid search (dense+ColBERT+BM25) · "
        "RRF fusion · multi-query expansion · MMR · contextual compression · "
        "ColBERT rerank · parent expansion · RAPTOR · QA cache · "
        "structured citations · self-RAG grounding · conversation history · "
        "RAGAS evaluation · multi-doc · streaming SSE"
    ),
    version="2.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply per-route rate limits via middleware
@app.middleware("http")
async def rate_limit_routes(request: Request, call_next):
    # Delegate to slowapi; per-route decorators can override
    return await call_next(request)

app.include_router(router)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    # ── Preload CPU models in background so first query has no cold-start ──────
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _local_model)          # BGE embedding model
    loop.run_in_executor(None, _get_cross_encoder)    # cross-encoder reranker

    stores = RAGStores()

    # ── Restore persisted state when using Docker / persistent Qdrant ─────────
    # load_local() restores BM25, ParentStore, DocRegistry, and _all_children.
    # restore_qdrant_if_needed() then verifies Qdrant has the vectors and
    # re-populates from persisted children if the collection is missing/empty
    # (e.g. Docker volume wiped, or DELETE /collection called previously).
    restored = False
    qdrant_repopulated = False
    if QDRANT_URL:
        restored = stores.load_local()
        if restored:
            qdrant_repopulated = await stores.restore_qdrant_if_needed()

    app.state.stores       = stores
    app.state.cache        = SemanticCache(stores.qdrant)
    app.state.sessions     = SessionStore()
    app.state.raptor_ready = {"ready": False}
    app.state.last_doc_id  = None

    logger.info("=" * 60)
    logger.info("  RAG PDF v2 ready")
    if QDRANT_URL:
        logger.info("  Qdrant: %s  (persistent)", QDRANT_URL)
        if restored:
            docs      = stores.registry.all_docs()
            n_parents = len(stores.parents._store)
            n_children = len(stores._all_children)
            logger.info("  Restored: %d docs, %d parents, %d children",
                        len(docs), n_parents, n_children)
            if qdrant_repopulated:
                logger.info("  Qdrant vectors re-populated from persisted state")
        else:
            logger.info("  No prior state found – ingest a PDF to begin")
    else:
        logger.info("  Qdrant: in-memory  (data lost on restart)")
        logger.info("  Tip: set QDRANT_URL=http://localhost:6333 for persistence")
    logger.info("  POST /ingest  to ingest a PDF")
    logger.info("  POST /query   to query (streaming SSE)")
    logger.info("  GET  /docs    Swagger UI")
    logger.info("=" * 60)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
