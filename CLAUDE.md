# CLAUDE.md — RAG PDF

Project context for Claude Code. Read this before touching any file.

---

## What this project is

A production RAG (Retrieval-Augmented Generation) system that ingests PDF documents and answers questions about them via a streaming FastAPI + SSE API. It runs fully locally (in-memory Qdrant, local embeddings) with no external LLM dependencies beyond the OpenRouter API.

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (copy and fill in OPENROUTER_API_KEY at minimum)
cp .env.example .env

# Start server
uvicorn main:app --reload --port 8000

# Swagger UI
open http://localhost:8000/docs
```

---

## File map

```
config.py               All env-var settings. Single source of truth — add new
                        settings here first, then import them wherever needed.

main.py                 FastAPI app. Startup wires up RAGStores, SemanticCache,
                        SessionStore onto app.state. slowapi rate limiting lives here.

rag/models.py           Pure dataclasses — no logic. Add new data shapes here.
                        Key types: ParentChunk, ChildChunk, SearchHit,
                        RetrievedContext, RetrievalTrace, EvalResult, DocRecord.

rag/embeddings.py       Single embedding interface used everywhere.
                        Priority: Jina API (native late chunking) → local
                        sentence-transformers (manual token mean-pooling).
                        Public API: embed_children(), embed_query(), embed_texts().

rag/ingestion.py        PDF → chunks → embeddings → stores.
                        parse_pdf()           pdfplumber + markdown tables + pypdf fallback
                        build_chunks()        section-boundary parents → child splits
                        fingerprint_file()    MD5 dedup
                        ingest_pdf()          blocking, returns (parents, children)
                        ingest_pdf_streaming() async generator, yields progress dicts

rag/stores.py           All three indexes + parent store + doc registry.
                        QdrantStore     dense vector search, per-doc filter via Filter()
                        BM25Store       rank-bm25, rebuilt from all children on each ingest
                        ColBERTStore    RAGatouille; cross-encoder fallback if unavailable
                        ParentStore     in-memory dict: parent_id → ParentChunk
                        DocRegistry     doc_id ↔ fingerprint mapping + DocRecord metadata
                        RAGStores       unified facade used by the rest of the app;
                                        holds _all_children list for BM25 rebuilds

rag/cache.py            SemanticCache wraps QdrantStore.search_cache().
                        lookup(query, doc_id=None) → cached answer or None
                        store(id, question, answer, doc_id)

rag/retrieval.py        The full retrieval pipeline.
                        route_query()       cached → raptor → hyde → direct
                        expand_query()      fast model generates 3 query variants
                        hybrid_search()     dense + colbert + bm25 in parallel → RRF
                        rerank()            ColBERT MaxSim or cross-encoder
                        expand_to_parents() child hits → deduplicated parents
                        apply_mmr()         relevance × diversity reordering
                        compress_all()      fast model extracts relevant sentences
                        retrieve()          full pipeline, returns (route, contexts,
                                            cached_answer, RetrievalTrace)

rag/generation.py       LLM calls via OpenRouter (OpenAI-compatible API).
                        generate_hyde()     hypothetical answer for HyDE embedding
                        stream_answer()     async generator, yields tokens
                        generate_answer()   non-streaming version (eval + cache builder)
                        parse_citations()   extracts [p.N] markers → typed list
                        verify_grounding()  self-RAG: fast model scores answer faithfulness
                        generate_summary()  used by RAPTOR cluster summarisation

rag/sessions.py         In-memory conversation history.
                        SessionStore        dict of Session objects, 1-hr TTL eviction
                        Session             holds messages list + optional doc_id filter

rag/evaluation.py       RAGAS-style metrics via OpenRouter fast model.
                        evaluate_single()   scores one (query, expected, contexts, answer)
                        evaluate_batch()    runs evaluate_single in parallel, returns aggregate

rag/background.py       Async tasks launched after ingest (non-blocking).
                        build_qa_cache()    headings → questions → answers → cache
                        build_raptor_tree() KMeans cluster + summarise recursively
                        run_background_jobs() orchestrates both with sleep delays
                                             +2 min QA cache, +5 min RAPTOR

api/routes.py           All HTTP endpoints.
                        POST /ingest        file upload + dedup + optional SSE progress
                        POST /query         full pipeline + SSE streaming
                        GET  /status        system health
                        GET  /docs          document registry list
                        DELETE /docs/{id}   remove one document
                        DELETE /collection  wipe everything
                        POST /evaluate      RAGAS eval endpoint
                        GET  /sessions      active session IDs
                        DELETE /sessions/id delete session
```

---

## Data flow: ingest

```
PDF bytes
  fingerprint_file()  →  DocRegistry.find_by_fingerprint()  →  reject duplicate
  parse_pdf()         →  List[PageContent]  (text + tables + headings per page)
  build_chunks()      →  List[ParentChunk] + List[ChildChunk]
  embed_children()    →  ChildChunk.embedding filled (late chunking per parent batch)
  RAGStores.add_documents()
    ├── ParentStore.add()
    ├── QdrantStore.add_children()
    ├── BM25Store.rebuild()        ← rebuilds from ALL children (multi-doc safe)
    └── ColBERTStore.build_index() ← rebuilds from ALL children
  asyncio.create_task(run_background_jobs())   ← non-blocking
```

## Data flow: query

```
QueryRequest
  retrieve()
    route_query()              → QueryRoute enum
    expand_query()             → [original, v1, v2, v3]  (fast model)
    for each query variant:
      hybrid_search()          → fused SearchHit list (RRF of 3 indexes)
    rrf_fusion(all lists)      → top-15
    rerank()                   → top-5
    expand_to_parents()        → List[RetrievedContext]
    apply_mmr()                → reordered by diversity
    compress_all()             → parent_text trimmed to relevant sentences (fast model)
    returns (route, contexts, cached, RetrievalTrace)
  stream_answer()              → async token generator (OPENROUTER_MODEL streaming)
  parse_citations()            → [{page, heading, doc_id}, ...]
  verify_grounding()           → {grounded, score, issues}  (fast model, after done event)
```

---

## Key design decisions

**Single RAGStores facade on app.state**
All indexes live on `app.state.stores`. Routes access it via `request.app.state`. Never instantiate RAGStores inside a request handler.

**BM25 is rebuilt on every ingest**
`rank_bm25` has no incremental add. `RAGStores._all_children` accumulates every child from every document; BM25Store.rebuild() re-indexes the whole list. Fast enough up to ~100k chunks.

**ColBERT also rebuilds on every ingest**
Same reason. RAGatouille's `overwrite_index=True` handles this.

**Qdrant does NOT rebuild — it upserts**
Qdrant supports incremental upsert natively. Per-doc filtering uses `Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])`.

**Two model tiers: OPENROUTER_MODEL vs OPENROUTER_FAST_MODEL**
`OPENROUTER_FAST_MODEL` is used for: query expansion, contextual compression, QA cache question generation, RAPTOR summarisation, self-RAG grounding, RAGAS evaluation. `OPENROUTER_MODEL` is used only for the final answer and HyDE. Both can point to the same model; splitting them lets you swap in a cheaper/faster model for background tasks.

**Late chunking is per-parent batch**
`embed_children(parent_text, child_texts)` sends one parent + its children as a batch. The Jina API does this natively. The local fallback runs a single forward pass on the full parent and mean-pools token spans for each child.

**SSE event ordering on /query**
`route` → `queries` → `trace` → `context` (×N) → `token` (×M) → `citations` → `done` → `grounding`
The grounding event comes *after* `done` intentionally so the client can render the answer first.

**Streaming ingest vs blocking ingest**
`ingest_pdf()` — blocking, used by the non-streaming POST /ingest path.
`ingest_pdf_streaming()` — async generator, used when `stream=true` form field is set.
Both call the same underlying parse/chunk/embed logic; only the progress reporting differs.

**Sessions are ephemeral**
`SessionStore` is in-memory only. Sessions expire after 1 hour of inactivity (`EXPIRE_SECS = 3600`). On server restart all sessions are lost. If persistence is needed, replace the dict with Redis.

**Auth is optional**
If `API_KEY` env var is empty, `verify_api_key()` returns immediately. Set it to any string to require `X-API-Key: <value>` on all requests.

**ColBERT fallback**
RAGatouille requires `langchain==0.1.x` due to an import of `langchain.retrievers`. If the import fails, `COLBERT_AVAILABLE = False` and `ColBERTStore` automatically falls back to `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking (search returns empty list; RRF just uses dense + BM25).

---

## Adding a new feature — checklist

1. **New config value** → add to `config.py` and `.env.example`
2. **New data shape** → add dataclass to `rag/models.py`
3. **New retrieval step** → implement in `rag/retrieval.py`, thread it through `retrieve()`, add field to `RetrievalTrace`
4. **New SSE event** → emit from `api/routes.py:event_generator()`, document event type in README
5. **New endpoint** → add to `api/routes.py`, add `Depends(verify_api_key)`, document in README

---

## Common tasks

**Add a new embedding model**
Set `EMBEDDING_MODEL` and `EMBEDDING_DIM` in `.env`. Any HuggingFace sentence-transformers model works. For models that need `trust_remote_code=True` the loader in `rag/embeddings.py:_local_model()` already passes it.

**Switch to persistent Qdrant**
```bash
docker run -p 6333:6333 qdrant/qdrant
# .env:
QDRANT_URL=http://localhost:6333
```
No code changes needed.

**Disable expensive features for testing**
```bash
MULTI_QUERY_ENABLED=false
MMR_ENABLED=false
COMPRESSION_ENABLED=false
SELF_RAG_ENABLED=false
```

**Run without an API key (local dev)**
Leave `API_KEY=` empty in `.env`.

**Increase context window**
Raise `TOP_K_PARENTS` (more parents) and `MAX_ANSWER_TOKENS` (longer answer). Also raise `PARENT_CHUNK_TOKENS` before ingesting if you want larger context units — requires re-ingesting the PDF.

---

## Dependencies to know

| Package | Used for | Notes |
|---|---|---|
| `openai` | LLM calls via OpenRouter | OpenAI-compatible client; `base_url=https://openrouter.ai/api/v1` |
| `qdrant-client` | Vector store | In-memory (`:memory:`) or server mode |
| `rank-bm25` | Keyword index | `BM25Okapi`; no incremental add, rebuild each time |
| `ragatouille` | ColBERT | Requires `langchain==0.1.x`; falls back to cross-encoder |
| `sentence-transformers` | Local embeddings + cross-encoder | Lazy-loaded on first use |
| `pdfplumber` | PDF parsing | Built on pdfminer; handles tables + complex layouts |
| `tiktoken` | Token counting | `cl100k_base` encoding (GPT-4 tokeniser, good proxy) |
| `scikit-learn` | KMeans for RAPTOR | `KMeans(n_clusters=N, n_init=10)` |
| `slowapi` | Rate limiting | Wraps limits lib; key = remote IP |
| `fastapi` + `uvicorn` | HTTP server | SSE via `StreamingResponse` |
