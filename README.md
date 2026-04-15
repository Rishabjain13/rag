# RAG PDF

A production-grade Retrieval-Augmented Generation system for PDF documents.

---

## Architecture

### Ingestion

```
PDF
 └─ pdfplumber parser          text + tables (markdown) + headings per page
      └─ section-boundary       split at detected headings first,
         chunking               then overflow by token count
              Parent: 1024 tok  stored for LLM context
              Child:   256 tok  used for search
                   └─ late-chunking embed   Jina v3 API (or local bge-small)
                        each child encoded with full parent context
                             │
                   ┌─────────┼──────────┐
                   ▼         ▼          ▼
              Qdrant       ColBERT    BM25
              (dense)   (RAGatouille) (rank-bm25)
```

### Query

```
User query
    │
    ▼
Adaptive Router ──────────────────────────────────────────────────────────
  cached   → Semantic cache hit → return instantly (~50 ms)
  raptor   → RAPTOR tree search (summary/overview queries)
  hyde     → Hypothetical answer → embed → search
  direct   → Plain embed → search

    │ (hyde / direct path)
    ▼
Multi-query expansion (fast model)
  original + 3 LLM-generated variants
    │
    ▼ (all queries run in parallel)
Triple Hybrid Search
  Dense vector  (Qdrant cosine)     ─┐
  ColBERT MaxSim (RAGatouille)       ├─→ RRF Fusion → Top-15
  BM25 Okapi    (keyword)           ─┘
    │
    ▼
ColBERT Rerank  (or cross-encoder fallback) → Top-5
    │
    ▼
Parent Expansion   child_id → fetch 1024-tok parent, deduplicate
    │
    ▼
MMR                reorder contexts for relevance × diversity
    │
    ▼
Contextual Compression (fast model)   extract only relevant sentences
    │
    ▼
OpenRouter LLM (streaming)   system prompt + citations [p.N] + style control
    │
    ├─→ Structured citations   parse [p.N] markers → {page, heading, doc_id}
    └─→ Self-RAG grounding     fast model scores answer support (0.0 – 1.0)
```

### Background jobs (non-blocking, start after ingest)

```
+2 min   QA Cache
         headings → fast model generates questions → pre-compute answers
         → embed questions → store in Qdrant cache collection
         Common queries now return in ~50 ms

+5 min   RAPTOR Tree
         Layer 0: child chunks
         Layer 1: KMeans cluster summaries (via OpenRouter LLM)
         Layer 2: summaries of summaries
         Layer 3: single document root
         Broad / summary queries use this tree for full-doc context
```

---

## Features

| Feature | Description |
|---|---|
| **Late chunking** | Jina v3 encodes the full parent; child embeddings carry cross-chunk context |
| **pdfplumber parser** | Handles complex layouts, multi-column text, and tables |
| **Table extraction** | Tables converted to markdown rows — embeddable and retrievable |
| **Section-boundary chunking** | Splits at headings first; merges tiny sections; overflows by tokens |
| **Document fingerprinting** | MD5 dedup — re-uploading the same PDF returns instantly |
| **Triple hybrid search** | Dense + ColBERT + BM25 run in parallel, merged with RRF |
| **Multi-query expansion** | 3 OpenRouter LLM query variants improve recall on ambiguous questions |
| **MMR** | Maximal Marginal Relevance prevents redundant context passages |
| **Contextual compression** | Strips irrelevant sentences from each parent chunk before LLM |
| **ColBERT rerank** | Token-level MaxSim reranking; cross-encoder fallback if unavailable |
| **RAPTOR tree** | Recursive summarisation for broad / document-level queries |
| **QA semantic cache** | Pre-computed answers for likely questions; ~50 ms cache hits |
| **Structured citations** | `[p.N]` markers parsed into `{page, heading, doc_id}` arrays |
| **Self-RAG grounding** | Post-generation faithfulness score (0.0 – 1.0) |
| **Conversation history** | Session IDs carry multi-turn context into the LLM |
| **Answer styles** | `concise` · `detailed` · `bullets` via query parameter |
| **Multi-document** | Per-doc Qdrant + BM25 filters; document registry; per-doc delete |
| **RAGAS evaluation** | `POST /evaluate` scores context_recall, precision, faithfulness, relevance |
| **Streaming ingest** | SSE progress events: parsing → chunking → embedding → indexing |
| **API key auth** | `X-API-Key` header guard (disabled when `API_KEY` is empty) |
| **Rate limiting** | slowapi: configurable per-minute limits on query and ingest |
| **File size limit** | Configurable max upload size (default 50 MB) |

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd RAG_PDF
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set OPENROUTER_API_KEY
```

### 3. (Optional) Persistent Qdrant via Docker

```bash
docker run -p 6333:6333 qdrant/qdrant
# Then set in .env:  QDRANT_URL=http://localhost:6333
```

### 4. Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API

### `POST /ingest`

Upload a PDF. Returns when indexing is complete; background jobs (QA cache, RAPTOR) run asynchronously.

```bash
# Standard (JSON response when done)
curl -X POST http://localhost:8000/ingest \
  -F "file=@paper.pdf"

# Streaming progress (SSE)
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@paper.pdf" \
  -F "stream=true"
```

Response:
```json
{
  "status": "ingested",
  "doc_id": "3f2a1b...",
  "parents": 42,
  "children": 168,
  "message": "QA cache and RAPTOR tree building in background."
}
```

Duplicate upload returns:
```json
{ "status": "duplicate", "doc_id": "existing-id", "message": "..." }
```

---

### `POST /query`

Query with streaming SSE. Each event is a JSON object on a `data:` line.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main contribution of this paper?"}'
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | The question |
| `doc_id` | string | null | Scope to a specific document |
| `session_id` | string | null | Enable conversation history |
| `style` | string | `"default"` | `concise` · `detailed` · `bullets` |
| `max_tokens` | int | 1024 | Max answer tokens |
| `stream` | bool | true | Always streamed |

**SSE event types:**

```
{"type": "route",     "route": "direct"}
{"type": "queries",   "queries": ["original", "variant1", "variant2"]}
{"type": "trace",     "dense_hits": 12, "colbert_hits": 9, "bm25_hits": 11,
                      "fused": 15, "reranked": 5, "parents": 4,
                      "mmr": true, "compressed": true}
{"type": "context",   "page": 3, "heading": "Methods", "doc_id": "...", "preview": "..."}
{"type": "token",     "text": "The model achieves "}
{"type": "citations", "citations": [{"page": 3, "heading": "Methods", "doc_id": "..."}]}
{"type": "done"}
{"type": "grounding", "grounded": true, "score": 0.94, "issues": ""}
```

---

### `GET /status`

```json
{
  "ready": true,
  "raptor_ready": false,
  "docs": 2,
  "total_parents": 84,
  "total_children": 336,
  "active_sessions": 1
}
```

---

### `GET /docs`

List all ingested documents.

```json
{
  "documents": [
    {
      "doc_id": "3f2a1b...",
      "filename": "paper.pdf",
      "pages": 12,
      "parents": 42,
      "children": 168,
      "status": "ready",
      "ingested_at": "2026-04-14T10:30:00"
    }
  ]
}
```

---

### `DELETE /docs/{doc_id}`

Remove a single document from all indexes.

---

### `DELETE /collection`

Wipe all documents, indexes, and the cache.

---

### `POST /evaluate`

RAGAS-style offline evaluation. For each test case the system runs full retrieval, generates an answer, then scores it on four metrics using the fast OpenRouter model.

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "query": "What dataset was used?",
        "expected": "The authors used the ImageNet dataset.",
        "doc_id": "optional-doc-id"
      }
    ]
  }'
```

Response:
```json
{
  "aggregate": {
    "context_recall":    0.91,
    "context_precision": 0.87,
    "faithfulness":      0.95,
    "answer_relevance":  0.93,
    "mean_score":        0.915,
    "n_evaluated": 1,
    "n_failed": 0
  },
  "per_case": [...]
}
```

---

### Session endpoints

```bash
GET    /sessions                  # list active session IDs
DELETE /sessions/{session_id}     # delete a session
```

---

## Configuration

All settings are in `.env` (see `.env.example` for full documentation).

Key toggles:

| Variable | Default | Effect |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Main model for generation and HyDE |
| `OPENROUTER_FAST_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Fast model for expansion, compression, grounding, eval |
| `MULTI_QUERY_ENABLED` | `true` | LLM query expansion |
| `MMR_ENABLED` | `true` | Diversity reranking |
| `COMPRESSION_ENABLED` | `true` | Context compression |
| `SELF_RAG_ENABLED` | `true` | Grounding score |
| `API_KEY` | *(empty)* | Auth disabled when empty |
| `QDRANT_URL` | *(empty)* | In-memory when empty |

---

## Project structure

```
RAG_PDF/
├── main.py               FastAPI app, rate limiting, startup
├── config.py             All settings from .env
├── requirements.txt
├── .env.example
│
├── rag/
│   ├── models.py         Data classes (chunks, hits, trace, eval results)
│   ├── embeddings.py     Late chunking — Jina API or local token mean-pooling
│   ├── ingestion.py      PDF parse → section chunks → embed → index
│   ├── stores.py         Qdrant + BM25 + ColBERT + ParentStore + DocRegistry
│   ├── cache.py          Semantic cache lookup / write
│   ├── retrieval.py      Router · multi-query · hybrid search · RRF · MMR · compress
│   ├── generation.py     HyDE · streaming answer · citations · self-RAG · summary
│   ├── sessions.py       Conversation history (in-memory, 1-hr TTL)
│   ├── evaluation.py     RAGAS-style metrics via OpenRouter fast model
│   └── background.py     RAPTOR tree + QA cache (async tasks)
│
└── api/
    └── routes.py         All HTTP endpoints
```

---

## Embedding models

| Model | Where | Dim | Notes |
|---|---|---|---|
| Jina v3 (API) | `JINA_API_KEY` set | 1024 | Native late chunking; recommended |
| `BAAI/bge-small-en-v1.5` | local fallback | 384 | Default; ~33 M params, fast |
| Any HuggingFace model | `EMBEDDING_MODEL` | set `EMBEDDING_DIM` | Must support `SentenceTransformer` |

---

## Concepts

| Concept | Purpose |
|---|---|
| **Small-to-big chunking** | Precise child search → rich 1024-tok parent context for LLM |
| **Late chunking** | Child embeddings carry full-parent attention context |
| **Triple hybrid search** | Dense covers semantics; ColBERT covers token precision; BM25 covers exact terms |
| **RRF fusion** | Merges ranked lists without score normalisation: `score = Σ 1/(60 + rank)` |
| **Multi-query expansion** | 3 LLM variants improve recall on vague or ambiguous queries |
| **MMR** | Prevents 4 near-duplicate parent passages reaching the LLM |
| **Contextual compression** | Cuts irrelevant sentences from each 1024-tok parent before generation |
| **ColBERT rerank** | Token-level MaxSim rescoring; cross-encoder fallback if RAGatouille unavailable |
| **RAPTOR** | Recursive KMeans + summarisation tree; enables full-document context |
| **QA cache** | Pre-computed answers for likely questions; ~50 ms cache hits |
| **HyDE** | Hypothetical document embedding bridges vocabulary gap on abstract queries |
| **Self-RAG** | Post-generation grounding check prevents hallucination going unnoticed |
| **Structured citations** | `[p.N]` markers → typed `{page, heading, doc_id}` array in response |
