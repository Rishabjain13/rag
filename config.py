"""Centralised configuration – loaded from .env"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenRouter (main model) ───────────────────────────────
OPENROUTER_API_KEY: str  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Main model – final answer generation and HyDE
OPENROUTER_MODEL: str    = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

# ── Fast model (query expansion, compression, grounding, RAPTOR, QA cache) ──
# Can point to a completely different provider.
# Recommended: Groq (free tier: 14,400 req/day, no CoT leakage)
#   FAST_MODEL_API_KEY=<groq_key>
#   FAST_MODEL_BASE_URL=https://api.groq.com/openai/v1
#   OPENROUTER_FAST_MODEL=llama-3.1-8b-instant
# If FAST_MODEL_API_KEY / FAST_MODEL_BASE_URL are not set they fall back to
# the main OpenRouter key/URL above.
FAST_MODEL_API_KEY: str  = os.getenv("FAST_MODEL_API_KEY", "") or OPENROUTER_API_KEY
FAST_MODEL_BASE_URL: str = os.getenv("FAST_MODEL_BASE_URL", "") or OPENROUTER_BASE_URL
OPENROUTER_FAST_MODEL: str = os.getenv("OPENROUTER_FAST_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

# ── Embeddings ────────────────────────────────────────────
JINA_API_KEY: str    = os.getenv("JINA_API_KEY", "")
JINA_MODEL: str      = os.getenv("JINA_MODEL", "jina-embeddings-v3")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM: int   = int(os.getenv("EMBEDDING_DIM", "384"))

# ── Qdrant ────────────────────────────────────────────────
QDRANT_URL: str                = os.getenv("QDRANT_URL", "")
QDRANT_COLLECTION: str         = os.getenv("QDRANT_COLLECTION", "rag_children")
QDRANT_CACHE_COLLECTION: str   = os.getenv("QDRANT_CACHE_COLLECTION", "rag_cache")
QDRANT_RAPTOR_COLLECTION: str  = os.getenv("QDRANT_RAPTOR_COLLECTION", "rag_raptor")

# ── Chunking ──────────────────────────────────────────────
PARENT_CHUNK_TOKENS: int  = int(os.getenv("PARENT_CHUNK_TOKENS", "1024"))
CHILD_CHUNK_TOKENS: int   = int(os.getenv("CHILD_CHUNK_TOKENS", "256"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "32"))

# ── Retrieval ─────────────────────────────────────────────
TOP_K_SEARCH: int  = int(os.getenv("TOP_K_SEARCH", "15"))
TOP_K_RERANK: int  = int(os.getenv("TOP_K_RERANK", "5"))
TOP_K_PARENTS: int = int(os.getenv("TOP_K_PARENTS", "4"))
RRF_K: int         = int(os.getenv("RRF_K", "60"))

# ── Multi-query expansion ─────────────────────────────────
MULTI_QUERY_ENABLED: bool = os.getenv("MULTI_QUERY_ENABLED", "true").lower() == "true"
MULTI_QUERY_COUNT: int    = int(os.getenv("MULTI_QUERY_COUNT", "3"))

# ── MMR ───────────────────────────────────────────────────
MMR_ENABLED: bool  = os.getenv("MMR_ENABLED", "true").lower() == "true"
MMR_LAMBDA: float  = float(os.getenv("MMR_LAMBDA", "0.7"))  # 1.0 = pure relevance

# ── Contextual compression ────────────────────────────────
COMPRESSION_ENABLED: bool = os.getenv("COMPRESSION_ENABLED", "true").lower() == "true"

# ── Self-RAG grounding check ──────────────────────────────
SELF_RAG_ENABLED: bool = os.getenv("SELF_RAG_ENABLED", "true").lower() == "true"

# ── ColBERT ───────────────────────────────────────────────
COLBERT_MODEL: str      = os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")
COLBERT_INDEX_PATH: str = os.getenv("COLBERT_INDEX_PATH", "./colbert_index")

# ── RAPTOR ────────────────────────────────────────────────
RAPTOR_CLUSTER_SIZE: int = int(os.getenv("RAPTOR_CLUSTER_SIZE", "5"))
RAPTOR_MAX_LEVELS: int   = int(os.getenv("RAPTOR_MAX_LEVELS", "3"))

# ── Semantic cache ────────────────────────────────────────
CACHE_THRESHOLD: float = float(os.getenv("CACHE_THRESHOLD", "0.92"))

# ── Generation ────────────────────────────────────────────
MAX_ANSWER_TOKENS: int = int(os.getenv("MAX_ANSWER_TOKENS", "1024"))

# ── Security ──────────────────────────────────────────────
API_KEY: str           = os.getenv("API_KEY", "")           # empty = no auth
MAX_FILE_SIZE_MB: int  = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
RATE_LIMIT_QUERY: str  = os.getenv("RATE_LIMIT_QUERY", "30/minute")
RATE_LIMIT_INGEST: str = os.getenv("RATE_LIMIT_INGEST", "5/minute")

# ── Jina rate limiting ────────────────────────────────────
# JINA_RPM: requests per minute your Jina tier allows.
#   Free tier  → 10 RPM   |   Paid tier → 60+ RPM
# The token-bucket limiter proactively paces requests so 429s never occur.
JINA_RPM: int         = int(os.getenv("JINA_RPM", "10"))
JINA_CONCURRENCY: int = int(os.getenv("JINA_CONCURRENCY", "2"))
# JINA_FOR_INGEST: use Jina for embed_children() during ingestion.
#   true  → Jina native late-chunking (higher quality, but RPM-limited)
#   false → local model for ingestion only (no rate limits, fast for large PDFs)
#           Jina still used for embed_query() and embed_texts() when key is set.
JINA_FOR_INGEST: bool = os.getenv("JINA_FOR_INGEST", "false").lower() == "true"

# ── Persistence (for use with Docker Qdrant) ───────────────
# BM25, ParentStore, DocRegistry are in-memory. When QDRANT_URL
# is set they are also saved to disk so the app recovers fully
# after a restart without re-ingesting.
PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./persist")

# ── Paths ─────────────────────────────────────────────────
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
