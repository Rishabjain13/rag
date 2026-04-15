"""
Embedding layer with Late Chunking support.

Rate-limiting strategy (v2.3)
──────────────────────────────
  Token-bucket limiter    Proactively paces Jina requests at JINA_RPM/min so
                          429s are avoided before they happen.  Starts full
                          (instant burst for the first JINA_RPM calls), then
                          refills at JINA_RPM/60 tokens per second.

  Concurrency semaphore   JINA_CONCURRENCY caps simultaneous in-flight requests.

  Jitter backoff          Any retry uses random jitter to prevent thundering herd
                          when multiple concurrent calls fail simultaneously.

  Circuit breaker         After JINA_CB_THRESHOLD consecutive failures, Jina is
                          skipped for the rest of the ingest and the local model
                          takes over immediately — no more wasted retries.

  Fast local fallback     sentence-transformers with manual token mean-pooling.
                          No API calls, no rate limits.  Good quality, ~1-2 s/parent
                          on CPU, faster on GPU.

Priority order
──────────────
  1. Jina AI API  – native late-chunking (if JINA_API_KEY set and circuit closed)
  2. Local sentence-transformers – manual late-chunking via token mean-pooling
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import List

import threading

from config import (
    JINA_API_KEY, JINA_MODEL,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    JINA_CONCURRENCY, JINA_RPM,
    JINA_FOR_INGEST,
)

logger = logging.getLogger(__name__)


# ── Circuit breaker ───────────────────────────────────────────────────────────
# After this many consecutive Jina failures, stop trying Jina for this session.
_JINA_CB_THRESHOLD = 5
_jina_failures     = 0      # consecutive failure counter
_jina_cb_open      = False  # True = skip Jina, use local only


def _jina_ok() -> bool:
    """Return False once the circuit breaker trips."""
    return not _jina_cb_open


def _jina_record_success() -> None:
    global _jina_failures, _jina_cb_open
    _jina_failures = 0
    _jina_cb_open  = False


def _jina_record_failure() -> None:
    global _jina_failures, _jina_cb_open
    _jina_failures += 1
    if _jina_failures >= _JINA_CB_THRESHOLD:
        if not _jina_cb_open:
            logger.warning(
                "Jina: %d consecutive failures — switching to local model for this session.",
                _jina_failures,
            )
        _jina_cb_open = True


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

class _TokenBucket:
    """
    Proactive rate limiter.  Starts full (burst = JINA_RPM calls instant),
    then throttles to JINA_RPM / 60 calls per second.
    `acquire()` sleeps only as long as needed — no wasted retry cycles.
    """

    def __init__(self, rpm: int) -> None:
        self._rate     = rpm / 60.0
        self._capacity = float(rpm)
        self._tokens   = float(rpm)   # start full → instant burst
        self._last_ts  = 0.0
        self._lock     = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_event_loop()
            now  = loop.time()
            if self._last_ts == 0.0:
                self._last_ts = now

            elapsed       = now - self._last_ts
            self._tokens  = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_ts = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            wait = (1.0 - self._tokens) / self._rate
            logger.debug("Jina rate limiter: waiting %.2f s", wait)
            await asyncio.sleep(wait)
            self._tokens  = 0.0
            self._last_ts = asyncio.get_event_loop().time()


# ── Singletons (lazy, created inside the running event loop) ──────────────────
_bucket:    _TokenBucket | None      = None
_semaphore: asyncio.Semaphore | None = None
_http_client                         = None


def _get_bucket() -> _TokenBucket:
    global _bucket
    if _bucket is None:
        _bucket = _TokenBucket(JINA_RPM)
    return _bucket


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(JINA_CONCURRENCY)
    return _semaphore


def _get_http_client():
    global _http_client
    if _http_client is None:
        import httpx
        _http_client = httpx.AsyncClient(
            timeout=120,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=JINA_CONCURRENCY + 4),
        )
    return _http_client


# ── Lazy-loaded local model (thread-safe singleton) ───────────────────────────
_st_model      = None
_st_model_lock = threading.Lock()   # prevents multiple threads loading simultaneously


def _local_model():
    global _st_model
    if _st_model is not None:        # fast path — no lock needed once loaded
        return _st_model
    with _st_model_lock:             # only one thread loads
        if _st_model is None:        # double-check inside lock
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model: %s", EMBEDDING_MODEL)
            _st_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _st_model


# ── Jina API core ─────────────────────────────────────────────────────────────

async def _jina_post(payload: dict, retries: int = 2) -> dict:
    """
    POST to Jina API with token-bucket pacing and jittered retry.

    - Rate bucket acquired BEFORE each attempt (proactive 429 prevention).
    - On 429: jittered backoff (rare with bucket; handles clock-skew edge cases).
    - Raises RuntimeError after all retries so callers fall through to local.
    """
    client   = _get_http_client()
    delay    = 1.0
    last_err = None

    for attempt in range(retries):
        await _get_bucket().acquire()
        try:
            r = await client.post("https://api.jina.ai/v1/embeddings", json=payload)
            if r.status_code == 429:
                jitter = random.uniform(0.0, delay * 0.4)
                wait   = delay + jitter
                logger.warning(
                    "Jina 429 (attempt %d/%d) – backing off %.1f s",
                    attempt + 1, retries, wait,
                )
                await asyncio.sleep(wait)
                delay *= 2
                continue
            r.raise_for_status()
            _jina_record_success()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                jitter = random.uniform(0.0, delay * 0.4)
                await asyncio.sleep(delay + jitter)
                delay *= 2

    _jina_record_failure()
    raise RuntimeError(f"Jina API failed after {retries} attempts: {last_err}")


async def _jina_embed(texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
    async with _get_semaphore():
        data = await _jina_post({"model": JINA_MODEL, "input": texts, "task": task})
    return [d["embedding"] for d in data["data"]]


async def _jina_late_chunk(parent_text: str, child_texts: List[str]) -> List[List[float]]:
    """
    Jina native late-chunking.  Raises on any failure — caller falls back to local.
    (Previously fell back to _jina_embed which is also a Jina call and doubles
    the wasted time when Jina is consistently failing.)
    """
    async with _get_semaphore():
        data = await _jina_post({
            "model": JINA_MODEL,
            "input": [parent_text],
            "task": "retrieval.passage",
            "late_chunking": True,
            "segments": [{"text": c} for c in child_texts],
        })
    return [d["embedding"] for d in data["data"]]


# ── Local late-chunking ───────────────────────────────────────────────────────

def _late_chunk_local(parent_text: str, child_texts: List[str]) -> List[List[float]]:
    """
    Manual late-chunking via sentence-transformers:
      1. Forward-pass the full parent → token hidden states
      2. Proportionally map each child's token count to a span
      3. Mean-pool that span → child embedding
    """
    import torch

    model     = _local_model()
    tokenizer = model.tokenizer

    parent_enc = tokenizer(
        parent_text,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(tokenizer, "model_max_length", 512),
        padding=False,
    )

    with torch.no_grad():
        out        = model[0].auto_model(**parent_enc, output_hidden_states=False)
        token_embs = out.last_hidden_state[0]   # (seq_len, hidden_dim)

    seq_len = token_embs.shape[0]

    child_tok_lens = [
        max(1, len(tokenizer.encode(c, add_special_tokens=False)))
        for c in child_texts
    ]
    total  = sum(child_tok_lens)
    usable = max(1, seq_len - 2)

    embeddings: List[List[float]] = []
    cursor = 1
    for tok_len in child_tok_lens:
        span = max(1, round(tok_len * usable / total))
        end  = min(cursor + span, seq_len - 1)
        emb  = token_embs[cursor:end].mean(dim=0)
        embeddings.append(emb.cpu().float().numpy().tolist())
        cursor = end

    return embeddings


def _embed_local(texts: List[str]) -> List[List[float]]:
    vecs = _local_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


def _embed_query_local(query: str) -> List[float]:
    model = _local_model()
    try:
        vec = model.encode([query], normalize_embeddings=True,
                           prompt_name="query", show_progress_bar=False)
    except TypeError:
        vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return vec[0].tolist()


# ── Public interface ──────────────────────────────────────────────────────────

async def embed_children(parent_text: str, child_texts: List[str]) -> List[List[float]]:
    """
    Late-chunk embed child chunks using full parent context.

    JINA_FOR_INGEST=false (recommended for large PDFs) skips Jina entirely and
    uses the local model — no rate limits, no API calls, much faster at scale.
    """
    if not child_texts:
        return []
    if JINA_API_KEY and JINA_FOR_INGEST and _jina_ok():
        try:
            return await _jina_late_chunk(parent_text, child_texts)
        except Exception as e:
            logger.warning("Jina failed: %s — local fallback", e)
    # Use _embed_local (model.encode()) — thread-safe, no raw tokenizer access.
    # _late_chunk_local accesses the Rust tokenizer directly which is not
    # thread-safe under concurrent run_in_executor calls (causes "Already borrowed").
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_local, child_texts)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed arbitrary texts (RAPTOR, QA cache, etc.)."""
    if not texts:
        return []
    if JINA_API_KEY and JINA_FOR_INGEST and _jina_ok():
        try:
            return await _jina_embed(texts, task="retrieval.passage")
        except Exception as e:
            logger.warning("Jina failed: %s — local fallback", e)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_local, texts)


async def embed_query(query: str) -> List[float]:
    """Embed a search query."""
    if JINA_API_KEY and JINA_FOR_INGEST and _jina_ok():
        try:
            return (await _jina_embed([query], task="retrieval.query"))[0]
        except Exception as e:
            logger.warning("Jina failed: %s — local fallback", e)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_query_local, query)


def get_embedding_dim() -> int:
    if JINA_API_KEY and JINA_FOR_INGEST:
        return 1024
    try:
        return _local_model().get_sentence_embedding_dimension() or EMBEDDING_DIM
    except Exception:
        return EMBEDDING_DIM
