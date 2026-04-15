"""
RAGAS-style offline evaluation.

Metrics
───────
  context_recall      Does the retrieved context contain info from the expected answer?
  context_precision   Are the retrieved contexts relevant to the query?
  faithfulness        Is every claim in the generated answer supported by the context?
  answer_relevance    Does the generated answer directly address the question?

All four metrics are rated 0.0 – 1.0 by the fast OpenRouter model.
A single LLM call rates all four to minimize latency and cost.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from openai import AsyncOpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_FAST_MODEL
from rag.models import EvalResult, RetrievedContext

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    return _client


_EVAL_SYSTEM = (
    "You are a precise RAG evaluation assistant. "
    "Rate each metric from 0.0 (worst) to 1.0 (best). "
    "Respond ONLY with a valid JSON object – no extra text, no markdown."
)

_EVAL_PROMPT = """\
Query: {query}

Retrieved context (truncated):
{context}

Expected answer (ground truth):
{expected}

Generated answer:
{answer}

Rate all four metrics and explain briefly:
{{
  "context_recall":    <0.0-1.0>,   // does context contain info from expected?
  "context_precision": <0.0-1.0>,   // are contexts relevant to the query?
  "faithfulness":      <0.0-1.0>,   // is generated answer supported by context?
  "answer_relevance":  <0.0-1.0>,   // does answer address the question?
  "notes":             "<one sentence>"
}}"""


async def evaluate_single(
    query: str,
    expected: str,
    contexts: List[RetrievedContext],
    answer: str,
) -> EvalResult:
    """Score one (query, expected, contexts, answer) tuple."""
    context_text = "\n\n---\n\n".join(
        f"[p.{c.page_num}] {c.parent_text[:400]}" for c in contexts
    )
    prompt = _EVAL_PROMPT.format(
        query=query,
        context=context_text[:2000],
        expected=expected[:400],
        answer=answer[:400],
    )
    try:
        response = await _get_client().chat.completions.create(
            model=OPENROUTER_FAST_MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": _EVAL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return EvalResult(
            query=query,
            context_recall=float(data.get("context_recall", 0)),
            context_precision=float(data.get("context_precision", 0)),
            faithfulness=float(data.get("faithfulness", 0)),
            answer_relevance=float(data.get("answer_relevance", 0)),
            answer=answer,
        )
    except Exception as e:
        logger.warning("Evaluation failed for query '%s': %s", query[:50], e)
        return EvalResult(
            query=query,
            context_recall=0.0, context_precision=0.0,
            faithfulness=0.0, answer_relevance=0.0,
            answer=answer, error=str(e),
        )


async def evaluate_batch(
    test_cases: List[dict],   # each: {query, expected, contexts, answer}
) -> dict:
    """
    Evaluate a list of test cases and return aggregate + per-case results.

    Each test_case dict:
      query    : str
      expected : str
      contexts : List[RetrievedContext]
      answer   : str
    """
    import asyncio
    tasks = [
        evaluate_single(
            tc["query"],
            tc.get("expected", ""),
            tc.get("contexts", []),
            tc.get("answer", ""),
        )
        for tc in test_cases
    ]
    results: List[EvalResult] = await asyncio.gather(*tasks)

    def avg(attr: str) -> float:
        vals = [getattr(r, attr) for r in results if not r.error]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "aggregate": {
            "context_recall":    avg("context_recall"),
            "context_precision": avg("context_precision"),
            "faithfulness":      avg("faithfulness"),
            "answer_relevance":  avg("answer_relevance"),
            "mean_score":        avg("mean_score"),
            "n_evaluated":       sum(1 for r in results if not r.error),
            "n_failed":          sum(1 for r in results if r.error),
        },
        "per_case": [
            {
                "query":             r.query,
                "context_recall":    r.context_recall,
                "context_precision": r.context_precision,
                "faithfulness":      r.faithfulness,
                "answer_relevance":  r.answer_relevance,
                "mean_score":        round(r.mean_score, 4),
                "error":             r.error,
            }
            for r in results
        ],
    }
