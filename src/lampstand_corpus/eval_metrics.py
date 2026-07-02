"""Retrieval-eval metric math (pure functions — no DBs, no numpy).

Definitions deliberately MATCH the app's RetrievalEvalTests semantics
(lampstand-ios ``Tests/RetrievalEvalTests.swift``) so corpus-side and app-side
numbers are comparable:

  * recall@k  — the fraction of queries with AT LEAST ONE relevant chunk in the
    top k ("a case passes@k if ANY gold matches a hit in the top-k"). This is a
    hit-rate, not a set-completeness recall; named recall@k for continuity with
    the app report.
  * MRR       — mean reciprocal rank of the FIRST relevant hit within the top
    ``MRR_DEPTH`` (20, the app's topK); 0 when no relevant hit appears there.
  * nDCG@k    — standard binary-gain nDCG: DCG = Σ 1/log2(rank+1) over relevant
    hits in the top k, normalized by the ideal DCG for min(|relevant|, k) hits.
    (The app does not compute nDCG; this is the corpus-side addition.)

Everything here is deterministic and order-driven: a ranking is a list of chunk
ids, relevance is set membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

# The app evaluates hybridContext(topK: 20); MRR is capped at the same depth.
MRR_DEPTH = 20
RECALL_KS = (5, 10, 20)
NDCG_K = 10


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> bool:
    """True iff any relevant id appears in the first ``k`` ranked ids."""
    return any(cid in relevant for cid in ranked[:k])


def first_rank(ranked: list[str], relevant: set[str], limit: int) -> int | None:
    """1-based rank of the first relevant id within ``limit``, else None."""
    for i, cid in enumerate(ranked[:limit], start=1):
        if cid in relevant:
            return i
    return None


def reciprocal_rank(ranked: list[str], relevant: set[str], limit: int = MRR_DEPTH) -> float:
    """1/rank of the first relevant hit within ``limit``; 0.0 if absent."""
    r = first_rank(ranked, relevant, limit)
    return 0.0 if r is None else 1.0 / r


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = NDCG_K) -> float:
    """Binary-gain nDCG@k. 0.0 when ``relevant`` is empty."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / log2(i + 1)
        for i, cid in enumerate(ranked[:k], start=1)
        if cid in relevant
    )
    ideal = sum(1.0 / log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


@dataclass
class Metrics:
    """Aggregate metrics over a set of queries (each query weighs equally)."""

    count: int = 0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "recall_at_20": self.recall_at_20,
            "mrr": self.mrr,
            "ndcg_at_10": self.ndcg_at_10,
        }


def aggregate(results: list[tuple[list[str], set[str]]]) -> Metrics:
    """Aggregate ``(ranked_ids, relevant_ids)`` pairs into one Metrics block."""
    n = len(results)
    if n == 0:
        return Metrics()
    hits = {k: 0 for k in RECALL_KS}
    rr_sum = 0.0
    ndcg_sum = 0.0
    for ranked, relevant in results:
        for k in RECALL_KS:
            if hit_at_k(ranked, relevant, k):
                hits[k] += 1
        rr_sum += reciprocal_rank(ranked, relevant)
        ndcg_sum += ndcg_at_k(ranked, relevant)
    return Metrics(
        count=n,
        recall_at_5=hits[5] / n,
        recall_at_10=hits[10] / n,
        recall_at_20=hits[20] / n,
        mrr=rr_sum / n,
        ndcg_at_10=ndcg_sum / n,
    )


def pairwise_wins(
    outcomes: list[tuple[int | None, int | None]],
) -> dict:
    """Hard-negative pairwise accuracy.

    Each outcome is ``(relevant_rank, hard_negative_rank)`` where a rank is the
    1-based fused rank of that chunk in the arm's full ranking, or None when the
    chunk was not retrieved at all. A WIN requires the relevant chunk to be
    retrieved AND to outrank the hard negative (an absent hard negative counts
    as outranked). An absent relevant chunk is always a LOSS.
    """
    wins = 0
    for rel, neg in outcomes:
        if rel is None:
            continue
        if neg is None or rel < neg:
            wins += 1
    n = len(outcomes)
    return {
        "count": n,
        "wins": wins,
        "win_rate": (wins / n) if n else 0.0,
    }
