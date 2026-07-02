"""Retrieval-eval metric math (pure functions, no DBs).

Definitions must match the app's RetrievalEvalTests semantics: recall@k is a
per-query any-relevant hit rate; MRR is first-relevant reciprocal rank capped
at depth 20; nDCG@10 is standard binary-gain nDCG.
"""

from __future__ import annotations

from math import log2

from lampstand_corpus import eval_metrics as em


# --- hit / rank primitives -----------------------------------------------------
def test_hit_at_k_respects_window():
    ranked = ["a", "b", "c", "d"]
    assert em.hit_at_k(ranked, {"c"}, 3)
    assert not em.hit_at_k(ranked, {"c"}, 2)
    assert not em.hit_at_k(ranked, {"z"}, 4)


def test_first_rank_is_one_based_and_limited():
    ranked = ["a", "b", "c"]
    assert em.first_rank(ranked, {"a"}, 3) == 1
    assert em.first_rank(ranked, {"c"}, 3) == 3
    assert em.first_rank(ranked, {"c"}, 2) is None


def test_reciprocal_rank_zero_when_absent_within_depth():
    ranked = [f"x{i}" for i in range(30)] + ["gold"]
    # gold is at rank 31 — beyond MRR_DEPTH(20) — so RR is 0, like the app.
    assert em.reciprocal_rank(ranked, {"gold"}) == 0.0
    assert em.reciprocal_rank(["gold"], {"gold"}) == 1.0


# --- nDCG -----------------------------------------------------------------------
def test_ndcg_perfect_ranking_is_one():
    ranked = ["r1", "r2", "x", "y"]
    assert em.ndcg_at_k(ranked, {"r1", "r2"}, k=10) == 1.0


def test_ndcg_single_relevant_at_rank_2():
    ranked = ["x", "r", "y"]
    expected = (1.0 / log2(3)) / 1.0  # DCG at rank 2 over ideal rank 1
    assert abs(em.ndcg_at_k(ranked, {"r"}, k=10) - expected) < 1e-12


def test_ndcg_ideal_caps_at_k():
    # 15 relevant, k=10: ideal is the best 10 — a full top-10 of relevant = 1.0.
    relevant = {f"r{i}" for i in range(15)}
    ranked = [f"r{i}" for i in range(10)]
    assert em.ndcg_at_k(ranked, relevant, k=10) == 1.0


def test_ndcg_empty_relevant_is_zero():
    assert em.ndcg_at_k(["a"], set(), k=10) == 0.0


# --- aggregation -----------------------------------------------------------------
def test_aggregate_mixed_cases():
    results = [
        (["g", "x"], {"g"}),            # hit@5/10/20, rr=1, ndcg=1
        (["x"] * 20 + ["g"], {"g"}),    # miss everywhere (rank 21)
    ]
    m = em.aggregate(results)
    assert m.count == 2
    assert m.recall_at_5 == 0.5
    assert m.recall_at_20 == 0.5
    assert m.mrr == 0.5
    assert m.ndcg_at_10 == 0.5


def test_aggregate_empty_is_zeroes():
    m = em.aggregate([])
    assert m.count == 0 and m.mrr == 0.0


# --- hard-negative pairwise --------------------------------------------------------
def test_pairwise_wins_rules():
    outcomes = [
        (1, 2),        # win: relevant outranks negative
        (3, 1),        # loss
        (2, None),     # win: negative not retrieved
        (None, 1),     # loss: relevant not retrieved
        (None, None),  # loss: relevant not retrieved
    ]
    res = em.pairwise_wins(outcomes)
    assert res["count"] == 5
    assert res["wins"] == 2
    assert abs(res["win_rate"] - 0.4) < 1e-12
