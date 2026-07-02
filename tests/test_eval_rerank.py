"""Cross-encoder rerank harness tests — no network, no torch.

Verifies the pure re-ranking logic (window re-sort, tail preservation, stable
tie-break, header/raw text selection) and the delta computation with a FAKE
cross-encoder, so the quality-gate math is covered without downloading a model.
The actual cross-encoder scoring is a thin ``model.predict`` call exercised in
the (network-gated) CLI run, not here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus import eval_rerank as rr


class FakeModel:
    """Stand-in RerankModel: scores by a caller-supplied text->score map."""

    key = "fake"
    hf_id = "fake/fake"
    license = "test"
    arch = "test"
    revision = "testrev"

    def __init__(self, score_of):
        self._score_of = score_of
        self.seen: list[tuple[str, str]] = []

    def score_pairs(self, pairs):
        self.seen.extend(pairs)
        return [self._score_of(passage) for _q, passage in pairs]


@pytest.fixture
def texts(tmp_path: Path) -> rr.ChunkTextIndex:
    db = tmp_path / "embeddings.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE chunk (id TEXT PRIMARY KEY, header TEXT, text TEXT, "
        "indexed INTEGER)")
    conn.executemany("INSERT INTO chunk VALUES (?,?,?,?)", [
        ("a", "Genesis 1:1 — ", "in the beginning", 1),
        ("b", "John 3:16 — ", "for God so loved", 1),
        ("c", "", "no header here", 1),
    ])
    conn.commit()
    conn.close()
    return rr.ChunkTextIndex(db)


def test_pair_text_header_vs_raw(texts):
    assert texts.pair_text("a", with_header=True) == "Genesis 1:1 — in the beginning"
    assert texts.pair_text("a", with_header=False) == "in the beginning"
    # No header -> both variants are the raw text.
    assert texts.pair_text("c", with_header=True) == "no header here"


def test_rerank_resorts_window_and_keeps_tail(texts):
    # Fused order a,b,c; reranker prefers b (highest), then c, then a.
    model = FakeModel(lambda p: {"in the beginning": 0.1, "for God so loved": 0.9,
                                 "no header here": 0.5}[p])
    out = rr.rerank_ids(["a", "b", "c"], "q", model, texts,
                        with_header=False, k=2)
    # k=2 reranks only [a,b] -> [b,a]; c is the untouched tail.
    assert out == ["b", "a", "c"]


def test_rerank_full_window(texts):
    model = FakeModel(lambda p: {"in the beginning": 0.1, "for God so loved": 0.9,
                                 "no header here": 0.5}[p])
    out = rr.rerank_ids(["a", "b", "c"], "q", model, texts,
                        with_header=False, k=3)
    assert out == ["b", "c", "a"]


def test_rerank_stable_on_score_ties(texts):
    # All equal scores -> original fused order preserved (stable sort).
    model = FakeModel(lambda p: 1.0)
    out = rr.rerank_ids(["a", "b", "c"], "q", model, texts, with_header=False, k=3)
    assert out == ["a", "b", "c"]


def test_rerank_uses_header_variant(texts):
    model = FakeModel(lambda p: 1.0 if p.startswith("Genesis") else 0.0)
    out = rr.rerank_ids(["b", "a"], "q", model, texts, with_header=True, k=2)
    # Only the header-prefixed Genesis passage scores 1.0 -> ranks first.
    assert out[0] == "a"
    assert any(p.startswith("Genesis") for _q, p in model.seen)


def test_rerank_empty_window_is_identity(texts):
    model = FakeModel(lambda p: 1.0)
    assert rr.rerank_ids([], "q", model, texts, with_header=False, k=5) == []


def test_compute_deltas_signs_and_categories():
    baseline = {
        "overall": {"recall_at_5": 0.2, "recall_at_10": 0.3, "recall_at_20": 0.4,
                    "mrr": 0.10, "ndcg_at_10": 0.05, "count": 10},
        "per_category": {
            "crossref": {"recall_at_5": 0.2, "recall_at_10": 0.3,
                         "recall_at_20": 0.4, "mrr": 0.1, "ndcg_at_10": 0.05,
                         "count": 5},
            "hardneg": {"recall_at_5": 0.0, "recall_at_10": 0.0,
                        "recall_at_20": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0,
                        "count": 5},
        },
        "hardneg_pairwise": {"win_rate": 0.9, "wins": 9, "count": 10},
    }
    arm = {
        "model_key": "fake", "variant": "raw",
        "overall": {"recall_at_5": 0.25, "recall_at_10": 0.33, "recall_at_20": 0.4,
                    "mrr": 0.14, "ndcg_at_10": 0.06, "count": 10},
        "per_category": {
            "crossref": {"recall_at_5": 0.3, "recall_at_10": 0.36,
                         "recall_at_20": 0.42, "mrr": 0.15, "ndcg_at_10": 0.07,
                         "count": 5},
            "hardneg": {"recall_at_5": 0.0, "recall_at_10": 0.0,
                        "recall_at_20": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0,
                        "count": 5},
        },
        "hardneg_pairwise": {"win_rate": 0.95, "wins": 19, "count": 20},
    }
    deltas = rr._compute_deltas(baseline, [arm])
    assert len(deltas) == 1
    d = deltas[0]
    assert d["overall"]["mrr"] == pytest.approx(0.04)
    # hardneg is excluded from the per-category delta (scored separately).
    assert "hardneg" not in d["per_category"]
    assert d["per_category"]["crossref"]["mrr"] == pytest.approx(0.05)
    assert d["hardneg_win_rate_delta"] == pytest.approx(0.05)


def test_verdict_ship_on_category_lift():
    from lampstand_corpus.eval_rerank_report import _verdict

    results = {
        "deltas": [{
            "model_key": "m", "variant": "raw",
            "overall": {k: 0.0 for k in rr._METRIC_KEYS},
            "per_category": {
                "commentary-anchor": {**{k: 0.0 for k in rr._METRIC_KEYS},
                                      "mrr": 0.05},
                "crossref": {k: 0.0 for k in rr._METRIC_KEYS},
            },
            "hardneg_win_rate": 1.0, "hardneg_win_rate_delta": 0.0,
        }],
    }
    verdict, _best, _lines = _verdict(results)
    assert verdict == "SHIP"


def test_verdict_hold_on_marginal():
    from lampstand_corpus.eval_rerank_report import _verdict

    results = {
        "deltas": [{
            "model_key": "m", "variant": "raw",
            "overall": {**{k: 0.0 for k in rr._METRIC_KEYS}, "mrr": 0.005},
            "per_category": {
                "commentary-anchor": {**{k: 0.0 for k in rr._METRIC_KEYS},
                                      "mrr": 0.01},
                "crossref": {**{k: 0.0 for k in rr._METRIC_KEYS}, "mrr": -0.01},
            },
            "hardneg_win_rate": 1.0, "hardneg_win_rate_delta": 0.0,
        }],
    }
    verdict, _best, _lines = _verdict(results)
    assert verdict == "HOLD"
