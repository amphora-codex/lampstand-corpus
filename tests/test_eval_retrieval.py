"""App-parity retrieval runner tests — tiny synthetic index, hand-checked math.

Verifies the BM25 scorer (Lucene non-negative IDF + Okapi term score), the
per-type balance, the Scripture translation dedup, the RRF fusion semantics
(asymmetric list depths, anchor tie-break), and the raw-rank-exact dense
slicing used by the sweep cache.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from lampstand_corpus import eval_retrieval as er
from lampstand_corpus.embeddings import tokenize

DIM = 4  # tiny test dim; EvalIndex reads it from meta, never hard-codes 384

# (id, resource_type, source, anchor, book, chapter, vs, ve, text)
_CHUNKS = [
    ("c_com1", "commentary", "henry", "henry:GEN.1.1#p1#0", "GEN", 1, 1, 1,
     "faith and works discussed at length here"),
    ("f_wsc33", "confession", "wsc", "WSC 33", None, None, None, None,
     "justification received by faith alone"),
    ("s_bsb1", "scripture", "bsb", "bsb:ROM 3:21-31", "ROM", 3, 21, 31,
     "by faith apart from works of the law faith faith"),
    ("s_kjv1", "scripture", "kjv", "kjv:ROM 3:21-31", "ROM", 3, 21, 31,
     "by faith without the deeds of the law faith faith"),
]


@pytest.fixture
def index(tmp_path: Path) -> er.EvalIndex:
    db = tmp_path / "embeddings.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE chunk (id TEXT PRIMARY KEY, resource_type TEXT, source TEXT,"
        " anchor TEXT, book TEXT, chapter INTEGER, verse_start INTEGER,"
        " verse_end INTEGER, text TEXT);"
        "CREATE TABLE embedding (chunk_id TEXT PRIMARY KEY, dim INTEGER,"
        " vector BLOB);"
        "CREATE TABLE bm25_doc (chunk_id TEXT PRIMARY KEY, length INTEGER);"
        "CREATE TABLE bm25_term (term_id INTEGER PRIMARY KEY, term TEXT UNIQUE,"
        " doc_freq INTEGER);"
        "CREATE TABLE bm25_posting (term_id INTEGER, chunk_id TEXT,"
        " term_freq INTEGER, PRIMARY KEY (term_id, chunk_id));"
        "CREATE TABLE bm25_stats (key TEXT PRIMARY KEY, value REAL);"
    )
    conn.executemany("INSERT INTO chunk VALUES (?,?,?,?,?,?,?,?,?)", _CHUNKS)

    # BM25 index built with the pipeline's own tokenizer (as build_embeddings does).
    postings: dict[str, dict[str, int]] = {}
    lengths: dict[str, int] = {}
    for cid, *_rest, text in [(c[0], c[8]) for c in _CHUNKS]:
        toks = tokenize(text)
        lengths[cid] = len(toks)
        for t in toks:
            postings.setdefault(t, {}).setdefault(cid, 0)
            postings[t][cid] += 1
    conn.executemany("INSERT INTO bm25_doc VALUES (?,?)", sorted(lengths.items()))
    terms = sorted(postings)
    conn.executemany("INSERT INTO bm25_term VALUES (?,?,?)", [
        (tid, t, len(postings[t])) for tid, t in enumerate(terms)])
    conn.executemany("INSERT INTO bm25_posting VALUES (?,?,?)", [
        (tid, cid, tf)
        for tid, t in enumerate(terms)
        for cid, tf in sorted(postings[t].items())])
    n_docs = len(_CHUNKS)
    avgdl = sum(lengths.values()) / n_docs
    conn.executemany("INSERT INTO bm25_stats VALUES (?,?)", [
        ("n_docs", float(n_docs)), ("avgdl", avgdl),
        ("k1", 1.5), ("b", 0.75)])

    # Unit vectors: c_com1 -> e1, f_wsc33 -> e2, s_bsb1 -> e3, s_kjv1 ~ e3-ish.
    vecs = {
        "c_com1": [1, 0, 0, 0],
        "f_wsc33": [0, 1, 0, 0],
        "s_bsb1": [0, 0, 1, 0],
        "s_kjv1": [0, 0, 0.8, 0.6],
    }
    conn.executemany("INSERT INTO embedding VALUES (?,?,?)", [
        (cid, DIM, np.asarray(v, dtype="<f4").tobytes()) for cid, v in vecs.items()])
    conn.executemany("INSERT INTO meta VALUES (?,?)", [
        ("embedding_dim", str(DIM)), ("model_revision", "testrev")])
    conn.commit()
    conn.close()
    idx = er.EvalIndex(db)
    yield idx
    idx.close()


# --- BM25 math -------------------------------------------------------------------
def test_bm25_score_matches_hand_calculation(index):
    """Single-term score == IDF(term) * Okapi term factor, app formulas."""
    scores = index.bm25_scores("justification", [])
    i = index.id_to_idx["f_wsc33"]
    df, n = 1, 4
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
    dl = index.doc_len[i]
    norm = 1.5 * (1 - 0.75 + 0.75 * dl / index.avgdl)
    expected = idf * (1 * 2.5) / (1 + norm)
    assert scores[i] == pytest.approx(expected, rel=1e-12)
    # No other doc contains the term.
    others = [j for j in range(index.n_chunks) if j != i]
    assert all(scores[j] == 0.0 for j in others)


def test_bm25_query_terms_are_deduplicated(index):
    """'faith faith' must score identically to 'faith' (app tokenizes to a Set)."""
    once = index.bm25_scores("faith", []).copy()
    twice = index.bm25_scores("faith faith", []).copy()
    assert np.array_equal(once, twice)


def test_bm25_exclusion_removes_candidates(index):
    i = index.id_to_idx["f_wsc33"]
    scores = index.bm25_scores("justification", [i])
    assert scores[i] == 0.0


def test_bm25_type_lists_balance_per_type(index):
    lists = index.bm25_type_lists("faith", [])
    types = {er.RESOURCE_TYPES[t]: [index.ids[i] for i, _s in v]
             for t, v in lists.items() if v}
    assert "c_com1" in types["commentary"]
    assert "f_wsc33" in types["confession"]
    assert set(types["scripture"]) == {"s_bsb1", "s_kjv1"}


# --- ranking assembly ---------------------------------------------------------------
def test_bm25_ranking_dedups_scripture_translations(index):
    lists = index.bm25_type_lists("faith", [])
    ranked = er.bm25_ranking(lists, per_type=20, dedup_key=index.dedup_key)
    ids = [index.ids[i] for i in ranked]
    # Both translations cover ROM 3:21-31 — only the better-ranked survives.
    assert len([i for i in ids if i.startswith("s_")]) == 1
    assert len(ids) == len(set(ids))


def test_dense_deduped_records_raw_ranks_and_dedups(index):
    sims = np.array([0.1, 0.2, 0.9, 0.8], dtype=np.float32)  # id-order rows
    deduped = index.dense_deduped(sims, [], raw_depth=4)
    ids = [(index.ids[i], raw) for i, _s, raw in deduped]
    # s_bsb1 (0.9, raw rank 1) wins; s_kjv1 (raw 2) collapses into it.
    assert ids == [("s_bsb1", 1), ("f_wsc33", 3), ("c_com1", 4)]


def test_dense_ranking_slices_by_raw_rank_exactly(index):
    deduped = [(2, 0.9, 1), (1, 0.2, 3), (0, 0.1, 4)]
    assert er.dense_ranking(deduped, raw_fetch=3, depth=10) == [2, 1]
    assert er.dense_ranking(deduped, raw_fetch=4, depth=1) == [2]


# --- RRF fusion (HybridRetriever.fuse parity) ------------------------------------------
def test_rrf_fuse_matches_hand_calculation():
    anchors = ["a0", "a1", "a2"]
    fused = er.rrf_fuse([0, 1], [1, 2], anchors, rrf_k=60.0)
    # doc1: 1/62 + 1/61 (both lists) > doc0: 1/61 > doc2: 1/62.
    assert fused == [1, 0, 2]


def test_rrf_fuse_breaks_score_ties_by_anchor_ascending():
    anchors = ["b", "a"]
    fused = er.rrf_fuse([0], [1], anchors, rrf_k=60.0)
    # Same 1/61 contribution — anchor 'a' (doc 1) must come first, like the app.
    assert fused == [1, 0]


def test_rrf_fuse_lambda_half_is_exact_app_fusion():
    anchors = [f"a{i}" for i in range(5)]
    plain = er.rrf_fuse([0, 1, 2], [3, 4], anchors, rrf_k=60.0)
    weighted = er.rrf_fuse([0, 1, 2], [3, 4], anchors, rrf_k=60.0, dense_lambda=0.5)
    assert plain == weighted


def test_rrf_fuse_keeps_full_bm25_depth_asymmetry():
    """The app fuses the FULL BM25 list but only denseDepth dense hits."""
    anchors = [f"a{i:02d}" for i in range(40)]
    bm25 = list(range(30))          # 30 BM25 entries all enter fusion
    dense = list(range(30, 35))     # 5 dense entries
    fused = er.rrf_fuse(bm25, dense, anchors, rrf_k=60.0)
    assert set(fused) == set(range(35))


# --- TSK graph boost -----------------------------------------------------------------
def test_graph_boost_injects_new_pericopes_only(index):
    """Neighbors sharing a ranked verse range or excluded for the query are
    skipped; genuinely new candidates are RRF-fused in."""
    h = er.Harness(index, dense_available=False)
    i_bsb = index.id_to_idx["s_bsb1"]
    i_kjv = index.id_to_idx["s_kjv1"]      # same verse range as s_bsb1
    i_com = index.id_to_idx["c_com1"]
    i_wsc = index.id_to_idx["f_wsc33"]
    h.expansion = {i_bsb: [i_kjv, i_wsc, i_com]}
    h.graph_available = True
    cache = er.QueryCache(
        qid="q", category="crossref", relevant=set(), hard_negative=None,
        bm25_types={}, exclude_idx=[i_wsc])

    boosted = h._graph_boost([i_bsb], cache, rrf_k=60.0, graph_lambda=0.5)
    # s_kjv1 (same range) and f_wsc33 (excluded) are skipped; c_com1 joins.
    assert set(boosted) == {i_bsb, i_com}
    # Equal-weight tie (both rank 1 in their lists) breaks by anchor asc:
    # "bsb:ROM…" < "henry:GEN…".
    assert boosted == [i_bsb, i_com]


def test_graph_boost_is_identity_without_candidates(index):
    h = er.Harness(index, dense_available=False)
    cache = er.QueryCache(
        qid="q", category="crossref", relevant=set(), hard_negative=None,
        bm25_types={})
    base = [index.id_to_idx["s_bsb1"], index.id_to_idx["c_com1"]]
    assert h._graph_boost(base, cache, rrf_k=60.0, graph_lambda=0.5) == base


# --- harness arms -------------------------------------------------------------------
def test_harness_bm25_arm_runs_without_encoder(index, tmp_path):
    gold = [{
        "id": "q1", "category": "prooftext",
        "query": "justification by faith",
        "relevant": ["s_bsb1", "s_kjv1"], "exclude": ["f_wsc33"],
    }]
    h = er.Harness(index, dense_available=False)
    h.caches.append(er.QueryCache(
        qid="q1", category="prooftext", relevant=set(gold[0]["relevant"]),
        hard_negative=None,
        bm25_types=index.bm25_type_lists(gold[0]["query"], [
            index.id_to_idx["f_wsc33"]]),
    ))
    res = h.evaluate_arm("bm25", er.APP_CONFIG)
    assert res["overall"]["count"] == 1
    assert res["overall"]["recall_at_5"] == 1.0
    # The excluded confession chunk must never appear in the ranking.
    ranked = h.ranked_ids(h.caches[0], "bm25", er.APP_CONFIG)
    assert "f_wsc33" not in ranked
