"""Retrieval-eval runner: BM25 / dense / hybrid arms with APP-PARITY fusion.

This reimplements, in Python over ``embeddings.sqlite``, exactly the ranking
path the iOS app's ``HybridRetriever.hybridContext`` takes (lampstand-ios,
``Data/Corpus/HybridRetriever.swift`` / ``SearchStore.swift`` /
``DenseRetriever.swift`` / ``ChunkRepository.swift``):

  BM25 half   tokenize (pipeline tokenizer) → dedup terms → Okapi BM25 with the
              pack's k1/b/avgdl/N and the Lucene non-negative IDF
              ln((N−df+0.5)/(df+0.5)+1) → top ``bm25_per_type`` chunks of EACH
              resource type (score desc, id asc) → merge (score desc, id asc)
              → Scripture 4-translation dedup (keep first per verse-range key).
              The FULL deduped list enters fusion (the app never truncates it).
  dense half  BGE query vector (instruction-prefixed) → cosine (dot on unit
              vectors) over all chunks → top ``dense_raw_fetch`` raw (score
              desc, id asc) → Scripture dedup → first ``dense_depth`` distinct.
  fusion      RRF: fused[id] += w/(k + rank), rank 1-based per list; sort fused
              desc, anchor asc. App constants: k=60, per-type 20, dense depth
              20, raw fetch 80, weights 1/1 (the app has NO weighted-RRF lambda;
              ``dense_lambda`` here is an exploratory sweep axis only, with 0.5
              == exact app behavior scaled by a rank-preserving constant).

Known corpus↔app notes (kept in the report, not silently resolved):
  * ``retrieve.py`` in this repo is a dense-only smoke helper (global argsort,
    no per-type balance, no dedup, no BM25/RRF) — it is NOT the app path. This
    module is the parity implementation; retrieve.py is reused only for its
    query-instruction constant via embeddings.QUERY_INSTRUCTION.
  * The pipeline records k1/b/avgdl/N but not the query-time IDF variant; the
    Lucene non-negative form is the app's documented choice and is mirrored.
  * The app's dense TopK admits on ``score > worst`` while streaming, then
    tie-breaks (score desc, id asc); exact float score ties at the admission
    boundary could differ from this module's full-sort tie-break. Scores are
    float dot products, so exact ties are vanishingly rare.

Self-hit exclusion: a gold query's ``exclude`` chunk ids are removed from BOTH
halves' candidate pools before ranking (the app does no exclusion; the eval
needs it so a query's own verbatim source chunk cannot occupy rank 1).

Determinism: pure functions of the artifact + gold set; ties broken by id /
anchor ascending exactly as the app does.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embeddings import tokenize
from .eval_metrics import Metrics, aggregate, pairwise_wins

# --- app constants (HybridRetriever.swift) ------------------------------------
APP_RRF_K = 60.0
APP_BM25_PER_TYPE = 20
APP_DENSE_DEPTH = 20
APP_DENSE_RAW_FETCH = 80

# Cache depths: per-query rankings are cached ONCE this deep, then any sweep
# config at/below these depths is a cheap in-memory re-fusion.
CACHE_BM25_PER_TYPE = 60
CACHE_DENSE_RAW = 640

RESOURCE_TYPES = ("scripture", "confession", "commentary", "lexicon")

RESULTS_FILENAME = "eval_results_v1.json"
SWEEP_FILENAME = "eval_sweep_v1.json"


@dataclass(frozen=True)
class FusionConfig:
    """One fusion configuration (the knobs the app hard-codes)."""

    rrf_k: float = APP_RRF_K
    bm25_per_type: int = APP_BM25_PER_TYPE
    dense_depth: int = APP_DENSE_DEPTH
    dense_raw_fetch: int = APP_DENSE_RAW_FETCH
    # NOT an app knob: weighted-RRF exploration. 0.5 == the app's unweighted
    # fusion (weights 2(1-λ)=1 and 2λ=1).
    dense_lambda: float = 0.5

    def as_dict(self) -> dict:
        return {
            "rrf_k": self.rrf_k,
            "bm25_per_type": self.bm25_per_type,
            "dense_depth": self.dense_depth,
            "dense_raw_fetch": self.dense_raw_fetch,
            "dense_lambda": self.dense_lambda,
        }

    def label(self) -> str:
        base = (f"k={self.rrf_k:g} perType={self.bm25_per_type} "
                f"denseDepth={self.dense_depth} rawFetch={self.dense_raw_fetch}")
        if self.dense_lambda != 0.5:
            base += f" λ={self.dense_lambda:g}"
        return base


APP_CONFIG = FusionConfig()


# --- index -----------------------------------------------------------------------
class EvalIndex:
    """Chunk metadata + BM25 scorer over ``embeddings.sqlite``.

    All chunk arrays are aligned to one index space sorted by chunk id
    ASCENDING, so "tie-break by id asc" == "tie-break by row index asc".
    """

    def __init__(self, emb_db: Path) -> None:
        self._conn = sqlite3.connect(emb_db)
        rows = self._conn.execute(
            "SELECT c.id, c.resource_type, c.anchor, c.book, c.chapter, "
            "c.verse_start, c.verse_end, d.length "
            "FROM chunk c JOIN bm25_doc d ON d.chunk_id = c.id ORDER BY c.id"
        ).fetchall()
        n = len(rows)
        self.ids: list[str] = [r[0] for r in rows]
        self.anchors: list[str] = [r[2] for r in rows]
        self.id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(self.ids)}
        type_code = {t: i for i, t in enumerate(RESOURCE_TYPES)}
        self.rtype = np.array([type_code[r[1]] for r in rows], dtype=np.int8)
        self.doc_len = np.array([r[7] for r in rows], dtype=np.float64)
        # Scripture dedup key: translation-independent verse-range identity,
        # exactly ChunkRepository's "book|chapter|vstart|vend".
        self.dedup_key: list[str | None] = [
            f"{r[3]}|{r[4]}|{r[5]}|{r[6]}" if r[1] == "scripture" else None
            for r in rows
        ]

        stats = dict(self._conn.execute("SELECT key, value FROM bm25_stats"))
        self.k1 = float(stats["k1"])
        self.b = float(stats["b"])
        self.avgdl = float(stats["avgdl"])
        self.n_docs = float(stats["n_docs"])
        meta = dict(self._conn.execute("SELECT key, value FROM meta"))
        self.model_revision = meta.get("model_revision", "")
        self.dim = int(meta.get("embedding_dim", 384))
        self.n_chunks = n

        self._postings: dict[str, tuple[np.ndarray, np.ndarray, float] | None] = {}
        self._scores = np.zeros(n, dtype=np.float64)

    def close(self) -> None:
        self._conn.close()

    # -- dense matrix ---------------------------------------------------------
    def load_matrix(self) -> np.ndarray:
        """All vectors as an (n, dim) float32 matrix aligned to the id order."""
        mat = np.empty((self.n_chunks, self.dim), dtype=np.float32)
        i = 0
        for cid, blob in self._conn.execute(
                "SELECT chunk_id, vector FROM embedding ORDER BY chunk_id"):
            if cid != self.ids[i]:
                raise ValueError(f"embedding/chunk id misalignment at row {i}")
            mat[i] = np.frombuffer(blob, dtype="<f4")
            i += 1
        if i != self.n_chunks:
            raise ValueError(f"embedding rows {i} != chunks {self.n_chunks}")
        return mat

    # -- BM25 -------------------------------------------------------------------
    def _term_postings(self, term: str) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(chunk_idx array, tf array, idf) for one term; None if unindexed.

        Cached across queries — common terms ("the", "lord") dominate cost.
        """
        if term in self._postings:
            return self._postings[term]
        row = self._conn.execute(
            "SELECT term_id, doc_freq FROM bm25_term WHERE term=?", (term,)
        ).fetchone()
        if row is None:
            self._postings[term] = None
            return None
        term_id, df = row
        idf = float(np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0))
        posts = self._conn.execute(
            "SELECT chunk_id, term_freq FROM bm25_posting WHERE term_id=?",
            (term_id,),
        ).fetchall()
        idx = np.fromiter(
            (self.id_to_idx[cid] for cid, _tf in posts), dtype=np.int64, count=len(posts))
        tf = np.fromiter((t for _cid, t in posts), dtype=np.float64, count=len(posts))
        out = (idx, tf, idf)
        self._postings[term] = out
        return out

    def bm25_scores(self, query: str, exclude_idx: list[int]) -> np.ndarray:
        """Okapi BM25 scores over all chunks (0 = unmatched), app-identical.

        Terms are deduplicated (the app tokenizes into a Set); a term with
        idf <= 0 is skipped (unreachable with the +1 IDF form, kept for parity).
        """
        scores = self._scores
        scores[:] = 0.0
        for term in sorted(set(tokenize(query))):
            p = self._term_postings(term)
            if p is None:
                continue
            idx, tf, idf = p
            if idf <= 0:
                continue
            norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[idx] / self.avgdl)
            scores[idx] += idf * (tf * (self.k1 + 1.0)) / (tf + norm)
        if exclude_idx:
            scores[exclude_idx] = 0.0
        return scores

    def bm25_type_lists(
        self, query: str, exclude_idx: list[int], depth: int = CACHE_BM25_PER_TYPE
    ) -> dict[int, list[tuple[int, float]]]:
        """Per-resource-type top-``depth`` matched chunks (score desc, id asc)."""
        scores = self.bm25_scores(query, exclude_idx)
        matched = scores > 0
        out: dict[int, list[tuple[int, float]]] = {}
        for t in range(len(RESOURCE_TYPES)):
            cand = np.nonzero(matched & (self.rtype == t))[0]
            if cand.size == 0:
                out[t] = []
                continue
            order = np.lexsort((cand, -scores[cand]))[:depth]
            picked = cand[order]
            out[t] = [(int(i), float(scores[i])) for i in picked]
        return out

    # -- dense ------------------------------------------------------------------
    def dense_deduped(
        self, sims: np.ndarray, exclude_idx: list[int], raw_depth: int = CACHE_DENSE_RAW
    ) -> list[tuple[int, float, int]]:
        """Scripture-deduped dense ranking with raw-rank bookkeeping.

        Returns ``[(chunk_idx, score, raw_rank)]`` where ``raw_rank`` is the
        1-based position in the RAW (pre-dedup) cosine ranking — so any
        ``(dense_raw_fetch, dense_depth)`` config at/below ``raw_depth`` can be
        sliced exactly: keep items with raw_rank <= raw_fetch, take the first
        dense_depth.
        """
        s = sims.astype(np.float64, copy=True)
        if exclude_idx:
            s[exclude_idx] = -np.inf
        k = min(raw_depth, s.size)
        part = np.argpartition(-s, k - 1)[:k]
        order = np.lexsort((part, -s[part]))
        raw = part[order]
        out: list[tuple[int, float, int]] = []
        seen: set[str] = set()
        for rank, i in enumerate(raw, start=1):
            key = self.dedup_key[int(i)]
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            out.append((int(i), float(s[i]), rank))
        return out


# --- per-config ranking assembly ----------------------------------------------
def bm25_ranking(
    type_lists: dict[int, list[tuple[int, float]]],
    per_type: int,
    dedup_key: list[str | None],
) -> list[int]:
    """Merge per-type lists (score desc, id asc) and dedup Scripture.

    Mirrors SearchStore.rankedBM25 + ChunkRepository.hydrate: the merged list
    is NOT truncated — the whole deduped ranking enters fusion.
    """
    merged: list[tuple[int, float]] = []
    for t in sorted(type_lists):
        merged.extend(type_lists[t][:per_type])
    merged.sort(key=lambda e: (-e[1], e[0]))
    out: list[int] = []
    seen: set[str] = set()
    for i, _score in merged:
        key = dedup_key[i]
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        out.append(i)
    return out


def dense_ranking(
    deduped: list[tuple[int, float, int]], raw_fetch: int, depth: int
) -> list[int]:
    """Slice the cached dense ranking to one (raw_fetch, depth) config."""
    out = [i for (i, _s, raw_rank) in deduped if raw_rank <= raw_fetch]
    return out[:depth]


def rrf_fuse(
    bm25_ids: list[int],
    dense_ids: list[int],
    anchors: list[str],
    *,
    rrf_k: float = APP_RRF_K,
    dense_lambda: float = 0.5,
) -> list[int]:
    """Reciprocal Rank Fusion, app-identical at ``dense_lambda=0.5``.

    fused[id] = w_bm25/(k+rank_bm25) + w_dense/(k+rank_dense) with weights
    (2(1-λ), 2λ) — identity (1, 1) at λ=0.5. Sort: fused desc, anchor asc.
    """
    w_bm25 = 2.0 * (1.0 - dense_lambda)
    w_dense = 2.0 * dense_lambda
    fused: dict[int, float] = {}
    for rank, i in enumerate(bm25_ids, start=1):
        fused[i] = fused.get(i, 0.0) + w_bm25 / (rrf_k + rank)
    for rank, i in enumerate(dense_ids, start=1):
        fused[i] = fused.get(i, 0.0) + w_dense / (rrf_k + rank)
    return [i for i, _ in sorted(fused.items(), key=lambda kv: (-kv[1], anchors[kv[0]]))]


# --- query cache + harness ------------------------------------------------------
@dataclass
class QueryCache:
    """Deep cached rankings for one query; every sweep config slices these."""

    qid: str
    category: str
    relevant: set[str]
    hard_negative: str | None
    bm25_types: dict[int, list[tuple[int, float]]]
    dense: list[tuple[int, float, int]] = field(default_factory=list)


class Harness:
    """Builds per-query caches once, then evaluates any FusionConfig cheaply."""

    def __init__(self, index: EvalIndex, dense_available: bool) -> None:
        self.index = index
        self.dense_available = dense_available
        self.caches: list[QueryCache] = []

    @classmethod
    def build(
        cls, emb_db: Path, gold_queries: list[dict], *, encode_queries=None
    ) -> Harness:
        """Cache BM25 + dense rankings for every gold query.

        ``encode_queries`` is a callable ``list[str] -> np.ndarray`` producing
        instruction-prefixed unit query vectors; None (or a load failure
        upstream) degrades to BM25-only arms.
        """
        index = EvalIndex(emb_db)
        h = cls(index, dense_available=encode_queries is not None)

        sims_by_row: np.ndarray | None = None
        if encode_queries is not None:
            matrix = index.load_matrix()
            qvecs = encode_queries([q["query"] for q in gold_queries])
            sims_by_row = qvecs.astype(np.float32) @ matrix.T
            del matrix

        for row, q in enumerate(gold_queries):
            exclude_idx = [
                index.id_to_idx[cid] for cid in q.get("exclude", [])
                if cid in index.id_to_idx
            ]
            cache = QueryCache(
                qid=q["id"],
                category=q["category"],
                relevant=set(q["relevant"]),
                hard_negative=q.get("hard_negative"),
                bm25_types=index.bm25_type_lists(q["query"], exclude_idx),
            )
            if sims_by_row is not None:
                cache.dense = index.dense_deduped(sims_by_row[row], exclude_idx)
            h.caches.append(cache)
        return h

    # -- arms -------------------------------------------------------------------
    def ranked_ids(
        self, cache: QueryCache, arm: str, cfg: FusionConfig
    ) -> list[str]:
        idx = self.index
        bm25_ids: list[int] = []
        dense_ids: list[int] = []
        if arm in ("bm25", "hybrid"):
            bm25_ids = bm25_ranking(cache.bm25_types, cfg.bm25_per_type, idx.dedup_key)
        if arm in ("dense", "hybrid"):
            dense_ids = dense_ranking(cache.dense, cfg.dense_raw_fetch, cfg.dense_depth)
        fused = rrf_fuse(
            bm25_ids, dense_ids, idx.anchors,
            rrf_k=cfg.rrf_k, dense_lambda=cfg.dense_lambda)
        return [idx.ids[i] for i in fused]

    def evaluate_arm(self, arm: str, cfg: FusionConfig) -> dict:
        """Metrics for one arm/config: overall, per category, hardneg pairwise."""
        per_cat: dict[str, list[tuple[list[str], set[str]]]] = {}
        pairwise: list[tuple[int | None, int | None]] = []
        for cache in self.caches:
            ranked = self.ranked_ids(cache, arm, cfg)
            per_cat.setdefault(cache.category, []).append((ranked, cache.relevant))
            if cache.hard_negative is not None:
                rel_rank = _rank_of(ranked, cache.relevant)
                neg_rank = _rank_of(ranked, {cache.hard_negative})
                pairwise.append((rel_rank, neg_rank))
        scored = [pairs for cat, pairs in per_cat.items() if cat != "hardneg"]
        overall = aggregate([p for pairs in scored for p in pairs])
        return {
            "overall": overall.as_dict(),
            "per_category": {
                cat: aggregate(pairs).as_dict() for cat, pairs in sorted(per_cat.items())
            },
            "hardneg_pairwise": pairwise_wins(pairwise),
        }

    def overall_metrics(self, arm: str, cfg: FusionConfig) -> Metrics:
        pairs = [
            (self.ranked_ids(c, arm, cfg), c.relevant)
            for c in self.caches if c.category != "hardneg"
        ]
        return aggregate(pairs)


def _rank_of(ranked: list[str], targets: set[str]) -> int | None:
    for i, cid in enumerate(ranked, start=1):
        if cid in targets:
            return i
    return None


# --- sweep -----------------------------------------------------------------------
SWEEP_RRF_K = (20.0, 40.0, 60.0, 100.0, 160.0)
SWEEP_PER_TYPE = (10, 20, 40, 60)
SWEEP_DENSE_DEPTH = (10, 20, 40, 80, 160)
SWEEP_LAMBDA = (0.3, 0.4, 0.5, 0.6, 0.7)
SWEEP_METRICS = ("recall_at_20", "mrr", "ndcg_at_10")


def sweep_grid() -> list[FusionConfig]:
    """The (rrf_k × per_type × dense_depth) grid; rawFetch tracks 4×depth
    (the app's own 80 = 4×20 over-fetch rationale)."""
    grid = []
    for k in SWEEP_RRF_K:
        for pt in SWEEP_PER_TYPE:
            for dd in SWEEP_DENSE_DEPTH:
                grid.append(FusionConfig(
                    rrf_k=k, bm25_per_type=pt, dense_depth=dd,
                    dense_raw_fetch=max(4 * dd, APP_DENSE_RAW_FETCH)))
    return grid


def run_sweep(harness: Harness) -> dict:
    """Grid sweep over the hybrid arm + a lambda pass on the grid winner.

    Returns baseline (app config), the full grid, the best config per metric
    with deltas vs baseline, and the lambda exploration. Deterministic: grid
    order fixed; ties on a metric keep the earliest (smallest-knob) config.
    """
    baseline = harness.overall_metrics("hybrid", APP_CONFIG).as_dict()
    rows: list[dict] = []
    for cfg in sweep_grid():
        m = harness.overall_metrics("hybrid", cfg).as_dict()
        rows.append({"config": cfg.as_dict(), "label": cfg.label(), "metrics": m})

    best: dict[str, dict] = {}
    for metric in SWEEP_METRICS:
        top = max(rows, key=lambda r: r["metrics"][metric])
        best[metric] = {
            **top,
            "delta_vs_app": {
                k: top["metrics"][k] - baseline[k]
                for k in ("recall_at_5", "recall_at_10", "recall_at_20",
                          "mrr", "ndcg_at_10")
            },
        }

    # Lambda pass on the nDCG@10 winner's knobs (exploratory only — the app has
    # no lambda; 0.5 is the app's unweighted fusion).
    win = best["ndcg_at_10"]["config"]
    lam_rows = []
    for lam in SWEEP_LAMBDA:
        cfg = FusionConfig(
            rrf_k=win["rrf_k"], bm25_per_type=win["bm25_per_type"],
            dense_depth=win["dense_depth"], dense_raw_fetch=win["dense_raw_fetch"],
            dense_lambda=lam)
        lam_rows.append({
            "lambda": lam,
            "metrics": harness.overall_metrics("hybrid", cfg).as_dict(),
        })

    # BM25-ONLY sensitivity to the per-type limit — the honesty check for any
    # hybrid "win": if BM25 alone at the same per-type limit beats the swept
    # hybrid, the gain came from the tighter limit, not from dense fusion.
    bm25_rows = [
        {
            "bm25_per_type": pt,
            "metrics": harness.overall_metrics(
                "bm25", FusionConfig(bm25_per_type=pt)).as_dict(),
        }
        for pt in SWEEP_PER_TYPE
    ]

    return {
        "baseline": {"config": APP_CONFIG.as_dict(), "label": APP_CONFIG.label(),
                     "metrics": baseline},
        "grid": rows,
        "best": best,
        "lambda_exploration": lam_rows,
        "bm25_sensitivity": bm25_rows,
    }
