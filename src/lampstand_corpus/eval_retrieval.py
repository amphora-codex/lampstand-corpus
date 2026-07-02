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
        # Matrix-row -> global chunk idx map; identity until load_matrix()
        # replaces it with the real embedding-subset mapping.
        self.emb_idx = np.arange(n, dtype=np.int64)

    def close(self) -> None:
        self._conn.close()

    # -- dense matrix ---------------------------------------------------------
    def load_matrix(self) -> np.ndarray:
        """Vectors for the EMBEDDED subset as an (n_emb, dim) float32 matrix.

        Under the Rank-8 re-chunk only the BSB Scripture children + the
        non-scripture retrieval units carry vectors, so the dense matrix is a
        SUBSET of the BM25 index. Rows are ordered by chunk id ascending;
        ``self.emb_idx[row]`` maps a matrix row to its global chunk index.
        """
        rows = self._conn.execute(
            "SELECT chunk_id, vector FROM embedding ORDER BY chunk_id"
        ).fetchall()
        mat = np.empty((len(rows), self.dim), dtype=np.float32)
        emb_idx = np.empty(len(rows), dtype=np.int64)
        for i, (cid, blob) in enumerate(rows):
            if cid not in self.id_to_idx:
                raise ValueError(
                    f"embedded chunk {cid} is not an indexed retrieval unit")
            emb_idx[i] = self.id_to_idx[cid]
            mat[i] = np.frombuffer(blob, dtype="<f4")
        self.emb_idx = emb_idx
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

        ``sims`` is aligned to the EMBEDDING-subset rows (see load_matrix);
        outputs carry GLOBAL chunk indexes. ``exclude_idx`` is global.

        Returns ``[(chunk_idx, score, raw_rank)]`` where ``raw_rank`` is the
        1-based position in the RAW (pre-dedup) cosine ranking — so any
        ``(dense_raw_fetch, dense_depth)`` config at/below ``raw_depth`` can be
        sliced exactly: keep items with raw_rank <= raw_fetch, take the first
        dense_depth.
        """
        s = sims.astype(np.float64, copy=True)
        if exclude_idx:
            excl = set(exclude_idx)
            mask = [r for r, g in enumerate(self.emb_idx) if int(g) in excl]
            if mask:
                s[mask] = -np.inf
        k = min(raw_depth, s.size)
        part = np.argpartition(-s, k - 1)[:k]
        # Tie-break by matrix row ascending == chunk id ascending (rows are
        # loaded ORDER BY chunk_id), matching the app's id-asc tie-break.
        order = np.lexsort((part, -s[part]))
        raw = part[order]
        out: list[tuple[int, float, int]] = []
        seen: set[str] = set()
        for rank, row in enumerate(raw, start=1):
            i = int(self.emb_idx[row])
            key = self.dedup_key[i]
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            out.append((i, float(s[row]), rank))
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
    # Dense ranking against the int8-quantized vectors (pack-diet quality
    # probe); empty unless the harness was built with int8_variant=True.
    dense_q: list[tuple[int, float, int]] = field(default_factory=list)
    # Self-hit exclusions (chunk idx) — kept so post-fusion steps (the TSK
    # graph boost) can never reintroduce an excluded chunk.
    exclude_idx: list[int] = field(default_factory=list)


# TSK graph boost (Rank 13 measurement): how many top-ranked Scripture hits
# contribute their cross-referenced pericopes as extra candidates. 5 sources ×
# ≤8 pack neighbors keeps the injection within one retriever's fusion budget.
GRAPH_SOURCE_TOP = 5


class Harness:
    """Builds per-query caches once, then evaluates any FusionConfig cheaply."""

    def __init__(self, index: EvalIndex, dense_available: bool,
                 int8_available: bool = False) -> None:
        self.index = index
        self.dense_available = dense_available
        self.int8_available = int8_available
        self.graph_available = False
        # chunk idx -> TSK-adjacent chunk idx list (weight order), mirroring
        # the bundled chunk_crossref table exactly (same builder).
        self.expansion: dict[int, list[int]] = {}
        self.caches: list[QueryCache] = []

    def load_expansion(self, crossrefs_db: Path, emb_db: Path) -> None:
        """Build the per-pericope TSK expansion (the pack's own code path)."""
        from .crossref_pack import build_expansion, load_scripture_chunk_refs

        chunks = load_scripture_chunk_refs(emb_db)
        expansion = build_expansion(chunks, crossrefs_db)
        idx_of = self.index.id_to_idx
        self.expansion = {
            idx_of[sid]: [idx_of[n] for n, _w in nbrs if n in idx_of]
            for sid, nbrs in expansion.items() if sid in idx_of
        }
        self.graph_available = True

    @classmethod
    def build(
        cls, emb_db: Path, gold_queries: list[dict], *, encode_queries=None,
        int8_variant: bool = False, crossrefs_db: Path | None = None,
    ) -> Harness:
        """Cache BM25 + dense rankings for every gold query.

        ``encode_queries`` is a callable ``list[str] -> np.ndarray`` producing
        instruction-prefixed unit query vectors; None (or a load failure
        upstream) degrades to BM25-only arms. ``int8_variant`` additionally
        scores against the int8 quantize→dequantize round-trip of the corpus
        matrix — mathematically identical to scoring the int8 vectors pack —
        so the pack diet's quality delta is measured by this same harness.
        ``crossrefs_db`` (the built crossrefs.sqlite) enables the experimental
        TSK graph-boost arms.
        """
        index = EvalIndex(emb_db)
        h = cls(index, dense_available=encode_queries is not None,
                int8_available=encode_queries is not None and int8_variant)
        if crossrefs_db is not None and crossrefs_db.exists():
            h.load_expansion(crossrefs_db, emb_db)

        sims_by_row: np.ndarray | None = None
        sims_q_by_row: np.ndarray | None = None
        if encode_queries is not None:
            matrix = index.load_matrix()
            qvecs = encode_queries([q["query"] for q in gold_queries])
            qvecs = qvecs.astype(np.float32)
            sims_by_row = qvecs @ matrix.T
            if int8_variant:
                from .pack_codec import quantize_roundtrip_matrix
                sims_q_by_row = qvecs @ quantize_roundtrip_matrix(matrix).T
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
                exclude_idx=exclude_idx,
            )
            if sims_by_row is not None:
                cache.dense = index.dense_deduped(sims_by_row[row], exclude_idx)
            if sims_q_by_row is not None:
                cache.dense_q = index.dense_deduped(sims_q_by_row[row], exclude_idx)
            h.caches.append(cache)
        return h

    # -- arms -------------------------------------------------------------------
    def ranked_ids(
        self, cache: QueryCache, arm: str, cfg: FusionConfig
    ) -> list[str]:
        """Ranked chunk ids for one arm.

        Arms: ``bm25`` / ``dense`` / ``hybrid``; the ``-int8`` variants of the
        dense-bearing arms (rank against the quantized vectors, identical
        fusion path); and the experimental TSK arms ``hybrid-graph`` (graph
        candidates fused as an equal third retriever — the same weight the app
        gives dense) / ``hybrid-graph-weak`` (graph list down-weighted to ⅓ of
        a retriever slot via λ=0.25).
        """
        idx = self.index
        graph = arm in ("hybrid-graph", "hybrid-graph-weak")
        base = "hybrid" if graph else arm.removesuffix("-int8")
        dense_cache = cache.dense_q if arm.endswith("-int8") else cache.dense
        bm25_ids: list[int] = []
        dense_ids: list[int] = []
        if base in ("bm25", "hybrid"):
            bm25_ids = bm25_ranking(cache.bm25_types, cfg.bm25_per_type, idx.dedup_key)
        if base in ("dense", "hybrid"):
            dense_ids = dense_ranking(dense_cache, cfg.dense_raw_fetch, cfg.dense_depth)
        fused = rrf_fuse(
            bm25_ids, dense_ids, idx.anchors,
            rrf_k=cfg.rrf_k, dense_lambda=cfg.dense_lambda)
        if graph:
            fused = self._graph_boost(
                fused, cache,
                rrf_k=cfg.rrf_k,
                graph_lambda=0.25 if arm == "hybrid-graph-weak" else 0.5)
        return [idx.ids[i] for i in fused]

    def _graph_boost(
        self, base: list[int], cache: QueryCache, *, rrf_k: float,
        graph_lambda: float,
    ) -> list[int]:
        """Post-fusion TSK boost: RRF-fuse the base ranking with the top
        Scripture hits' cross-referenced pericopes.

        Graph candidates come from the first :data:`GRAPH_SOURCE_TOP` Scripture
        hits within the top 20, in (source rank, pack weight) order; candidates
        already ranked (by verse-range identity), excluded for this query, or
        duplicated are skipped, so the graph list only introduces NEW pericopes.
        """
        dedup_key = self.index.dedup_key
        sources = [i for i in base[:20] if dedup_key[i] is not None]
        sources = sources[:GRAPH_SOURCE_TOP]
        seen_ranges = {dedup_key[i] for i in base if dedup_key[i] is not None}
        blocked = set(base) | set(cache.exclude_idx)
        graph_ids: list[int] = []
        for s in sources:
            for nb in self.expansion.get(s, ()):
                key = dedup_key[nb]
                if nb in blocked or key in seen_ranges:
                    continue
                blocked.add(nb)
                seen_ranges.add(key)
                graph_ids.append(nb)
        if not graph_ids:
            return base
        return rrf_fuse(base, graph_ids, self.index.anchors,
                        rrf_k=rrf_k, dense_lambda=graph_lambda)

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
