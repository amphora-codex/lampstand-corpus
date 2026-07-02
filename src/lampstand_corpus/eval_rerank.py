"""Cross-encoder rerank measurement (Rank 4 — on-device reranker, EVAL-GATED).

This module measures whether a cross-encoder reranker EARNS its place on top of
the v2 hybrid retriever. The audit called a reranker the single biggest ranking
lever, but that was measured against the OLD (pre-re-chunk) retrieval. Since the
corpus re-chunk roughly doubled ranking quality, the reranker must beat a much
stronger baseline. **We measure first; the CoreML export happens only if the
quality gate passes** (see ``reports/reranker_eval_v1.md`` for the verdict).

What it does
------------
For each gold query, take the un-reranked v2 HYBRID top-``RERANK_K`` candidates
the :class:`~lampstand_corpus.eval_retrieval.Harness` already produces (the exact
app fusion path), fetch each candidate chunk's text, score the (query, chunk_text)
pairs with a PyTorch/sentence-transformers cross-encoder, re-sort the window by
descending cross-encoder score (stable, id-asc tie-break to match the app's
determinism), and recompute recall@5/10/20, MRR, nDCG@10 PER CATEGORY vs the
un-reranked hybrid baseline. Candidates below the reranked window keep their
fused order and stay after it (so recall@20 can only be affected by reordering
WITHIN the window, never by dropping a tail hit — reranking a top-30 window can
lift a rank-21..30 relevant chunk into the top-20).

Two text variants are measured when cheap:
  * ``header``  — the baked structural header (``"Psalms 23:1 — "``) + text, i.e.
    exactly the string that was BM25-indexed / embedded.
  * ``raw``     — the display ``text`` alone (no header). The header may help the
    cross-encoder disambiguate a bare verse, or hurt it by injecting a reference
    label that the query never contains; we let the numbers decide.

The cross-encoder (sentence-transformers / torch) is a DEV/EVAL-ONLY dependency
(the ``[rerank]`` extra) — it is NEVER shipped in the core package, and CoreML
is not needed to MEASURE. If no model can be loaded (offline / not cached), the
harness reports that the gate could not run and stops (the CLI degrades loudly).

Honesty caveat (stated in the report, load-bearing here): the corpus gold labels
favor LEXICAL overlap (crossref/commentary-anchor queries share verse wording
with their targets; documented in ``reports/retrieval_eval_v1.md`` §2-3). A
reranker that improves SEMANTIC matching will therefore show MUTED gains on this
label set. Weight the per-category exegetical-style lift (commentary-anchor,
crossref) and the hard-negative pairwise result heavily; the paraphrased-user-
query strength lives in the app's own 46-case eval, not here.

Determinism: single-thread CPU, fixed seeds (mirrors ``encode._set_deterministic_
cpu``); the model is loaded from a pinned revision; no timestamps in any artifact.
Cross-encoder logits are deterministic on CPU, so the reranked order — and thus
every metric — is reproducible run-to-run for a given model revision.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .eval_metrics import aggregate, pairwise_wins
from .eval_retrieval import APP_CONFIG, Harness, _rank_of

# Candidate window the reranker re-sorts. The app plan slots the reranker over
# the FUSED top-30 (between RRF fusion and the tradition multiplier); ~30 pairs/
# query is the on-device affordability target, so we measure exactly that depth.
RERANK_K = 30

# Permissively-licensed cross-encoders (license VERIFIED via the HF hub, recorded
# in the report). Primary is the small Apache-2.0 MiniLM (BERT-family, the most
# CoreML-export-friendly if the gate passes); the MIT bge-reranker-base is a
# stronger second data point. All are DEV/EVAL-only — never shipped in-package.
RERANK_MODELS: dict[str, dict[str, str]] = {
    "ms-marco-MiniLM-L-6-v2": {
        "hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "license": "apache-2.0",
        "arch": "BERT (MiniLM-L6)",
    },
    "bge-reranker-base": {
        "hf_id": "BAAI/bge-reranker-base",
        "license": "mit",
        "arch": "XLM-RoBERTa base",
    },
}
DEFAULT_RERANK_MODEL = "ms-marco-MiniLM-L-6-v2"

# Cross-encoder pair truncation. ~192 tokens comfortably covers a short query +
# a single-verse / short-paragraph candidate; it is also the on-device pair
# budget documented in the app-integration contract (docs/reranker-pack.md if
# the gate ships). Longer commentary candidates are truncated by the tokenizer.
MAX_PAIR_TOKENS = 192

RESULTS_FILENAME = "eval_rerank_v1.json"

# Metric keys carried through the deltas.
_METRIC_KEYS = ("recall_at_5", "recall_at_10", "recall_at_20", "mrr", "ndcg_at_10")


@dataclass
class RerankModel:
    """A loaded cross-encoder + its verified provenance."""

    key: str
    hf_id: str
    license: str
    arch: str
    revision: str
    _encoder: object  # sentence_transformers.CrossEncoder

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Cross-encoder relevance logits for (query, passage) pairs.

        Higher = more relevant. Deterministic on CPU (single thread, eval mode).
        """
        import numpy as np

        if not pairs:
            return []
        scores = self._encoder.predict(
            pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return [float(x) for x in np.asarray(scores, dtype=np.float64).reshape(-1)]


def load_cross_encoder(model_key: str, cache_dir: Path) -> RerankModel:
    """Load a pinned cross-encoder from the model cache (deterministic CPU).

    Downloads to ``cache_dir`` on first use; subsequent loads are offline. Raises
    on any failure so the CLI can degrade loudly (report: gate could not run).
    """
    import os

    from sentence_transformers import CrossEncoder

    from .encode import _set_deterministic_cpu

    spec = RERANK_MODELS[model_key]
    _set_deterministic_cpu()
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    encoder = CrossEncoder(spec["hf_id"], max_length=MAX_PAIR_TOKENS, device="cpu")
    # sentence-transformers CrossEncoder wraps an HF model; pin eval mode.
    try:
        encoder.model.eval()
    except AttributeError:  # pragma: no cover - version-dependent attr
        pass
    revision = _resolve_revision(spec["hf_id"], cache_dir)
    return RerankModel(
        key=model_key, hf_id=spec["hf_id"], license=spec["license"],
        arch=spec["arch"], revision=revision, _encoder=encoder)


def _resolve_revision(hf_id: str, cache_dir: Path) -> str:
    """Best-effort resolved commit hash of the cached snapshot (provenance)."""
    repo = "models--" + hf_id.replace("/", "--")
    snap_dir = cache_dir / repo / "snapshots"
    if snap_dir.exists():
        snaps = sorted(p.name for p in snap_dir.iterdir() if p.is_dir())
        if snaps:
            return snaps[-1]
    return "unknown"


# --- candidate text --------------------------------------------------------------
class ChunkTextIndex:
    """Fetch a chunk's display text (and structural header) by chunk id.

    Reads the SAME ``chunk`` rows the embeddings index was built from, so a
    candidate's rerank text is exactly the retrievable unit's content. Parent
    (``indexed=0``) rows carry NULL text, but the retriever only ever surfaces
    ``indexed=1`` children as candidates, so a NULL is treated as empty and
    flagged rather than reconstructed.
    """

    def __init__(self, emb_db: Path) -> None:
        conn = sqlite3.connect(emb_db)
        try:
            rows = conn.execute(
                "SELECT id, header, text FROM chunk").fetchall()
        finally:
            conn.close()
        self._header: dict[str, str] = {r[0]: (r[1] or "") for r in rows}
        self._text: dict[str, str] = {r[0]: (r[2] or "") for r in rows}

    def pair_text(self, chunk_id: str, *, with_header: bool) -> str:
        """The candidate text for a (query, passage) pair.

        ``with_header`` prepends the baked structural header exactly as it was
        indexed/embedded; otherwise returns the raw display text.
        """
        text = self._text.get(chunk_id, "")
        if with_header:
            header = self._header.get(chunk_id, "")
            return (header + text) if header else text
        return text


# --- rerank the fused window ------------------------------------------------------
def rerank_ids(
    fused: list[str],
    query: str,
    model: RerankModel,
    texts: ChunkTextIndex,
    *,
    with_header: bool,
    k: int = RERANK_K,
) -> list[str]:
    """Re-sort the top-``k`` fused candidates by cross-encoder score.

    The window ``fused[:k]`` is re-ordered by descending cross-encoder logit;
    ties (and any candidate with empty text) fall back to the ORIGINAL fused
    order (a stable sort keyed on the fused position), which is exactly the app's
    id/anchor-ascending determinism carried through fusion. The tail
    ``fused[k:]`` keeps its fused order and stays after the reranked window, so
    reranking can only reorder WITHIN the window (it never drops a candidate).
    """
    window = fused[:k]
    if not window:
        return list(fused)
    pairs = [(query, texts.pair_text(cid, with_header=with_header)) for cid in window]
    scores = model.score_pairs(pairs)
    # Stable sort: primary = cross-encoder score desc, secondary = original fused
    # rank asc (Python's sort is stable, so equal scores keep window order).
    order = sorted(range(len(window)), key=lambda i: -scores[i])
    reranked = [window[i] for i in order]
    return reranked + fused[k:]


# --- measurement ------------------------------------------------------------------
@dataclass
class RerankArm:
    """One (model, header-variant) reranked arm's measured metrics."""

    model_key: str
    hf_id: str
    license: str
    arch: str
    revision: str
    with_header: bool
    overall: dict = field(default_factory=dict)
    per_category: dict = field(default_factory=dict)
    hardneg_pairwise: dict = field(default_factory=dict)

    @property
    def variant(self) -> str:
        return "header" if self.with_header else "raw"

    def as_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "hf_id": self.hf_id,
            "license": self.license,
            "arch": self.arch,
            "revision": self.revision,
            "variant": self.variant,
            "with_header": self.with_header,
            "overall": self.overall,
            "per_category": self.per_category,
            "hardneg_pairwise": self.hardneg_pairwise,
        }


def _baseline_hybrid_rankings(harness: Harness) -> dict[str, list[str]]:
    """The un-reranked hybrid ranking (app constants) per gold query id."""
    return {
        c.qid: harness.ranked_ids(c, "hybrid", APP_CONFIG)
        for c in harness.caches
    }


def evaluate_baseline(harness: Harness) -> dict:
    """Un-reranked hybrid baseline metrics (the comparison point).

    Identical to the hybrid arm in ``eval_retrieval`` — recomputed here so the
    rerank report is self-contained and reads from one code path.
    """
    return harness.evaluate_arm("hybrid", APP_CONFIG)


def evaluate_rerank_arm(
    harness: Harness,
    baseline_rankings: dict[str, list[str]],
    model: RerankModel,
    texts: ChunkTextIndex,
    *,
    with_header: bool,
) -> RerankArm:
    """Rerank each query's hybrid top-K and recompute metrics per category."""
    per_cat: dict[str, list[tuple[list[str], set[str]]]] = {}
    pairwise: list[tuple[int | None, int | None]] = []
    for cache in harness.caches:
        fused = baseline_rankings[cache.qid]
        ranked = rerank_ids(
            fused, getattr(cache, "query", ""),
            model, texts, with_header=with_header)
        per_cat.setdefault(cache.category, []).append((ranked, cache.relevant))
        if cache.hard_negative is not None:
            rel_rank = _rank_of(ranked, cache.relevant)
            neg_rank = _rank_of(ranked, {cache.hard_negative})
            pairwise.append((rel_rank, neg_rank))
    scored = [pairs for cat, pairs in per_cat.items() if cat != "hardneg"]
    overall = aggregate([p for pairs in scored for p in pairs])
    return RerankArm(
        model_key=model.key, hf_id=model.hf_id, license=model.license,
        arch=model.arch, revision=model.revision, with_header=with_header,
        overall=overall.as_dict(),
        per_category={
            cat: aggregate(pairs).as_dict()
            for cat, pairs in sorted(per_cat.items())
        },
        hardneg_pairwise=pairwise_wins(pairwise),
    )


def run_rerank_eval(
    harness: Harness,
    gold_queries: list[dict],
    emb_db: Path,
    model_cache: Path,
    *,
    model_keys: list[str] | None = None,
    variants: tuple[bool, ...] = (True, False),
) -> dict:
    """Full rerank measurement: baseline + each (model, variant) arm + deltas.

    Returns a JSON-ready payload. Attaches ``query`` onto each harness cache so
    the reranker can build (query, passage) pairs from the same objects the
    retrieval metrics were computed on. Deterministic; no timestamps.
    """
    # Thread the raw query text onto each cache (QueryCache doesn't retain it).
    query_by_id = {q["id"]: q["query"] for q in gold_queries}
    for c in harness.caches:
        c.query = query_by_id.get(c.qid, "")  # type: ignore[attr-defined]

    texts = ChunkTextIndex(emb_db)
    baseline_rankings = _baseline_hybrid_rankings(harness)
    baseline = evaluate_baseline(harness)

    keys = model_keys or [DEFAULT_RERANK_MODEL]
    arms: list[dict] = []
    for key in keys:
        model = load_cross_encoder(key, model_cache)
        for with_header in variants:
            arm = evaluate_rerank_arm(
                harness, baseline_rankings, model, texts, with_header=with_header)
            arms.append(arm.as_dict())

    return {
        "rerank_k": RERANK_K,
        "max_pair_tokens": MAX_PAIR_TOKENS,
        "app_config_label": APP_CONFIG.label(),
        "baseline": baseline,
        "arms": arms,
        "deltas": _compute_deltas(baseline, arms),
    }


def _compute_deltas(baseline: dict, arms: list[dict]) -> list[dict]:
    """Per-arm overall + per-category metric deltas vs the un-reranked baseline."""
    out: list[dict] = []
    for arm in arms:
        overall_delta = {
            m: arm["overall"][m] - baseline["overall"][m] for m in _METRIC_KEYS
        }
        cat_delta: dict[str, dict] = {}
        for cat, m in arm["per_category"].items():
            if cat == "hardneg":
                continue
            base_cat = baseline["per_category"].get(cat, {})
            if not base_cat:
                continue
            cat_delta[cat] = {
                k: m[k] - base_cat[k] for k in _METRIC_KEYS
            }
        out.append({
            "model_key": arm["model_key"],
            "variant": arm["variant"],
            "overall": overall_delta,
            "per_category": cat_delta,
            "hardneg_win_rate": arm["hardneg_pairwise"].get("win_rate", 0.0),
            "hardneg_win_rate_delta": (
                arm["hardneg_pairwise"].get("win_rate", 0.0)
                - baseline["hardneg_pairwise"].get("win_rate", 0.0)),
        })
    return out
