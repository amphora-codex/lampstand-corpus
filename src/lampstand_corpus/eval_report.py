"""Render ``reports/retrieval_eval_v1.md`` from measured eval results.

Deterministic: no wall-clock timestamps — the only version reference is the
corpus identity (corpus_version placeholder + pinned model revision + chunk
count) carried in the gold set. Same inputs → byte-identical report.
"""

from __future__ import annotations

REPORT_FILENAME = "retrieval_eval_v1.md"

_METRIC_COLS = ("recall_at_5", "recall_at_10", "recall_at_20", "mrr", "ndcg_at_10")
_METRIC_HEADS = ("recall@5", "recall@10", "recall@20", "MRR", "nDCG@10")

# Verdict rule (transparent, applied to measured numbers — see render):
# hybrid vs BM25-only on the overall gold set. Gains are absolute points.
_F5_JUSTIFIED_MIN = 0.02   # >= +2pt on any headline metric -> JUSTIFIED
_F5_MARGINAL_MIN = 0.005   # >= +0.5pt -> MARGINAL, else NOT JUSTIFIED (by
#                            corpus-side evidence alone; app eval still applies)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _metrics_row(name: str, m: dict) -> str:
    cells = " | ".join(_fmt(m[c]) for c in _METRIC_COLS)
    return f"| {name} | {m['count']} | {cells} |"


def _metrics_table(rows: list[tuple[str, dict]]) -> list[str]:
    head = "| set | n | " + " | ".join(_METRIC_HEADS) + " |"
    sep = "|---|---:|" + "---:|" * len(_METRIC_HEADS)
    return [head, sep] + [_metrics_row(name, m) for name, m in rows]


def _f5_verdict(results: dict) -> tuple[str, list[str]]:
    """(verdict word, supporting lines) from hybrid-vs-BM25 deltas."""
    arms = results["arms"]
    if "hybrid" not in arms or "bm25" not in arms:
        return ("UNDETERMINED", ["dense arm unavailable — hybrid vs BM25 not measured."])
    lines: list[str] = []
    best_gain = -1.0
    for metric, head in zip(_METRIC_COLS, _METRIC_HEADS, strict=True):
        d = arms["hybrid"]["overall"][metric] - arms["bm25"]["overall"][metric]
        best_gain = max(best_gain, d)
        lines.append(f"- overall {head}: hybrid − BM25 = {d:+.3f}")
    for cat in sorted(arms["hybrid"]["per_category"]):
        if cat == "hardneg":
            continue
        d20 = (arms["hybrid"]["per_category"][cat]["recall_at_20"]
               - arms["bm25"]["per_category"][cat]["recall_at_20"])
        dm = (arms["hybrid"]["per_category"][cat]["mrr"]
              - arms["bm25"]["per_category"][cat]["mrr"])
        best_gain = max(best_gain, d20, dm)
        lines.append(f"- {cat}: recall@20 {d20:+.3f}, MRR {dm:+.3f}")
    if best_gain >= _F5_JUSTIFIED_MIN:
        verdict = "JUSTIFIED"
    elif best_gain >= _F5_MARGINAL_MIN:
        verdict = "MARGINAL"
    else:
        verdict = "NOT JUSTIFIED (on corpus-side evidence alone)"
    return verdict, lines


def render_report(gold: dict, results: dict, sweep: dict | None) -> str:
    """Assemble the full markdown report body."""
    w: list[str] = []
    corpus = gold["corpus"]
    w.append("# LampStand corpus — retrieval eval v1 (F5 measurement foundation)")
    w.append("")
    w.append("**CANDIDATE MEASUREMENT — gates retrieval tuning; changes no app "
             "constant.** The sweep recommends; the architect decides.")
    w.append("")
    w.append(f"Corpus reference: `{corpus.get('corpus_version') or 'unversioned'}` · "
             f"model `{corpus.get('model_revision', '')[:12]}` · "
             f"{corpus.get('n_chunks', 0):,} chunks · gold seed {gold['seed']}.")
    w.append("")

    # -- gold set ----------------------------------------------------------------
    w.append("## 1. Gold set")
    w.append("")
    w.append("Zero-annotation labels from the corpus itself; every query records "
             "its own source chunk(s) as EXCLUDED so a verbatim self-hit cannot "
             "occupy rank 1 (hardneg queries exclude nothing — their relevant "
             "chunk contains the question).")
    w.append("")
    w.append("| category | n | query | relevant |")
    w.append("|---|---:|---|---|")
    counts = gold["counts"]
    w.append(f"| prooftext | {counts.get('prooftext', 0)} | catechism question / "
             "confession opening sentence (citations stripped) | chunks covering "
             "the section's proof-text verses |")
    thr = gold["thresholds"]["crossref_votes_min"]
    w.append(f"| crossref | {counts.get('crossref', 0)} | source verse text (BSB) "
             f"| chunks covering TSK targets with votes ≥ {thr} (data-driven "
             "threshold) |")
    w.append(f"| commentary-anchor | {counts.get('commentary-anchor', 0)} | "
             "commentary paragraph (word-boundary truncated) | Scripture chunks "
             "at its verse anchor |")
    w.append(f"| hardneg | {counts.get('hardneg', 0)} | WSC/WLC question A | A's "
             "Q&A chunk (hard negative: doctrinally-adjacent B's chunk) — **DRAFT, "
             "pending theological-advisor review** (`data/eval/hard_negatives_v1.json`) |")
    w.append("")
    w.append("Metric definitions match the app eval (RetrievalEvalTests): "
             "recall@k = share of queries with ≥1 relevant chunk in the top k; "
             "MRR capped at rank 20; nDCG@10 binary-gain (corpus-side addition). "
             "hardneg queries are scored separately (pairwise) and excluded from "
             "the overall row.")
    w.append("")

    # -- arms ---------------------------------------------------------------------
    w.append("## 2. Arms at the app's shipped constants")
    w.append("")
    w.append(f"Fusion config: `{results['app_config_label']}` "
             "(HybridRetriever.swift). BM25 arm = the app's per-type-balanced "
             "lexical ranking; dense arm = the app's deduped dense contribution "
             "(depth 20 — a deeper dense-only arm cannot change any top-20 "
             "metric, since single-list RRF is rank-preserving); hybrid = RRF "
             "fusion of both, exactly as `hybridContext` fuses.")
    w.append("")
    w.append("Labels are CONSERVATIVE (zero-annotation): only the derived "
             "chunks count as relevant. A doctrinally-correct sibling hit — "
             "e.g. WSC 77 'ninth commandment' retrieved for a Heidelberg "
             "ninth-commandment question whose labels are its proof-text "
             "VERSES — scores as a miss. Absolute numbers are therefore floors, "
             "not user-experienced quality; ARM-VS-ARM deltas on identical "
             "labels are the meaningful signal.")
    w.append("")
    if not results.get("dense_available", False):
        w.append("> **DENSE ARMS UNAVAILABLE** — the query encoder failed to "
                 "load; BM25-only numbers below. "
                 f"Reason: {results.get('dense_unavailable_reason', 'unknown')}")
        w.append("")
    for arm in results["arm_order"]:
        arm_res = results["arms"][arm]
        w.append(f"### {arm}")
        w.append("")
        rows = [("OVERALL", arm_res["overall"])] + [
            (cat, m) for cat, m in arm_res["per_category"].items() if cat != "hardneg"
        ]
        w.extend(_metrics_table(rows))
        hn = arm_res["hardneg_pairwise"]
        w.append("")
        w.append(f"hard-negative pairwise (DRAFT suite): {hn['wins']}/{hn['count']} "
                 f"wins = {_fmt(hn['win_rate'])}")
        w.append("")

    # -- F5 -------------------------------------------------------------------------
    w.append("## 3. F5 verdict — does dense retrieval justify its pack?")
    w.append("")
    verdict, lines = _f5_verdict(results)
    w.append(f"**Verdict: {verdict}.** Rule: hybrid must beat BM25-only by "
             f"≥ {_F5_JUSTIFIED_MIN:.0%} absolute on some headline metric or "
             f"category for JUSTIFIED; ≥ {_F5_MARGINAL_MIN:.1%} for MARGINAL.")
    w.append("")
    w.extend(lines)
    if "hybrid" in results["arms"]:
        hn_parts = ", ".join(
            f"{arm} {results['arms'][arm]['hardneg_pairwise']['wins']}"
            f"/{results['arms'][arm]['hardneg_pairwise']['count']}"
            for arm in results["arm_order"])
        w.append(f"- hardneg pairwise wins (DRAFT suite): {hn_parts}")
    w.append("")
    w.append("Context for the verdict: these corpus-native labels favor lexical "
             "overlap (crossref and commentary-anchor queries share verse "
             "wording with their targets), while dense retrieval's known "
             "strength — short paraphrased USER queries (the app's ~46-case "
             "eval, recall@20≈0.826) — is structurally under-represented here. "
             "This verdict gates the corpus-side evidence only; the app-side "
             "eval remains the user-experience gate, and the two should be "
             "read together before any decision about the ~1.9 GB dense pack.")
    w.append("")

    # -- sweep ------------------------------------------------------------------------
    w.append("## 4. Fusion-constant sweep")
    w.append("")
    if sweep is None:
        w.append("_Not yet run — `python -m lampstand_corpus.cli sweep-retrieval`._")
        w.append("")
    else:
        w.append("Grid: rrfK × bm25PerType × denseDepth (rawFetch = 4×depth, "
                 "floored at the app's 80), hybrid arm, overall metrics. "
                 "**Recommendation only — no app constant is changed here.**")
        w.append("")
        base = sweep["baseline"]
        w.append(f"Baseline (app): `{base['label']}`")
        w.append("")
        w.extend(_metrics_table([("app baseline", base["metrics"])]))
        w.append("")
        w.append("Best config per metric (delta vs app baseline):")
        w.append("")
        w.append("| optimizing | config | " + " | ".join(_METRIC_HEADS) + " | Δ target |")
        w.append("|---|---|" + "---:|" * (len(_METRIC_HEADS) + 1))
        for metric, head in zip(
            ("recall_at_20", "mrr", "ndcg_at_10"),
            ("recall@20", "MRR", "nDCG@10"), strict=True,
        ):
            b = sweep["best"][metric]
            cells = " | ".join(_fmt(b["metrics"][c]) for c in _METRIC_COLS)
            delta = b["delta_vs_app"][metric]
            w.append(f"| {head} | `{b['label']}` | {cells} | {delta:+.3f} |")
        w.append("")
        w.append("Weighted-RRF λ exploration (NOT an app knob today; λ=0.5 ≡ the "
                 "app's unweighted fusion), on the nDCG@10 winner's knobs:")
        w.append("")
        w.append("| λ | " + " | ".join(_METRIC_HEADS) + " |")
        w.append("|---:|" + "---:|" * len(_METRIC_HEADS))
        for row in sweep["lambda_exploration"]:
            cells = " | ".join(_fmt(row["metrics"][c]) for c in _METRIC_COLS)
            w.append(f"| {row['lambda']:g} | {cells} |")
        w.append("")
        if sweep.get("bm25_sensitivity"):
            w.append("### Honesty check — BM25-only across the per-type limit")
            w.append("")
            w.append("Any hybrid \"win\" must be read against BM25 ALONE at the "
                     "same per-type limit; a gain that survives here is fusion's, "
                     "one that doesn't came from the tighter limit:")
            w.append("")
            w.extend(_metrics_table([
                (f"bm25-only perType={row['bm25_per_type']}", row["metrics"])
                for row in sweep["bm25_sensitivity"]
            ]))
            w.append("")
            best_h = max(r["metrics"]["recall_at_20"] for r in sweep["grid"])
            best_b = max(r["metrics"]["recall_at_20"]
                         for r in sweep["bm25_sensitivity"])
            rel = "BEATS" if best_h > best_b else "does NOT beat"
            w.append(f"Best swept hybrid recall@20 = {_fmt(best_h)}; best "
                     f"BM25-only recall@20 = {_fmt(best_b)} — the swept hybrid "
                     f"{rel} BM25-only on these labels.")
            w.append("")
            w.append("Caveat on the per-type limit: relevant chunks in all three "
                     "scored categories are SCRIPTURE, so a tighter per-type "
                     "limit mechanically favors these labels by squeezing "
                     "commentary/lexicon out of the top-20 window. A user query "
                     "like \"propitiation\" WANTS commentary and lexicon rows; "
                     "do not lower bm25PerType on this evidence alone.")
            w.append("")

    # -- parity + flags -------------------------------------------------------------
    w.append("## 5. retrieve.py ↔ app parity notes")
    w.append("")
    w.append("- `src/lampstand_corpus/retrieve.py` is a dense-only SMOKE helper "
             "(global argsort over all chunks; no per-type balance, no Scripture "
             "dedup, no BM25, no RRF). It is NOT the app's ranking path. The eval "
             "runner (`eval_retrieval.py`) implements the app path faithfully; "
             "retrieve.py is left untouched for the P6 smoke report.")
    w.append("- The pipeline records BM25 k1/b/avgdl/N in `bm25_stats` but NOT "
             "the query-time IDF variant; the app's documented choice (Lucene "
             "non-negative `ln((N−df+0.5)/(df+0.5)+1)`) is mirrored here. "
             "Recommend the pipeline record `idf_form` in `bm25_stats`/meta so "
             "the contract is explicit.")
    w.append("- Fusion is ASYMMETRIC in the app: the full deduped BM25 list "
             "(up to 4×perType entries) enters RRF, while dense contributes only "
             "its top `denseDepth`. Mirrored exactly; worth an explicit comment "
             "in HybridRetriever if it is intentional.")
    w.append("- The app's dense TopK admits raw hits on `score > worst` while "
             "streaming; at an exact float score tie at the admission boundary "
             "the kept id can differ from this runner's full-sort (score desc, "
             "id asc). Practically unreachable with float dot products.")
    w.append("")
    w.append("## 6. Flags for the architect / advisor")
    w.append("")
    w.append("- **DRAFT** hard-negative suite (`data/eval/hard_negatives_v1.json`) "
             "awaits theological-advisor review; pairs were generated by answer "
             "token overlap, so some may be doctrinally trivial or too close.")
    w.append("- WLC/WSC/Belgic/Dort sections carry NO proof_texts in "
             "`confessions.sqlite` (only WCF 172 / LBCF 159 / Heidelberg 124 do); "
             "the prooftext category therefore draws from those three documents.")
    for note in gold.get("notes", []):
        if "DRAFT" in note or "only" in note:
            w.append(f"- gold-builder note: {note}")
    w.append("")
    w.append("*Generated by `python -m lampstand_corpus.cli validate-retrieval` / "
             "`sweep-retrieval`. Deterministic: fixed seed, no timestamps.*")
    w.append("")
    return "\n".join(w)
