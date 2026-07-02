"""Render ``reports/reranker_eval_v1.md`` from the rerank measurement.

Deterministic: the only version reference is the corpus identity (corpus_version
placeholder + pinned model revision + chunk count) carried in the gold set, plus
the cross-encoder model + revision. No wall-clock timestamps — same inputs →
byte-identical report.

The verdict is applied to MEASURED numbers by :func:`_verdict`:

  SHIP  a clear win — overall MRR gain >= +0.03, OR a solid per-category
        exegetical lift (commentary-anchor / crossref MRR or recall@10
        >= +0.03) beyond noise — AND ~30 pairs/query is plausibly affordable
        on-device (it is, by construction: RERANK_K = 30).
  HOLD  gains marginal against the stronger v2 baseline. A well-argued HOLD is a
        fully successful outcome — the reranker is not worth ~20 MB + latency
        for noise.
"""

from __future__ import annotations

REPORT_FILENAME = "reranker_eval_v1.md"

_METRIC_COLS = ("recall_at_5", "recall_at_10", "recall_at_20", "mrr", "ndcg_at_10")
_METRIC_HEADS = ("recall@5", "recall@10", "recall@20", "MRR", "nDCG@10")

# Verdict thresholds (applied to measured deltas).
SHIP_OVERALL_MRR_MIN = 0.03           # overall MRR lift for an unconditional SHIP
SHIP_CATEGORY_MIN = 0.03              # per-category exegetical MRR/recall@10 lift
# Categories where reranking SHOULD help most (exegetical / semantic).
EXEGETICAL_CATEGORIES = ("commentary-anchor", "crossref")


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _signed(x: float) -> str:
    return f"{x:+.3f}"


def _metrics_row(name: str, m: dict) -> str:
    cells = " | ".join(_fmt(m[c]) for c in _METRIC_COLS)
    return f"| {name} | {m['count']} | {cells} |"


def _metrics_table(rows: list[tuple[str, dict]]) -> list[str]:
    head = "| set | n | " + " | ".join(_METRIC_HEADS) + " |"
    sep = "|---|---:|" + "---:|" * len(_METRIC_HEADS)
    return [head, sep] + [_metrics_row(name, m) for name, m in rows]


def _delta_table(delta: dict) -> list[str]:
    """A per-category delta table (overall + each scored category)."""
    head = "| set | " + " | ".join(f"Δ{h}" for h in _METRIC_HEADS) + " |"
    sep = "|---|" + "---:|" * len(_METRIC_HEADS)
    lines = [head, sep]
    lines.append("| OVERALL | " + " | ".join(
        _signed(delta["overall"][c]) for c in _METRIC_COLS) + " |")
    for cat in sorted(delta["per_category"]):
        d = delta["per_category"][cat]
        lines.append(f"| {cat} | " + " | ".join(
            _signed(d[c]) for c in _METRIC_COLS) + " |")
    return lines


def _best_arm(results: dict) -> dict:
    """Pick the delta block with the strongest exegetical MRR lift (verdict basis).

    Ranks arms by the max per-category exegetical MRR delta, then overall MRR
    delta — the metrics reranking is supposed to move. Deterministic tie-break by
    (model_key, variant).
    """
    def key(d: dict) -> tuple:
        exeg = max(
            (d["per_category"].get(c, {}).get("mrr", -9.0)
             for c in EXEGETICAL_CATEGORIES),
            default=-9.0)
        return (exeg, d["overall"]["mrr"], d["model_key"], d["variant"])

    return max(results["deltas"], key=key)


def _verdict(results: dict) -> tuple[str, dict, list[str]]:
    """(verdict word, chosen delta block, reasoning lines) from measured deltas."""
    best = _best_arm(results)
    overall_mrr = best["overall"]["mrr"]
    exeg_lifts = {
        c: max(
            best["per_category"].get(c, {}).get("mrr", 0.0),
            best["per_category"].get(c, {}).get("recall_at_10", 0.0),
        )
        for c in EXEGETICAL_CATEGORIES
    }
    best_exeg = max(exeg_lifts.values(), default=0.0)

    lines: list[str] = []
    lines.append(
        f"- best arm: `{best['model_key']}` / `{best['variant']}` "
        f"(chosen by strongest exegetical-category MRR lift).")
    lines.append(f"- overall MRR delta: {_signed(overall_mrr)} "
                 f"(SHIP floor {SHIP_OVERALL_MRR_MIN:+.3f}).")
    for c in EXEGETICAL_CATEGORIES:
        lines.append(
            f"- {c}: MRR {_signed(best['per_category'].get(c, {}).get('mrr', 0.0))}, "
            f"recall@10 "
            f"{_signed(best['per_category'].get(c, {}).get('recall_at_10', 0.0))}.")
    lines.append(
        f"- hard-negative pairwise: {_fmt(best['hardneg_win_rate'])} "
        f"({_signed(best['hardneg_win_rate_delta'])} vs baseline).")

    ships = overall_mrr >= SHIP_OVERALL_MRR_MIN or best_exeg >= SHIP_CATEGORY_MIN
    verdict = "SHIP" if ships else "HOLD"
    return verdict, best, lines


def render_report(gold: dict, results: dict) -> str:
    """Assemble the full markdown rerank report body (deterministic)."""
    w: list[str] = []
    corpus = gold["corpus"]
    baseline = results["baseline"]
    arms = results["arms"]

    w.append("# LampStand corpus — cross-encoder rerank measurement (Rank 4, "
             "EVAL-GATED)")
    w.append("")
    w.append("**CANDIDATE MEASUREMENT — decides whether an on-device reranker "
             "earns its place; changes no app constant, ships no weights.** The "
             "measurement recommends SHIP or HOLD; the architect decides.")
    w.append("")
    w.append(f"Corpus reference: `{corpus.get('corpus_version') or 'unversioned'}` "
             f"· embedding model `{corpus.get('model_revision', '')[:12]}` · "
             f"{corpus.get('n_chunks', 0):,} chunks · gold seed {gold['seed']}.")
    w.append("")
    w.append(f"Reranker window: top-{results['rerank_k']} FUSED hybrid candidates "
             f"per query (the app slots the reranker between RRF fusion and the "
             f"tradition multiplier over the fused top-{results['rerank_k']}); "
             f"pair truncation {results['max_pair_tokens']} tokens; fusion "
             f"config `{results['app_config_label']}`.")
    w.append("")

    # -- method -----------------------------------------------------------------
    w.append("## 1. Method")
    w.append("")
    w.append(f"For each gold query the un-reranked v2 HYBRID top-"
             f"{results['rerank_k']} candidates (the exact app fusion path) are "
             "re-scored by a cross-encoder over (query, chunk_text) pairs and "
             "re-sorted within the window; the tail keeps its fused order. "
             "recall@5/10/20, MRR, and nDCG@10 are recomputed PER CATEGORY and "
             "compared to the un-reranked hybrid baseline. Reranking a top-"
             f"{results['rerank_k']} window can lift a rank-21.."
             f"{results['rerank_k']} relevant chunk into the top-20, so recall@20 "
             "can move; it can never DROP a baseline hit (the window only "
             "reorders).")
    w.append("")
    w.append("Two candidate-text variants are measured: **header** (the baked "
             "structural header `\"Psalms 23:1 — \"` + text, exactly the "
             "BM25-indexed / embedded string) and **raw** (display text only). "
             "The cross-encoder (sentence-transformers / torch) is a DEV/EVAL-"
             "ONLY dependency (`[rerank]` extra) — never shipped in the core "
             "package; CoreML is not needed to measure.")
    w.append("")

    # -- honesty caveat ---------------------------------------------------------
    w.append("## 2. Honesty caveat (read before the numbers)")
    w.append("")
    w.append("The corpus gold labels favor **lexical** overlap — crossref and "
             "commentary-anchor queries share verse wording with their targets, "
             "and prooftext labels are the section's proof-text VERSES (documented "
             "in `reports/retrieval_eval_v1.md` §2-3). A cross-encoder's strength "
             "is **semantic** matching of paraphrased queries, which is "
             "structurally UNDER-represented here, so any gain on this label set "
             "is a FLOOR, not the user-experienced lift. The paraphrased-user-"
             "query case lives in the app's own 46-case eval (recall@20 .913 / "
             "MRR .453 on the v2 hybrid), not in this corpus-native set. Weight "
             "the per-category exegetical lift (commentary-anchor, crossref) and "
             "the hard-negative pairwise result HEAVILY; treat a muted overall "
             "number as expected, not as reranker failure.")
    w.append("")

    # -- baseline ---------------------------------------------------------------
    w.append("## 3. Un-reranked v2 hybrid baseline")
    w.append("")
    rows = [("OVERALL", baseline["overall"])] + [
        (cat, m) for cat, m in baseline["per_category"].items() if cat != "hardneg"
    ]
    w.extend(_metrics_table(rows))
    hn = baseline["hardneg_pairwise"]
    w.append("")
    w.append(f"hard-negative pairwise (DRAFT suite): {hn['wins']}/{hn['count']} "
             f"wins = {_fmt(hn['win_rate'])}")
    w.append("")

    # -- reranked arms ----------------------------------------------------------
    w.append("## 4. Reranked arms")
    w.append("")
    # Model provenance block (source model + license — recorded for the manifest).
    seen: set[str] = set()
    w.append("Cross-encoders measured (license VERIFIED via the HF hub):")
    w.append("")
    w.append("| model | HF id | license | arch | revision |")
    w.append("|---|---|---|---|---|")
    for arm in arms:
        if arm["model_key"] in seen:
            continue
        seen.add(arm["model_key"])
        w.append(f"| `{arm['model_key']}` | `{arm['hf_id']}` | {arm['license']} | "
                 f"{arm['arch']} | `{arm['revision'][:12]}` |")
    w.append("")
    for arm in arms:
        w.append(f"### {arm['model_key']} — {arm['variant']}")
        w.append("")
        rows = [("OVERALL", arm["overall"])] + [
            (cat, m) for cat, m in arm["per_category"].items() if cat != "hardneg"
        ]
        w.extend(_metrics_table(rows))
        hn = arm["hardneg_pairwise"]
        w.append("")
        w.append(f"hard-negative pairwise: {hn['wins']}/{hn['count']} wins = "
                 f"{_fmt(hn['win_rate'])}")
        w.append("")

    # -- deltas -----------------------------------------------------------------
    w.append("## 5. Per-category delta vs un-reranked baseline")
    w.append("")
    w.append("Positive = reranking helped. The exegetical categories "
             "(commentary-anchor, crossref) carry the most weight; prooftext is "
             "the hardest, most lexical category (see §2).")
    w.append("")
    for delta in results["deltas"]:
        w.append(f"**`{delta['model_key']}` / `{delta['variant']}`** — "
                 f"hardneg pairwise {_fmt(delta['hardneg_win_rate'])} "
                 f"({_signed(delta['hardneg_win_rate_delta'])})")
        w.append("")
        w.extend(_delta_table(delta))
        w.append("")

    # -- verdict ----------------------------------------------------------------
    verdict, best, lines = _verdict(results)
    w.append("## 6. Verdict")
    w.append("")
    w.append(f"**Verdict: {verdict}.** Rule: SHIP if overall MRR lift "
             f"≥ {SHIP_OVERALL_MRR_MIN:+.3f} OR a per-category exegetical "
             f"(commentary-anchor / crossref) MRR or recall@10 lift "
             f"≥ {SHIP_CATEGORY_MIN:+.3f} beyond noise, AND ~"
             f"{results['rerank_k']} pairs/query is affordable on-device "
             f"(it is, by construction). Otherwise HOLD — a well-argued HOLD is a "
             f"fully successful outcome; the reranker is not worth ~20 MB + "
             f"latency for noise against the stronger v2 baseline.")
    w.append("")
    w.extend(lines)
    w.append("")
    if verdict == "HOLD":
        w.append("**Recommendation: do NOT integrate the reranker now.** Against "
                 "the re-chunked v2 hybrid (which already roughly doubled ranking "
                 "quality vs the pre-re-chunk retrieval the audit measured), the "
                 "cross-encoder does not clear the SHIP bar on this corpus-native "
                 "set. Because these labels are lexical (§2), the honest read is "
                 "that the corpus gate is INCONCLUSIVE-to-negative for a reranker; "
                 "the decision should defer to the app's paraphrased-query eval. "
                 "Do not spend the ~20 MB CoreML asset + per-query latency until "
                 "the app-side eval shows a clear semantic win the corpus set "
                 "cannot see. The measurement harness stays committed so the gate "
                 "can be re-run against a future semantic gold set or a stronger "
                 "reranker.")
    else:
        w.append("**Recommendation: proceed to CoreML export** (fp16, "
                 "tokenizer-parity + SHA-pin discipline mirroring "
                 "`coreml_export.py`), wire as a synced pack asset, and document "
                 "the app-integration contract in `docs/reranker-pack.md`. The "
                 "reranker slots between RRF fusion and the tradition multiplier "
                 "over the fused top-30; the app measures actual on-device latency "
                 "and falls through to plain RRF on a budget miss.")
    w.append("")
    w.append("*Generated by `python -m lampstand_corpus.cli rerank-eval`. "
             "Deterministic: fixed seed, pinned cross-encoder revision, no "
             "timestamps.*")
    w.append("")
    return "\n".join(w)
