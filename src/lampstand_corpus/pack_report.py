"""Render ``reports/pack_diet_v1.md`` — measured pack sizes + int8 quality delta.

Deterministic (no timestamps). The BEFORE sizes are the committed v1 packaging
measurements from ``reports/corpus_validation_v1.md`` (corpus-v1.0.0-candidate)
— they are constants here because the v1 pack format no longer exists in the
pipeline and cannot be re-produced; their provenance is the committed report.
"""

from __future__ import annotations

PACK_DIET_REPORT_FILENAME = "pack_diet_v1.md"

# v1 sizes measured for corpus-v1.0.0-candidate (reports/corpus_validation_v1.md §2).
V1_SIZES = {
    "bundled_search.sqlite": 60_252_160,
    "ondemand_embeddings.sqlite": 1_990_508_544,
}

_METRIC_COLS = ("recall_at_5", "recall_at_10", "recall_at_20", "mrr", "ndcg_at_10")
_METRIC_HEADS = ("recall@5", "recall@10", "recall@20", "MRR", "nDCG@10")


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):,.1f} MB"


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _quant_delta_rows(results: dict) -> list[str]:
    """fp32-vs-int8 table rows for the dense-bearing arms, per category."""
    w: list[str] = []
    w.append("| arm | set | " + " | ".join(_METRIC_HEADS)
             + " | Δ vs fp32 (r@20 / MRR / nDCG@10) |")
    w.append("|---|---|" + "---:|" * (len(_METRIC_HEADS) + 1))
    for base in ("dense", "hybrid"):
        q = results["arms"].get(f"{base}-int8")
        f = results["arms"].get(base)
        if not (q and f):
            continue
        sets = [("OVERALL", "overall")] + [
            (cat, cat) for cat in sorted(f["per_category"]) if cat != "hardneg"]
        for label, key in sets:
            fm = f["overall"] if key == "overall" else f["per_category"][key]
            qm = q["overall"] if key == "overall" else q["per_category"][key]
            cells = " | ".join(_fmt(qm[c]) for c in _METRIC_COLS)
            delta = (f"{qm['recall_at_20'] - fm['recall_at_20']:+.3f} / "
                     f"{qm['mrr'] - fm['mrr']:+.3f} / "
                     f"{qm['ndcg_at_10'] - fm['ndcg_at_10']:+.3f}")
            w.append(f"| {base}-int8 | {label} | {cells} | {delta} |")
    return w


CROSSREF_REPORT_FILENAME = "crossref_pack_v1.md"


def _arm_row(name: str, m: dict) -> str:
    cells = " | ".join(_fmt(m[c]) for c in _METRIC_COLS)
    return f"| {name} | {m['count']} | {cells} |"


def render_crossref_pack_report(
    *,
    corpus_version: str,
    pack_file,               # PackFile for bundled_crossrefs.sqlite
    eval_results: dict | None,
) -> str:
    """Render ``reports/crossref_pack_v1.md`` (sizes + graph-boost deltas)."""
    w: list[str] = []
    c = pack_file.contents
    w.append("# LampStand corpus — TSK cross-reference pack v1 (Rank 13, "
             "corpus half)")
    w.append("")
    w.append("**CANDIDATE MEASUREMENT — corpus half only.** The app lane builds "
             "the UI panel + the hybridContext graph boost against "
             "`docs/crossrefs-pack.md`.")
    w.append("")
    w.append(f"Corpus reference: `{corpus_version}`.")
    w.append("")
    w.append("## 1. Pack size + home")
    w.append("")
    w.append(f"`bundled_crossrefs.sqlite` = **{pack_file.bytes:,} B "
             f"({_mb(pack_file.bytes)})** — {c.get('n_edges', 0):,} resolving "
             f"TSK edges over {c.get('n_sources', 0):,} source verses, plus a "
             f"top-{c.get('expansion_top_n', 8)} expansion for "
             f"{c.get('n_expanded_chunks', 0):,} Scripture chunks.")
    w.append("")
    w.append("Pack home decision: a **separate tiny bundled pack**, not "
             "`bundled_search.sqlite` — small enough to always ship, the "
             "reader is single-purpose (never touches BM25/vector tables), and "
             "the crossref layer serves the reading panel + Ask grounding, not "
             "the search index. The full-fidelity `ondemand_crossrefs.sqlite` "
             "(24.2 MB, unchanged v1 schema) still ships on-demand.")
    w.append("")
    w.append(f"Attribution carried in pack meta (CC-BY requirement): "
             f"\"{c.get('attribution', '')}\".")
    w.append("")
    w.append("## 2. Graph-boost measurement (experimental arms)")
    w.append("")
    if eval_results and eval_results.get("graph_arms"):
        w.append("Post-fusion TSK boost over the hybrid arm at the app's "
                 "constants: the top-5 Scripture hits' cross-referenced "
                 "pericopes are RRF-fused in as a third list — `hybrid-graph` "
                 "at an equal retriever weight (the same weight the app gives "
                 "dense), `hybrid-graph-weak` at ⅓ weight. Same 452-query gold "
                 "set as `retrieval_eval_v1.md`; self-hit exclusions apply to "
                 "graph candidates too.")
        w.append("")
        arms = eval_results["arms"]
        for cat_label, key in (
            ("OVERALL", "overall"),
            ("crossref (CIRCULAR — see below)", "crossref"),
            ("prooftext", "prooftext"),
            ("commentary-anchor", "commentary-anchor"),
        ):
            w.append(f"### {cat_label}")
            w.append("")
            w.append("| arm | n | " + " | ".join(_METRIC_HEADS) + " |")
            w.append("|---|---:|" + "---:|" * len(_METRIC_HEADS))
            for arm in ("hybrid", "hybrid-graph", "hybrid-graph-weak"):
                m = (arms[arm]["overall"] if key == "overall"
                     else arms[arm]["per_category"][key])
                w.append(_arm_row(arm, m))
            w.append("")
        w.append("**Circularity caveat:** the `crossref` gold category's labels "
                 "ARE TSK-derived (relevant = chunks covering high-vote TSK "
                 "targets of the query's source verse), so any gain there from "
                 "a TSK-derived boost is partially true-by-construction — "
                 "reported for completeness, NOT evidence. The honest signal "
                 "for the boost is `prooftext` + `commentary-anchor` (labels "
                 "from confession proof-texts and commentary anchors, "
                 "independent of TSK).")
        w.append("")
    else:
        w.append("_Not yet measured — run `validate-retrieval` (graph arms run "
                 "automatically when crossrefs.sqlite is built), then re-run "
                 "`package`._")
        w.append("")
    w.append("## 3. Schema contract")
    w.append("")
    w.append("See **`docs/crossrefs-pack.md`** (verse-key arithmetic, target/"
             "neighbor blob encodings, aggregation rules, reader sketches).")
    w.append("")
    w.append("*Generated by `python -m lampstand_corpus.cli package`. "
             "Deterministic: no timestamps.*")
    w.append("")
    return "\n".join(w)


def render_pack_diet_report(
    *,
    corpus_version: str,
    files: list,          # PackagingResult.files
    flags: list[str],
    eval_results: dict | None,
) -> str:
    by_name = {f.name: f for f in files}
    w: list[str] = []
    w.append("# LampStand corpus — pack diet v1 (integer ids, posting blobs, "
             "int8 vectors)")
    w.append("")
    w.append("**CANDIDATE MEASUREMENT — corpus half only.** The app-side "
             "pack-resolution migration happens in the app repo against the "
             "contract in `docs/pack-diet.md`.")
    w.append("")
    w.append(f"Corpus reference: `{corpus_version}`.")
    w.append("")

    # -- why -----------------------------------------------------------------
    w.append("## 1. Why (verified accounting of the v1 pack)")
    w.append("")
    w.append("`ondemand_embeddings.sqlite` (v1) was 1,990,508,544 B. Measured "
             "against the built DB:")
    w.append("")
    w.append("- BM25 postings: 10,831,300 rows, each repeating a 28-char TEXT "
             "chunk id — **289 MB of id strings per copy**, stored ~3× (table "
             "row + `(term_id, chunk_id)` PK index + `idx_posting_chunk`) plus "
             "per-row b-tree overhead ⇒ ~1.6 GB for ~tens of MB of information "
             "(df/tf/gaps).")
    w.append("- Dense vectors: 257 MB float32 (175,442 × 384 × 4 B).")
    w.append("- Chunk display text: 102 MB.")
    w.append("")
    w.append("v2 fixes the representation: stable **integer** chunk ids, one "
             "varint-delta posting BLOB per term, `bm25_doc` folded into "
             "`chunk.doc_len`, vectors split into their own pack and stored "
             "int8 with a per-vector scale (float32 available behind the "
             "`package fp32` flag).")
    w.append("")

    # -- sizes ---------------------------------------------------------------
    w.append("## 2. Measured pack sizes (before → after)")
    w.append("")
    w.append("| pack | v1 bytes | v2 bytes | change |")
    w.append("|---|---:|---:|---|")
    v1_search = V1_SIZES["ondemand_embeddings.sqlite"]
    sp = by_name.get("ondemand_search.sqlite")
    vp = by_name.get("ondemand_vectors.sqlite")
    if sp and vp:
        after = sp.bytes + vp.bytes
        w.append(f"| ondemand_embeddings → search + vectors | {v1_search:,} "
                 f"({_mb(v1_search)}) | {after:,} ({_mb(after)}) | "
                 f"×{v1_search / after:.1f} smaller |")
        w.append(f"| — ondemand_search.sqlite (BM25 + metadata + text) | | "
                 f"{sp.bytes:,} ({_mb(sp.bytes)}) | |")
        w.append(f"| — ondemand_vectors.sqlite (dense) | | "
                 f"{vp.bytes:,} ({_mb(vp.bytes)}) | |")
    bs = by_name.get("bundled_search.sqlite")
    if bs:
        v1_b = V1_SIZES["bundled_search.sqlite"]
        w.append(f"| bundled_search.sqlite | {v1_b:,} ({_mb(v1_b)}) | "
                 f"{bs.bytes:,} ({_mb(bs.bytes)}) | ×{v1_b / bs.bytes:.1f} "
                 f"smaller |")
    w.append("")
    w.append("(v1 sizes: committed `reports/corpus_validation_v1.md` §2, "
             "corpus-v1.0.0-candidate. Other packs are unchanged by the diet.)")
    w.append("")

    # -- display-text decision --------------------------------------------------
    w.append("## 3. Display-text decision")
    w.append("")
    if sp is not None:
        over = sp.bytes > 250 * 1024 * 1024
        if over:
            w.append(f"`ondemand_search.sqlite` measured {_mb(sp.bytes)} WITH "
                     "display text — **over the ~250 MB line; FLAGGED for the "
                     "architect** (fallback: drop `chunk.text` and resolve "
                     "display text from the per-resource packs at render time).")
        else:
            w.append(f"Display text (102 MB raw) is INCLUDED: the search pack "
                     f"measured {_mb(sp.bytes)}, under the ~250 MB line, and "
                     "keeping it makes the pack self-sufficient for rendering "
                     "search results (no join against five per-resource packs "
                     "on the hot path).")
    w.append("")

    # -- quantization quality --------------------------------------------------
    w.append("## 4. int8 quantization quality (measured by the retrieval eval)")
    w.append("")
    if eval_results and eval_results.get("int8_arms"):
        w.append("Dense-bearing arms re-ranked against the int8 "
                 "quantize→dequantize round-trip of the corpus matrix "
                 "(mathematically identical to scoring the shipped int8 pack), "
                 "same 452-query gold set as `retrieval_eval_v1.md`:")
        w.append("")
        w.extend(_quant_delta_rows(eval_results))
        w.append("")
    else:
        w.append("_Not yet measured — run `validate-retrieval` (int8 arms are "
                 "included automatically), then re-run `package`._")
        w.append("")

    # -- contract + flags ---------------------------------------------------------
    w.append("## 5. Schema contract")
    w.append("")
    w.append("See **`docs/pack-diet.md`** — the authoritative v2 table/column/"
             "encoding contract for the app-side reader migration (other lane).")
    w.append("")
    w.append("## 6. Flags")
    w.append("")
    if flags:
        for fl in flags:
            w.append(f"- FLAG: {fl}")
    else:
        w.append("- none (bundled + search packs within their targets)")
    w.append("")
    w.append("*Generated by `python -m lampstand_corpus.cli package`. "
             "Deterministic: no timestamps; sizes are of the freshly built "
             "packs.*")
    w.append("")
    return "\n".join(w)
