# LampStand corpus — retrieval eval v1 (F5 measurement foundation)

**CANDIDATE MEASUREMENT — gates retrieval tuning; changes no app constant.** The sweep recommends; the architect decides.

Corpus reference: `corpus-v2.0.0-candidate` · model `5c38ec7c405e` · 317,204 chunks · gold seed 613.

## 1. Gold set

Zero-annotation labels from the corpus itself; every query records its own source chunk(s) as EXCLUDED so a verbatim self-hit cannot occupy rank 1 (hardneg queries exclude nothing — their relevant chunk contains the question).

| category | n | query | relevant |
|---|---:|---|---|
| prooftext | 150 | catechism question / confession opening sentence (citations stripped) | chunks covering the section's proof-text verses |
| crossref | 150 | source verse text (BSB) | chunks covering TSK targets with votes ≥ 116 (data-driven threshold) |
| commentary-anchor | 152 | commentary paragraph (word-boundary truncated) | Scripture chunks at its verse anchor |
| hardneg | 60 | WSC/WLC question A | A's Q&A chunk (hard negative: doctrinally-adjacent B's chunk) — **architect-approved** (`data/eval/hard_negatives_v1.json`; advisor spot-check still recommended) |

> **Labels-grew caveat.** The Belgic Confession's proof-texts were added to `confessions.sqlite` from CCEL schaff/creeds3, so the prooftext gold set now includes Belgic articles (6 Belgic queries; prooftext n 149 → 150). More labels usually LOWER apparent scores (the denominator grows and the new queries are hard corpus-native proof-text queries), so a small dip here is EXPECTED and is NOT a retrieval regression — the retriever, packs, and fusion constants are byte-identical to the prior run; only the measured label set changed.

Metric definitions match the app eval (RetrievalEvalTests): recall@k = share of queries with ≥1 relevant chunk in the top k; MRR capped at rank 20; nDCG@10 binary-gain (corpus-side addition). hardneg queries are scored separately (pairwise) and excluded from the overall row.

## 2. Arms at the app's shipped constants

Fusion config: `k=60 perType=20 denseDepth=20 rawFetch=80` (HybridRetriever.swift). BM25 arm = the app's per-type-balanced lexical ranking; dense arm = the app's deduped dense contribution (depth 20 — a deeper dense-only arm cannot change any top-20 metric, since single-list RRF is rank-preserving); hybrid = RRF fusion of both, exactly as `hybridContext` fuses.

Labels are CONSERVATIVE (zero-annotation): only the derived chunks count as relevant. A doctrinally-correct sibling hit — e.g. WSC 77 'ninth commandment' retrieved for a Heidelberg ninth-commandment question whose labels are its proof-text VERSES — scores as a miss. Absolute numbers are therefore floors, not user-experienced quality; ARM-VS-ARM deltas on identical labels are the meaningful signal.

### bm25

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.190 | 0.237 | 0.296 | 0.141 | 0.055 |
| commentary-anchor | 152 | 0.289 | 0.342 | 0.408 | 0.219 | 0.089 |
| crossref | 150 | 0.247 | 0.320 | 0.393 | 0.192 | 0.073 |
| prooftext | 150 | 0.033 | 0.047 | 0.087 | 0.011 | 0.004 |

hard-negative pairwise (DRAFT suite): 58/60 wins = 0.967

### dense

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.113 | 0.153 | 0.223 | 0.080 | 0.030 |
| commentary-anchor | 152 | 0.112 | 0.158 | 0.211 | 0.078 | 0.035 |
| crossref | 150 | 0.220 | 0.260 | 0.387 | 0.151 | 0.051 |
| prooftext | 150 | 0.007 | 0.040 | 0.073 | 0.010 | 0.004 |

hard-negative pairwise (DRAFT suite): 60/60 wins = 1.000

### hybrid

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.128 | 0.223 | 0.296 | 0.084 | 0.042 |
| commentary-anchor | 152 | 0.151 | 0.329 | 0.408 | 0.086 | 0.060 |
| crossref | 150 | 0.233 | 0.313 | 0.387 | 0.158 | 0.065 |
| prooftext | 150 | 0.000 | 0.027 | 0.093 | 0.008 | 0.002 |

hard-negative pairwise (DRAFT suite): 60/60 wins = 1.000

## 3. F5 verdict — does dense retrieval justify its pack?

**Verdict: MARGINAL.** Rule: hybrid must beat BM25-only by ≥ 2% absolute on some headline metric or category for JUSTIFIED; ≥ 0.5% for MARGINAL.

- overall recall@5: hybrid − BM25 = -0.062
- overall recall@10: hybrid − BM25 = -0.013
- overall recall@20: hybrid − BM25 = +0.000
- overall MRR: hybrid − BM25 = -0.057
- overall nDCG@10: hybrid − BM25 = -0.013
- commentary-anchor: recall@20 +0.000, MRR -0.134
- crossref: recall@20 -0.007, MRR -0.034
- prooftext: recall@20 +0.007, MRR -0.003
- hardneg pairwise wins (DRAFT suite): bm25 58/60, dense 60/60, hybrid 60/60

Context for the verdict: these corpus-native labels favor lexical overlap (crossref and commentary-anchor queries share verse wording with their targets), while dense retrieval's known strength — short paraphrased USER queries (the app's ~46-case eval, recall@20≈0.826) — is structurally under-represented here. This verdict gates the corpus-side evidence only; the app-side eval remains the user-experience gate, and the two should be read together before any decision about the ~1.9 GB dense pack.

## 4. Fusion-constant sweep

Grid: rrfK × bm25PerType × denseDepth (rawFetch = 4×depth, floored at the app's 80), hybrid arm, overall metrics. **Recommendation only — no app constant is changed here.**

Baseline (app): `k=60 perType=20 denseDepth=20 rawFetch=80`

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| app baseline | 452 | 0.133 | 0.235 | 0.308 | 0.088 | 0.044 |

Best config per metric (delta vs app baseline):

| optimizing | config | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 | Δ target |
|---|---|---:|---:|---:|---:|---:|---:|
| recall@20 | `k=20 perType=10 denseDepth=10 rawFetch=80` | 0.168 | 0.243 | 0.334 | 0.097 | 0.048 | +0.027 |
| MRR | `k=20 perType=10 denseDepth=10 rawFetch=80` | 0.168 | 0.243 | 0.334 | 0.097 | 0.048 | +0.009 |
| nDCG@10 | `k=20 perType=10 denseDepth=10 rawFetch=80` | 0.168 | 0.243 | 0.334 | 0.097 | 0.048 | +0.005 |

Weighted-RRF λ exploration (NOT an app knob today; λ=0.5 ≡ the app's unweighted fusion), on the nDCG@10 winner's knobs:

| λ | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.170 | 0.243 | 0.363 | 0.107 | 0.045 |
| 0.4 | 0.168 | 0.243 | 0.372 | 0.106 | 0.044 |
| 0.5 | 0.168 | 0.243 | 0.334 | 0.097 | 0.048 |
| 0.6 | 0.115 | 0.162 | 0.334 | 0.079 | 0.028 |
| 0.7 | 0.115 | 0.162 | 0.334 | 0.076 | 0.028 |

### Honesty check — BM25-only across the per-type limit

Any hybrid "win" must be read against BM25 ALONE at the same per-type limit; a gain that survives here is fusion's, one that doesn't came from the tighter limit:

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| bm25-only perType=10 | 452 | 0.199 | 0.243 | 0.365 | 0.152 | 0.056 |
| bm25-only perType=20 | 452 | 0.199 | 0.248 | 0.305 | 0.148 | 0.058 |
| bm25-only perType=40 | 452 | 0.199 | 0.248 | 0.308 | 0.148 | 0.058 |
| bm25-only perType=60 | 452 | 0.199 | 0.248 | 0.308 | 0.148 | 0.058 |

Best swept hybrid recall@20 = 0.334; best BM25-only recall@20 = 0.365 — the swept hybrid does NOT beat BM25-only on these labels.

Caveat on the per-type limit: relevant chunks in all three scored categories are SCRIPTURE, so a tighter per-type limit mechanically favors these labels by squeezing commentary/lexicon out of the top-20 window. A user query like "propitiation" WANTS commentary and lexicon rows; do not lower bm25PerType on this evidence alone.

## 4b. Query expansion (Rank 7, measured)

BM25 query terms also score their mined expansions (INCLUDING the 100 advisor-approved theological synonyms — archaic↔modern doctrinal-vocabulary residue: sepulchre↔tomb, candlestick↔lampstand, oblation↔offering, devils↔demons, …) at a flat 0.3 weight. Baseline arms repeated for comparison:

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 452 | 0.190 | 0.237 | 0.296 | 0.141 | 0.055 |
| bm25-expand | 452 | 0.173 | 0.232 | 0.292 | 0.137 | 0.054 |
| hybrid | 452 | 0.128 | 0.223 | 0.296 | 0.084 | 0.042 |
| hybrid-expand | 452 | 0.128 | 0.206 | 0.296 | 0.082 | 0.040 |

Per-category delta (expand − base), the categories where synonym matching should help most:

| arm pair | category | Δrecall@5 | Δrecall@10 | Δrecall@20 | ΔMRR | ΔnDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| bm25→bm25-expand | commentary-anchor | -0.013 | +0.000 | +0.000 | -0.005 | -0.002 |
| bm25→bm25-expand | crossref | -0.013 | +0.000 | -0.007 | -0.006 | -0.003 |
| bm25→bm25-expand | prooftext | -0.027 | -0.013 | -0.007 | -0.002 | -0.001 |
| hybrid→hybrid-expand | commentary-anchor | +0.000 | -0.026 | +0.007 | -0.004 | -0.004 |
| hybrid→hybrid-expand | crossref | +0.000 | -0.013 | +0.007 | -0.001 | -0.001 |
| hybrid→hybrid-expand | prooftext | +0.000 | -0.013 | -0.013 | -0.001 | -0.001 |

**Query-expansion decision (synonym-inclusive): KEEP OFF for retrieval.** Best overall-headline (recall@20 / MRR) expand-vs-base gain = +0.000 (bar: ≥ +0.005 on a headline metric with no headline regression on the same arm).

Every headline and per-category delta is null-to-negative: wiring query expansion ON does NOT beat the v2 baseline on these corpus-native labels, even WITH the advisor-approved theological synonyms. The gold queries are drawn from MODERN-vocabulary corpus text (BSB verses, commentary paragraphs), so archaic↔modern synonym firing is structurally rare here — the same labels-favor-lexical caveat as §3. **Recommendation: keep the approved synonym table (it still powers the app's tap-to-gloss UX) but leave query expansion OFF for retrieval scoring.** The expansion table + the advisor-approved synonyms remain shipped in the search pack for the gloss/lookup path; only the query-time BM25 down-weighted expansion stays disabled. Revisit if a future gold set includes archaic-phrased (KJV-style) user queries, where the synonyms should finally earn their weight.

## 5. retrieve.py ↔ app parity notes

- `src/lampstand_corpus/retrieve.py` is a dense-only SMOKE helper (global argsort over all chunks; no per-type balance, no Scripture dedup, no BM25, no RRF). It is NOT the app's ranking path. The eval runner (`eval_retrieval.py`) implements the app path faithfully; retrieve.py is left untouched for the P6 smoke report.
- The pipeline records BM25 k1/b/avgdl/N in `bm25_stats` but NOT the query-time IDF variant; the app's documented choice (Lucene non-negative `ln((N−df+0.5)/(df+0.5)+1)`) is mirrored here. Recommend the pipeline record `idf_form` in `bm25_stats`/meta so the contract is explicit.
- Fusion is ASYMMETRIC in the app: the full deduped BM25 list (up to 4×perType entries) enters RRF, while dense contributes only its top `denseDepth`. Mirrored exactly; worth an explicit comment in HybridRetriever if it is intentional.
- The app's dense TopK admits raw hits on `score > worst` while streaming; at an exact float score tie at the admission boundary the kept id can differ from this runner's full-sort (score desc, id asc). Practically unreachable with float dot products.

## 6. Flags for the architect / advisor

- Hard-negative suite (`data/eval/hard_negatives_v1.json`) is **architect-approved** (2026-07-02); a theological-advisor spot-check is still recommended before public ship, since pairs were generated by answer token overlap and some may be doctrinally trivial or too close.
- The prooftext category now draws from WCF / WLC / WSC / LBCF / Heidelberg / Dort / Belgic (Belgic proofs added from CCEL schaff/creeds3: 34/37 articles, arts 4-6 the canon list carry none).
- gold-builder note: hardneg: suite approved (by architect); advisor spot-check still recommended before public ship

*Generated by `python -m lampstand_corpus.cli validate-retrieval` / `sweep-retrieval`. Deterministic: fixed seed, no timestamps.*
