# LampStand corpus — retrieval eval v1 (F5 measurement foundation)

**CANDIDATE MEASUREMENT — gates retrieval tuning; changes no app constant.** The sweep recommends; the architect decides.

Corpus reference: `corpus-v1.0.0-candidate` · model `5c38ec7c405e` · 317,204 chunks · gold seed 613.

## 1. Gold set

Zero-annotation labels from the corpus itself; every query records its own source chunk(s) as EXCLUDED so a verbatim self-hit cannot occupy rank 1 (hardneg queries exclude nothing — their relevant chunk contains the question).

| category | n | query | relevant |
|---|---:|---|---|
| prooftext | 150 | catechism question / confession opening sentence (citations stripped) | chunks covering the section's proof-text verses |
| crossref | 150 | source verse text (BSB) | chunks covering TSK targets with votes ≥ 116 (data-driven threshold) |
| commentary-anchor | 152 | commentary paragraph (word-boundary truncated) | Scripture chunks at its verse anchor |
| hardneg | 60 | WSC/WLC question A | A's Q&A chunk (hard negative: doctrinally-adjacent B's chunk) — **DRAFT, pending theological-advisor review** (`data/eval/hard_negatives_v1.json`) |

Metric definitions match the app eval (RetrievalEvalTests): recall@k = share of queries with ≥1 relevant chunk in the top k; MRR capped at rank 20; nDCG@10 binary-gain (corpus-side addition). hardneg queries are scored separately (pairwise) and excluded from the overall row.

## 2. Arms at the app's shipped constants

Fusion config: `k=60 perType=20 denseDepth=20 rawFetch=80` (HybridRetriever.swift). BM25 arm = the app's per-type-balanced lexical ranking; dense arm = the app's deduped dense contribution (depth 20 — a deeper dense-only arm cannot change any top-20 metric, since single-list RRF is rank-preserving); hybrid = RRF fusion of both, exactly as `hybridContext` fuses.

Labels are CONSERVATIVE (zero-annotation): only the derived chunks count as relevant. A doctrinally-correct sibling hit — e.g. WSC 77 'ninth commandment' retrieved for a Heidelberg ninth-commandment question whose labels are its proof-text VERSES — scores as a miss. Absolute numbers are therefore floors, not user-experienced quality; ARM-VS-ARM deltas on identical labels are the meaningful signal.

### bm25

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.199 | 0.248 | 0.305 | 0.148 | 0.058 |
| commentary-anchor | 152 | 0.289 | 0.342 | 0.408 | 0.219 | 0.089 |
| crossref | 150 | 0.247 | 0.320 | 0.393 | 0.192 | 0.073 |
| prooftext | 150 | 0.060 | 0.080 | 0.113 | 0.031 | 0.011 |

hard-negative pairwise (DRAFT suite): 58/60 wins = 0.967

### dense

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.119 | 0.162 | 0.235 | 0.086 | 0.032 |
| commentary-anchor | 152 | 0.112 | 0.158 | 0.211 | 0.078 | 0.035 |
| crossref | 150 | 0.220 | 0.260 | 0.387 | 0.151 | 0.051 |
| prooftext | 150 | 0.027 | 0.067 | 0.107 | 0.028 | 0.010 |

hard-negative pairwise (DRAFT suite): 60/60 wins = 1.000

### hybrid

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.133 | 0.235 | 0.308 | 0.088 | 0.044 |
| commentary-anchor | 152 | 0.151 | 0.329 | 0.408 | 0.086 | 0.060 |
| crossref | 150 | 0.233 | 0.313 | 0.387 | 0.158 | 0.065 |
| prooftext | 150 | 0.013 | 0.060 | 0.127 | 0.019 | 0.006 |

hard-negative pairwise (DRAFT suite): 60/60 wins = 1.000

## 3. F5 verdict — does dense retrieval justify its pack?

**Verdict: MARGINAL.** Rule: hybrid must beat BM25-only by ≥ 2% absolute on some headline metric or category for JUSTIFIED; ≥ 0.5% for MARGINAL.

- overall recall@5: hybrid − BM25 = -0.066
- overall recall@10: hybrid − BM25 = -0.013
- overall recall@20: hybrid − BM25 = +0.002
- overall MRR: hybrid − BM25 = -0.060
- overall nDCG@10: hybrid − BM25 = -0.014
- commentary-anchor: recall@20 +0.000, MRR -0.134
- crossref: recall@20 -0.007, MRR -0.034
- prooftext: recall@20 +0.013, MRR -0.011
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

BM25 query terms also score their mined expansions (archaic↔modern pairs + suffix classes; approved theological synonyms only) at a flat 0.3 weight. Baseline arms repeated for comparison:

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 452 | 0.199 | 0.248 | 0.305 | 0.148 | 0.058 |
| bm25-expand | 452 | 0.184 | 0.243 | 0.301 | 0.143 | 0.056 |
| hybrid | 452 | 0.133 | 0.235 | 0.308 | 0.088 | 0.044 |
| hybrid-expand | 452 | 0.133 | 0.217 | 0.308 | 0.086 | 0.042 |

## 5. retrieve.py ↔ app parity notes

- `src/lampstand_corpus/retrieve.py` is a dense-only SMOKE helper (global argsort over all chunks; no per-type balance, no Scripture dedup, no BM25, no RRF). It is NOT the app's ranking path. The eval runner (`eval_retrieval.py`) implements the app path faithfully; retrieve.py is left untouched for the P6 smoke report.
- The pipeline records BM25 k1/b/avgdl/N in `bm25_stats` but NOT the query-time IDF variant; the app's documented choice (Lucene non-negative `ln((N−df+0.5)/(df+0.5)+1)`) is mirrored here. Recommend the pipeline record `idf_form` in `bm25_stats`/meta so the contract is explicit.
- Fusion is ASYMMETRIC in the app: the full deduped BM25 list (up to 4×perType entries) enters RRF, while dense contributes only its top `denseDepth`. Mirrored exactly; worth an explicit comment in HybridRetriever if it is intentional.
- The app's dense TopK admits raw hits on `score > worst` while streaming; at an exact float score tie at the admission boundary the kept id can differ from this runner's full-sort (score desc, id asc). Practically unreachable with float dot products.

## 6. Flags for the architect / advisor

- **DRAFT** hard-negative suite (`data/eval/hard_negatives_v1.json`) awaits theological-advisor review; pairs were generated by answer token overlap, so some may be doctrinally trivial or too close.
- WLC/WSC/Belgic/Dort sections carry NO proof_texts in `confessions.sqlite` (only WCF 172 / LBCF 159 / Heidelberg 124 do); the prooftext category therefore draws from those three documents.
- gold-builder note: hardneg: suite is DRAFT — pending theological-advisor review

*Generated by `python -m lampstand_corpus.cli validate-retrieval` / `sweep-retrieval`. Deterministic: fixed seed, no timestamps.*
