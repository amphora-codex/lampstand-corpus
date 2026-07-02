# LampStand corpus — cross-encoder rerank measurement (Rank 4, EVAL-GATED)

**CANDIDATE MEASUREMENT — decides whether an on-device reranker earns its place; changes no app constant, ships no weights.** The measurement recommends SHIP or HOLD; the architect decides.

Corpus reference: `corpus-v2.0.0-candidate` · embedding model `5c38ec7c405e` · 317,204 chunks · gold seed 613.

Reranker window: top-30 FUSED hybrid candidates per query (the app slots the reranker between RRF fusion and the tradition multiplier over the fused top-30); pair truncation 192 tokens; fusion config `k=60 perType=20 denseDepth=20 rawFetch=80`.

## 1. Method

For each gold query the un-reranked v2 HYBRID top-30 candidates (the exact app fusion path) are re-scored by a cross-encoder over (query, chunk_text) pairs and re-sorted within the window; the tail keeps its fused order. recall@5/10/20, MRR, and nDCG@10 are recomputed PER CATEGORY and compared to the un-reranked hybrid baseline. Reranking a top-30 window can lift a rank-21..30 relevant chunk into the top-20, so recall@20 can move; it can never DROP a baseline hit (the window only reorders).

Two candidate-text variants are measured: **header** (the baked structural header `"Psalms 23:1 — "` + text, exactly the BM25-indexed / embedded string) and **raw** (display text only). The cross-encoder (sentence-transformers / torch) is a DEV/EVAL-ONLY dependency (`[rerank]` extra) — never shipped in the core package; CoreML is not needed to measure.

## 2. Honesty caveat (read before the numbers)

The corpus gold labels favor **lexical** overlap — crossref and commentary-anchor queries share verse wording with their targets, and prooftext labels are the section's proof-text VERSES (documented in `reports/retrieval_eval_v1.md` §2-3). A cross-encoder's strength is **semantic** matching of paraphrased queries, which is structurally UNDER-represented here, so any gain on this label set is a FLOOR, not the user-experienced lift. The paraphrased-user-query case lives in the app's own 46-case eval (recall@20 .913 / MRR .453 on the v2 hybrid), not in this corpus-native set. Weight the per-category exegetical lift (commentary-anchor, crossref) and the hard-negative pairwise result HEAVILY; treat a muted overall number as expected, not as reranker failure.

## 3. Un-reranked v2 hybrid baseline

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.128 | 0.223 | 0.296 | 0.084 | 0.042 |
| commentary-anchor | 152 | 0.151 | 0.329 | 0.408 | 0.086 | 0.060 |
| crossref | 150 | 0.233 | 0.313 | 0.387 | 0.158 | 0.065 |
| prooftext | 150 | 0.000 | 0.027 | 0.093 | 0.008 | 0.002 |

hard-negative pairwise (DRAFT suite): 60/60 wins = 1.000

## 4. Reranked arms

Cross-encoders measured (license VERIFIED via the HF hub):

| model | HF id | license | arch | revision |
|---|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | apache-2.0 | BERT (MiniLM-L6) | `c5ee24cb1601` |
| `bge-reranker-base` | `BAAI/bge-reranker-base` | mit | XLM-RoBERTa base | `2cfc18c9415c` |

### ms-marco-MiniLM-L-6-v2 — header

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.188 | 0.246 | 0.314 | 0.128 | 0.060 |
| commentary-anchor | 152 | 0.342 | 0.401 | 0.441 | 0.228 | 0.113 |
| crossref | 150 | 0.200 | 0.280 | 0.400 | 0.144 | 0.062 |
| prooftext | 150 | 0.020 | 0.053 | 0.100 | 0.011 | 0.005 |

hard-negative pairwise: 60/60 wins = 1.000

### ms-marco-MiniLM-L-6-v2 — raw

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.148 | 0.230 | 0.310 | 0.102 | 0.050 |
| commentary-anchor | 152 | 0.257 | 0.368 | 0.428 | 0.154 | 0.085 |
| crossref | 150 | 0.173 | 0.267 | 0.400 | 0.139 | 0.060 |
| prooftext | 150 | 0.013 | 0.053 | 0.100 | 0.013 | 0.006 |

hard-negative pairwise: 60/60 wins = 1.000

### bge-reranker-base — header

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.237 | 0.285 | 0.325 | 0.167 | 0.076 |
| commentary-anchor | 152 | 0.388 | 0.434 | 0.441 | 0.292 | 0.140 |
| crossref | 150 | 0.253 | 0.340 | 0.427 | 0.175 | 0.077 |
| prooftext | 150 | 0.067 | 0.080 | 0.107 | 0.033 | 0.011 |

hard-negative pairwise: 60/60 wins = 1.000

### bge-reranker-base — raw

| set | n | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | 452 | 0.239 | 0.285 | 0.323 | 0.169 | 0.075 |
| commentary-anchor | 152 | 0.355 | 0.408 | 0.441 | 0.253 | 0.124 |
| crossref | 150 | 0.287 | 0.347 | 0.413 | 0.203 | 0.085 |
| prooftext | 150 | 0.073 | 0.100 | 0.113 | 0.051 | 0.017 |

hard-negative pairwise: 60/60 wins = 1.000

## 5. Per-category delta vs un-reranked baseline

Positive = reranking helped. The exegetical categories (commentary-anchor, crossref) carry the most weight; prooftext is the hardest, most lexical category (see §2).

**`ms-marco-MiniLM-L-6-v2` / `header`** — hardneg pairwise 1.000 (+0.000)

| set | Δrecall@5 | Δrecall@10 | Δrecall@20 | ΔMRR | ΔnDCG@10 |
|---|---:|---:|---:|---:|---:|
| OVERALL | +0.060 | +0.022 | +0.018 | +0.044 | +0.018 |
| commentary-anchor | +0.191 | +0.072 | +0.033 | +0.142 | +0.054 |
| crossref | -0.033 | -0.033 | +0.013 | -0.014 | -0.003 |
| prooftext | +0.020 | +0.027 | +0.007 | +0.003 | +0.004 |

**`ms-marco-MiniLM-L-6-v2` / `raw`** — hardneg pairwise 1.000 (+0.000)

| set | Δrecall@5 | Δrecall@10 | Δrecall@20 | ΔMRR | ΔnDCG@10 |
|---|---:|---:|---:|---:|---:|
| OVERALL | +0.020 | +0.007 | +0.013 | +0.018 | +0.008 |
| commentary-anchor | +0.105 | +0.039 | +0.020 | +0.068 | +0.025 |
| crossref | -0.060 | -0.047 | +0.013 | -0.019 | -0.005 |
| prooftext | +0.013 | +0.027 | +0.007 | +0.005 | +0.004 |

**`bge-reranker-base` / `header`** — hardneg pairwise 1.000 (+0.000)

| set | Δrecall@5 | Δrecall@10 | Δrecall@20 | ΔMRR | ΔnDCG@10 |
|---|---:|---:|---:|---:|---:|
| OVERALL | +0.108 | +0.062 | +0.029 | +0.083 | +0.034 |
| commentary-anchor | +0.237 | +0.105 | +0.033 | +0.206 | +0.081 |
| crossref | +0.020 | +0.027 | +0.040 | +0.016 | +0.012 |
| prooftext | +0.067 | +0.053 | +0.013 | +0.025 | +0.009 |

**`bge-reranker-base` / `raw`** — hardneg pairwise 1.000 (+0.000)

| set | Δrecall@5 | Δrecall@10 | Δrecall@20 | ΔMRR | ΔnDCG@10 |
|---|---:|---:|---:|---:|---:|
| OVERALL | +0.111 | +0.062 | +0.027 | +0.085 | +0.033 |
| commentary-anchor | +0.204 | +0.079 | +0.033 | +0.167 | +0.064 |
| crossref | +0.053 | +0.033 | +0.027 | +0.045 | +0.021 |
| prooftext | +0.073 | +0.073 | +0.020 | +0.043 | +0.015 |

## 6. Verdict

**Verdict: SHIP.** Rule: SHIP if overall MRR lift ≥ +0.030 OR a per-category exegetical (commentary-anchor / crossref) MRR or recall@10 lift ≥ +0.030 beyond noise, AND ~30 pairs/query is affordable on-device (it is, by construction). Otherwise HOLD — a well-argued HOLD is a fully successful outcome; the reranker is not worth ~20 MB + latency for noise against the stronger v2 baseline.

- best arm: `bge-reranker-base` / `header` (chosen by strongest exegetical-category MRR lift).
- overall MRR delta: +0.083 (SHIP floor +0.030).
- commentary-anchor: MRR +0.206, recall@10 +0.105.
- crossref: MRR +0.016, recall@10 +0.027.
- hard-negative pairwise: 1.000 (+0.000 vs baseline).

**Recommendation: proceed to CoreML export** (fp16, tokenizer-parity + SHA-pin discipline mirroring `coreml_export.py`), wire as a synced pack asset, and document the app-integration contract in `docs/reranker-pack.md`. The reranker slots between RRF fusion and the tradition multiplier over the fused top-30; the app measures actual on-device latency and falls through to plain RRF on a budget miss.

*Generated by `python -m lampstand_corpus.cli rerank-eval`. Deterministic: fixed seed, pinned cross-encoder revision, no timestamps.*
