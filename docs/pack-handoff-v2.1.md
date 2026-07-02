# Corpus freeze handoff — `corpus-v2.1.0` (app re-sync checklist)

**This is the single consolidated checklist for the app to re-sync a stable pack
set.** The corpus lane is FROZEN at this version. It tells you EXACTLY what to
fetch, what changed since the app's last sync (Rank-13), and the final per-pack
sizes + checksums to verify against.

- **Corpus version:** `corpus-v2.1.0-candidate` (the architect tags
  `corpus-v2.1.0` at ship; the pipeline never self-marks ship-ready —
  `manifest.ship_ready = false` until the 23-point spot-check passes).
- **Manifest of record:** `corpus_manifest.json` (committed). Pack `.sqlite` /
  `.mlpackage` files are gitignored and synced, never committed.
- **Rebuild determinism:** VERIFIED bit-identical. Re-running `package` three
  times (incl. across the version-string bump) produced byte-identical pack
  checksums — the tree hashes below are stable. Same source snapshots → same
  bytes (no timestamps, fixed seeds).

## What changed since the app's last sync (Rank-13 → v2.1)

Read this list against what the app currently has; each item is an app-side
action.

1. **Full 7-document confessions with proof_texts** (the app currently has a
   stale **WSC-only, zero-proof DRAFT**).
   - `bundled_confessions.sqlite` — WSC (the architect-locked bundled scope) now
     carries **107 sections with `proof_texts`** (was zero/DRAFT).
   - `ondemand_confessions.sqlite` — the other **6 documents**: WCF (172 proof
     sections), WLC (194), 1689/LBCF (159), Heidelberg/HC (124), Canons of
     Dort (42), **Belgic/BC (34)**. Westminster + Belgic proof-texts are the new
     data.
   - App action: re-sync BOTH confession packs; the proof-text UI + the
     prooftext retrieval labels now resolve across all 7 documents.

2. **Parent-text-NULL search packs** (`bundled_search.sqlite` /
   `ondemand_search.sqlite`, `meta.format = "search-pack-v2"`).
   - Pericope PARENT rows (`indexed=0`) store **`text = NULL`**; only children
     (`indexed=1`) carry display text.
   - App action: **reconstruct a parent's text from its children** —
     `SELECT text FROM chunk WHERE parent_id = ? AND indexed = 1
     ORDER BY chapter, verse_start`, joined with single spaces; prepend the
     parent's `header` if a labeled block is wanted. Do NOT checksum-validate
     parent rows. Full contract: `docs/pack-diet.md` (§"Dual granularity" +
     "CONTRACT NOTE — parent chunk `text` is NULL").

3. **Bundled cross-references** (`bundled_crossrefs.sqlite`, ~7.4 MB) — the TSK
   edge network (`crossref`) + per-pericope top-8 expansion (`chunk_crossref`)
   + confession proof-text edges, over the SAME integer chunk ids as the search
   pack. Contract: `docs/crossrefs-pack.md`. App action: sync it; it powers the
   reading-panel cross-refs + the `hybridContext` graph-boost candidates.

4. **`ondemand_vectors` is default-tier** (`download_group = "retrieval-index"`).
   - Fetch every `packs.on_demand.files[]` with `tier == "default"` on first
     launch (all on-demand packs today). Treat
     `download_group == "retrieval-index"` (`ondemand_search` + `ondemand_vectors`)
     as ONE unit: the dense arm needs `ondemand_vectors`; BM25 needs only
     `ondemand_search`. If vectors are missing, retrieval degrades **gracefully
     to BM25-only** — never gate the search pack on the vectors pack.
   - Manifest keys: `packs.on_demand.app_reader` / `default_note` /
     `tiers.default`.

5. **Reranker pack** (`packs.reranker`) — NEW on-device cross-encoder reranker.
   - `Reranker.mlpackage` (fp16, ~43.4 MB) + `reranker_vocab.txt`. Apache-2.0
     (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Loaded from disk (cpuOnly on
     iOS-27); re-scores the fused top-30 (query, passage) pairs BETWEEN RRF
     fusion and the tradition multiplier; app measures latency and falls through
     to plain RRF on a budget miss.
   - App action: sync the reranker pack; wire per **`docs/reranker-pack.md`**
     (I/O incl. meaningful `token_type_ids`, 192-token pairs, raw-logit score
     semantics, parity + tokenizer fixtures). Gate: `reports/reranker_eval_v1.md`
     (SHIP), export provenance: `reports/coreml_reranker_export.txt`.

6. **Expansion table state — synonyms folded in, query expansion OFF for
   retrieval.**
   - The advisor approved `data/eval/theological_synonyms_v1.json`, so the
     search packs' `expansion` table now includes the **100 archaic↔modern
     doctrinal-vocabulary synonym pairs** (200 rows: sepulchre↔tomb,
     candlestick↔lampstand, oblation↔offering, devils↔demons, …) alongside the
     archaic + suffix rows.
   - **DECISION (measured, `reports/retrieval_eval_v1.md` §4b): keep query
     expansion OFF for retrieval scoring.** Even synonym-inclusive, every
     headline + per-category delta was null-to-negative on the corpus-native gold
     set (best overall recall@20/MRR gain +0.000). The expansion table SHIPS in
     the search pack for the **tap-to-gloss / lookup UX**; do NOT enable the
     query-time BM25 down-weighted expansion for retrieval ranking. Revisit only
     with a KJV-phrased user-query gold set.

Unchanged since the app's last sync: `bundled_bibles`, `ondemand_bibles /
commentaries / lexicons`, the BGEQuery query-encoder pack (`packs.models`), the
int8 vector encoding, and the integer-chunk-id assignment
(`ascending-string-chunk-id-1based`, per-corpus-version — never persist across
corpus updates).

## Final pack manifest (sizes + sha256)

Verify each synced file against these. `.sqlite` = file sha256; `.mlpackage` =
deterministic directory-TREE sha256.

### Bundled (ships in the app binary — 43.3 MB)

| file | bytes | sha256 |
|---|---:|---|
| `bundled_bibles.sqlite` | 6,524,928 | `5017282913e635c1702fe531b6c4f6c5d47bf209e75df4ec71fabcf97c6a31ff` |
| `bundled_confessions.sqlite` | 102,400 | `d27f8a9fce50184423bb31702fc0363e5a544cd682595f4b13eded0507115cdc` |
| `bundled_crossrefs.sqlite` | 7,389,184 | `116027ccce15a1156e5d4cccb122f7d37aec012d4fffa06840f2f626fa370754` |
| `bundled_search.sqlite` | 31,416,320 | `7a2d650274c2454473e5593bd8dbea82647ce9e6b756217041fba624a8895223` |

### On-demand (first-launch download, all `tier="default"` — 582.3 MB)

| file | bytes | download_group | sha256 |
|---|---:|---|---|
| `ondemand_bibles.sqlite` | 20,398,080 | content | `7014a42d262d6eed25ec19a9d204e9b48e01bdb338af6b024c1b056a95e8d455` |
| `ondemand_confessions.sqlite` | 1,368,064 | content | `f43160d49632db30701a5d5c44834a611c87844ff547aac243559bcce0ae9e69` |
| `ondemand_commentaries.sqlite` | 132,046,848 | content | `03c0f27486919ecb9f86825f3a9eb902d5773a7f2aa9d01867bbf8e9b0a3adf7` |
| `ondemand_lexicons.sqlite` | 68,857,856 | content | `213203a8b527643d87dcb4d179f6f34fb6736da350c2afd608d01c7b25de5a90` |
| `ondemand_crossrefs.sqlite` | 25,362,432 | content | `ba499adefce419bf5b2ec07659588fdc9111cebc4d249d03a834263068fc671f` |
| `ondemand_search.sqlite` | 276,221,952 | retrieval-index | `f7166daa35e2f084c6ae1e1123f0e76109299913354cb33727199d4b86689486` |
| `ondemand_vectors.sqlite` | 86,298,624 | retrieval-index | `bddd6e3dc865d6fd2b390338644d87fe7b37bec116856ad5ccc8e9d98ca042a7` |

### Models — query encoder (`packs.models`, app-binary)

| file | bytes | sha256 |
|---|---:|---|
| `BGEQuery.mlpackage` | 66,612,673 | `67140125bf914df858b796bcc666a02d2edd72a7fa868b79653c92e0b74eab7d` |
| `vocab.txt` | 231,508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |

### Reranker (`packs.reranker`, app-binary — NEW)

| file | bytes | sha256 |
|---|---:|---|
| `Reranker.mlpackage` | 45,535,881 | `d7fd5fcdf83cb4c7a090b6bf853f50cbafb38764f9aa94eb0a464405d130f894` |
| `reranker_vocab.txt` | 231,508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |

Manifest totals: bundled 45,432,832 B · on-demand 610,553,856 B · all
655,986,688 B (pack files only; the `.mlpackage` app-binary assets are tracked in
`packs.models` / `packs.reranker`, not in these totals).

## Re-sync order (recommended)

1. Pull `corpus_manifest.json`; read `corpus_version` (`corpus-v2.1.0-candidate`).
2. Sync bundled packs (binary): confirm the 4 checksums above.
3. First launch: fetch every `packs.on_demand.files[]` with `tier == "default"`;
   treat `retrieval-index` (`ondemand_search` + `ondemand_vectors`) as one unit.
4. Sync `packs.models` (BGEQuery) + `packs.reranker` (Reranker) app-binary assets;
   verify tree sha256s; wire the reranker per `docs/reranker-pack.md`.
5. Migrate the readers per `docs/pack-diet.md` (parent-text reconstruction; v2
   posting-blob decode; integer chunk ids) and `docs/crossrefs-pack.md`.
6. Keep query expansion OFF for retrieval; wire the `expansion` table only into
   the tap-to-gloss UX.

## Flags carried into the freeze

- `ondemand_search.sqlite` is 276 MB (> the 250 MiB soft target). Bulk is
  commentary/lexicon child display text + BM25 postings. Next lever (architect
  decision): drop child display text and resolve from the per-resource packs.
- `Reranker.mlpackage` is 43.4 MB fp16 (> the ~20-25 MB back-of-envelope; under
  the shipped BGEQuery ~66 MB). Architect confirms the binary-size budget.
- `ship_ready = false`: the architect's 23-point spot-check gates the
  `corpus-v2.1.0` tag. This doc is the app re-sync checklist, not a ship approval.
