---
type: reference
venture: amphora
title: Lampstand Corpus Knowledge
source: lampstand-corpus repo export (24a91f4, 2026-09-01)
updated: 2026-09-01
tags: [reference, lampstand-corpus, ingest]
---

# Lampstand Corpus Knowledge

> Knowledge export of the `lampstand-corpus` repository — the offline text-and-ML pipeline that builds the read-only Bible-study assets consumed by the [[lampstand-ios]] app. Synthesized from README, CLAUDE.md, docs, the corpus manifest, source layout, and validation reports. No source code, secrets, or credentials are reproduced here.

## What it is

`lampstand-corpus` is the **corpus pipeline** for **LampStand**, a native iOS Bible-study app by **Amphora LLC** (branded "LampStand · An Amphora Company"). It is a Python project that ingests versioned, public-domain religious texts, normalizes them, builds per-resource SQLite databases plus dense (embedding) and sparse (BM25) retrieval indexes, validates everything, and packages the result into the read-only asset files that [[lampstand-ios]] bundles or downloads.

- **Purpose.** Turn a set of canonical, license-clean source snapshots (Bible translations, Reformed confessions/catechisms, public-domain commentaries, Strong's/BDB/TBESG lexicons, Treasury-of-Scripture-Knowledge cross-references) into a reproducible, provenance-tracked corpus with on-device search.
- **What it contains.** Four English Bible translations (BSB, KJV, ASV, WEB), seven Reformed confessional documents, four commentary sets, Hebrew/Greek lexicons + Strong's-tagged original-language text, ~344k cross-reference edges, plus offline-computed embeddings, a BM25 keyword index, and Core ML on-device models (a query encoder and a reranker).
- **Who/what consumes it.** The private [[lampstand-ios]] app repo. The corpus repo emits checksummed **packs** (`.sqlite`) and **models** (`.mlpackage` + tokenizer vocab); the app bundles the small "bundled" set in its binary and pulls the larger "on-demand" set at runtime from a Cloudflare R2 bucket (`lampstand-packs/<corpus_version>/`). The two repos are separate lanes; `corpus_manifest.json` is the contract between them.
- **Operating notes.** As of 2026-09-01 the repo carries a `CLAUDE.md` — operating notes for coding agents (commands, current status, sharp edges: human-gated ship, determinism, never commit artifacts) — alongside a repo-root copy of this knowledge export.
- **Why the repo is public.** Deliberately open-source. LampStand's value proposition is **privacy and theological care**; open-sourcing the pipeline lets anyone verify exactly how texts are sourced, normalized, and processed. The README frames the repo as a **credibility asset**, not a product giveaway.
- **Critical scope boundary.** The repo contains **pipeline + validation _code_ only**. Licensed/restricted texts and all compiled artifacts (`.sqlite` DBs, embedding indexes, `.mlpackage` models) are **never committed** — they are gitignored and synced out-of-band. Only `corpus_manifest.json` (the manifest of record) and the plaintext validation reports are committed.

## Architecture

- **Language / runtime.** Python ≥ 3.11. Packaged with **hatchling** (`pyproject.toml`); code lives in `src/lampstand_corpus/`. Lint via **ruff** (line length 100, rules E/F/I/UP/B), tests via **pytest** (`tests/`).
- **Core dependencies (light install).** `requests` (snapshot downloads), `beautifulsoup4` + `lxml` (CCEL HTML / USX / XML parsing), `pydantic` v2 (typed normalized records + provenance), `rank-bm25` (reference BM25; the pipeline writes its own index), `numpy` (float32 vector blobs + cosine smoke tests). Deliberately lean so the snapshot/build/validate phases stay fast.
- **Heavy optional extras (kept out of the default install).**
  - `[embeddings]` → `sentence-transformers` + `torch` (CPU/MPS; no CUDA) for the P6 embedding phase (the BGE-small encoder).
  - `[coreml]` → `coremltools` for exporting the on-device query-embedding Core ML package (`BGEQuery.mlpackage`).
  - `[rerank]` → `sentence-transformers`/`torch` again, for the cross-encoder rerank **measurement** gate only (never shipped in the core package).
  - `[dev]` → `pytest`, `ruff`.
- **Pipeline stages (the P0–P8 phase model).** Documented in `docs/normalized-schema.md` and `__init__.py`:
  1. **P0 scaffold** — project + schema skeleton.
  2. **P1 Bibles** — ingest USFM Bible snapshots → `bibles.sqlite`.
  3. **P2 confessions** — parse confessions/catechisms → `confessions.sqlite`.
  4. **P3 commentaries** — parse CCEL/OCR commentaries → `commentaries.sqlite` (pure structural-header chunks dropped at write time since v2.1.2; `build-commentaries` prints the dropped count).
  5. **P4 lexicons** — Strong's/BDB/TBESG + tagged original text → `lexicons.sqlite`.
  6. **P5 cross-refs** — TSK edges → `crossrefs.sqlite`.
  7. **P6 embeddings + BM25** — chunk the built DBs, encode with the pinned model, build the sparse index → `embeddings.sqlite`.
  8. **P7 packaging + snapshot** — split/quantize into app-side packs (`package` command).
  9. **P8 validation + spot-check handoff** — per-phase validators + consolidated report + human spot-check worksheet.
- **The unifying idea: normalize, then everything is uniform.** Whatever the source format (USFM, USX, CCEL ThML HTML, OpenScriptures JSON/XML, DjVu OCR), each source converges on **one normalized representation** before anything is written to SQLite or embedded. Every normalized chunk carries a full **provenance record** (source id, upstream version, license string, retrieval date, canonical URL, SHA-256 of the source snapshot). The **verse reference** (`book`, `chapter`, `verseStart`, `verseEnd`) is the spine — commentary, cross-refs, and word studies anchor to canonical verse refs, not translation-specific text, so they hold across translations.
- **How it's run.** A single CLI: `python -m lampstand_corpus.cli <command>`. There is a `snapshot → build → validate` triad per resource family (e.g. `snapshot-confessions` / `build-confessions` / `validate-confessions`), plus embedding, eval, export, and packaging commands. Selected commands (from `cli.py`):
  - **Bibles:** `snapshot`, `build`, `validate`.
  - **Confessions:** `snapshot-confessions`, `build-confessions`, `validate-confessions`.
  - **Commentaries:** `snapshot-commentaries`, `snapshot-spurgeon`, `build-commentaries`, `validate-commentaries`.
  - **Lexicons:** `snapshot-lexicons`, `snapshot-tagged`, `snapshot-stepbible`, `build-lexicons`, `validate-lexicons`.
  - **Cross-refs:** `snapshot-crossrefs`, `build-crossrefs`, `validate-crossrefs`.
  - **Embeddings:** `snapshot-model` (download BGE-small at pinned revision → `models/`), `build-embeddings` (incremental by default; `build-embeddings full` forces a from-scratch encode), `validate-embeddings`.
  - **Retrieval eval:** `build-eval` (derive gold set), `validate-retrieval` (measure BM25 / dense / hybrid arms), `sweep-retrieval` (fusion-knob sweep, recommends only), `rerank-eval` (cross-encoder rerank gate).
  - **Model export:** `coreml-export` (query encoder), `coreml-export-reranker`.
  - **Packaging:** `package` (build v2 packs; `package fp32` keeps float32 vectors instead of int8).
- **Pack transport (shell scripts, not CLI commands).** `scripts/upload-packs-r2.sh [packs-dir]` pushes the on-demand packs to the Cloudflare R2 bucket `lampstand-packs` under a `<corpus_version>/` prefix (so corpora coexist and clients never see a mid-upgrade mix). It **refuses to run unless the manifest says `ship_ready: true`**, and verifies each local file's size + sha256 against the manifest before uploading. `scripts/verify-packs-remote.sh <public-base-url>` then re-downloads every on-demand pack (~610 MB) from the public URL the app will use and checks size + sha256 against the manifest — the end-to-end proof the transport serves exactly the bytes the app's baked checksums will accept; run after upload, before TestFlight. Needs `jq` + `npx wrangler` (one-time browser OAuth login on the build machine; no credentials in the repo).
- **Determinism is a first-class design constraint.** Same source snapshots (identified by SHA-256) must produce **bit-for-bit identical** output: no timestamps written into output, no unfixed random seeds. Embeddings are encoded on **CPU, single-threaded**, with `torch.manual_seed(0)`, deterministic algorithms enabled, and `TOKENIZERS_PARALLELISM=false`. The build re-encodes a fixed sample and requires a byte-match; a mismatch is recorded as cosine-within-tolerance (the architect-approved **"Option A"**, since CPU float reductions jitter at ~1e-7) and **flagged**, never silently accepted. Rebuild determinism was verified bit-identical across three `package` runs.
- **Incremental re-encode.** `build-embeddings` reuses vectors whose chunk content is unchanged (a chunk id is the content-addressed `sha256(resource_type · source · anchor · text_checksum)`), re-encoding only changed/new chunks. This turns a small-slice corpus update (e.g. swapping the Strong's dictionaries re-texts ~14k of ~175k chunks) from a ~4-hour full encode into minutes. The BM25 index is always rebuilt over the full new chunk set (order-stable, deterministic).

## Data model / corpus structure

### Normalized intermediate format
Defined in `docs/normalized-schema.md`. Provenance fields on every chunk: `source`, `version`, `license`, `retrieved`, `url`, `checksum`. Resource types and their chunking granularity:

| Resource | Chunk granularity | Build DB |
|---|---|---|
| Scripture | Pericope (~5–15 verses) for embedding; verse-addressable for display | `bibles.sqlite` |
| Commentary | Paragraph, mapped to a verse range | `commentaries.sqlite` |
| Confessions / catechisms | Section / Q&A | `confessions.sqlite` |
| Lexicons | Entry (by Strong's number / lemma) | `lexicons.sqlite` |
| Cross-references | Verse → verse(s) edges (a graph, **excluded** from embedding) | `crossrefs.sqlite` |

Scripture pericopes use a deterministic verse-window fallback (`bibles.sqlite` has no paragraph markers): walk verses in canonical order, group into windows of `PERICOPE_TARGET=10` verses, never crossing a chapter boundary, folding a trailing remainder `< PERICOPE_MIN_TAIL=4` into the previous window. Chunks with no embeddable text (e.g. BDB lemma-only stubs, all-omitted critical-text windows) are **skipped and flagged**, never silently dropped from the source DB.

### The retrieval index (`embeddings.sqlite`, build artifact)
Built from the already-built per-resource DBs into one gitignored `embeddings.sqlite` (~2.6 GB on disk). Both indexes are plain, app-readable SQLite — no exotic formats.
- **Dense.** `BAAI/bge-small-en-v1.5`, dim **384**, L2-normalized float32 little-endian (so cosine == dot product). Query convention: the app prepends BGE's instruction string `"Represent this sentence for searching relevant passages: "`; passage vectors are stored bare.
- **Sparse (BM25).** Stored in `bm25_term`/`bm25_posting`/`bm25_doc`/`bm25_stats` tables. Tokenizer is deterministic and documented: Unicode NFKC → casefold → `[a-z0-9]+('[a-z0-9]+)*` (internal apostrophes kept), **no stemming, no stopwords, no language rules**. `bm25_stats` carries `N`, `avgdl`, `k1`, `b` so the app scores identically.

### `corpus_manifest.json` (the committed manifest of record)
The single source of truth the app reads to know what to fetch and verify. It tracks:
- **`corpus_version`** — currently **`corpus-v2.1.2`** with **`ship_ready: true`**, the first ship-ready corpus: spot-check passed 2026-07-06, architect-affirmed 2026-08-11. The ship marker is **manifest-based, not a git tag** — the only tags in the repo are `corpus-v1.0.0` and `pre-author-rewrite`. The pipeline itself still never self-marks ship-ready: it only ever emits `-candidate` versions with `ship_ready: false`, and the flip to `true` is the architect's own commit after the 23-point spot-check.
- **`embedding_dim`** (384) and a `format_migration` note describing the v1→v2 pack diet.
- **`packs`** grouped by delivery channel:
  - **`bundled`** (ships in the app binary, public-domain/CC0 only): `bundled_bibles.sqlite` (BSB, 31,102 verses, 3,086 headings), `bundled_confessions.sqlite` (WSC, 107 sections), `bundled_crossrefs.sqlite` (TSK edge network + per-pericope top-8 expansion), `bundled_search.sqlite` (BSB/WSC-scoped dense+BM25 index). ~45.7 MB total.
  - **`on_demand`** (downloaded at runtime from R2, free): `ondemand_bibles.sqlite` (KJV/ASV/WEB), `ondemand_commentaries.sqlite`, `ondemand_confessions.sqlite` (the other 6 documents), `ondemand_crossrefs.sqlite`, `ondemand_lexicons.sqlite`, `ondemand_search.sqlite` (full BM25 + metadata + display text; ~275 MB), `ondemand_vectors.sqlite` (int8 dense vectors; ~85 MB). ~610 MB total.
  - **`models`** — `BGEQuery.mlpackage` (fp16 on-device query encoder) + `vocab.txt`. ~66 MB.
  - **`reranker`** — `Reranker.mlpackage` (fp16 cross-encoder) + `reranker_vocab.txt`. ~45.5 MB.
- Every pack file records **`bytes`** and a **`sha256`** for app-side verification; the `.mlpackage` sha256 is a deterministic directory-tree hash.
- **`acknowledgements[]`** — one entry per source with `id`, `name`, `resource_type`, `version`, `license`, `attribution`, `source_url`, `retrieved`, and `source_checksum`. This is the licensing/provenance ledger (see Licensing section).
- **Download tiering.** Every on-demand file has a `tier` (`default` / `optional`) and a `download_group` (`content` / `retrieval-index`). The default first-launch set = every `tier == "default"` file. `ondemand_search` + `ondemand_vectors` form the `retrieval-index` group; `ondemand_vectors` is **default, not opt-in**, but if absent, retrieval **degrades gracefully to BM25-only** (the search pack is self-sufficient and is never gated on the vectors pack).

### v2 pack schema (the "pack diet", `docs/pack-diet.md`)
The v1 monolithic `ondemand_embeddings.sqlite` (~1.99 GB) was split into `ondemand_search.sqlite` + `ondemand_vectors.sqlite` for size.
- **Integer chunk ids.** Assigned 1-based by ascending string chunk id over the whole corpus (`id_assignment = "ascending-string-chunk-id-1based"`). Deterministic and content-addressed, but **per-corpus-version only** — adding/removing any chunk renumbers everything. Cross-version identity lives in `chunk.string_id`. The app must **never persist integer ids across corpus updates**.
- **Search pack (`search-pack-v2`).** `chunk` table (metadata + display `text` + `doc_len` + structural `header` + `parent_id`/`indexed` for dual granularity), `bm25_term` (one varint-delta **posting BLOB** per term, `uvarint-gap-tf-v1` LEB128 encoding), `bm25_stats`, plus `expansion` (symmetric query-expansion pairs) and `gloss` (one-way archaic→modern, display-only).
- **Dual granularity (Rank 8).** Scripture rows split into **children** (single verses, `indexed=1`, the only rankable Scripture rows) and **pericope parents** (`indexed=0`, `parent_id NULL`, in no posting blob/vector). Parents store `text = NULL`; the app **reconstructs** a parent's text by concatenating its children in `(book, chapter, verse_start)` order. Dense vectors exist only for BSB children + non-scripture (KJV/ASV/WEB children are BM25-only).
- **Vectors pack (`vectors-pack-v2`).** `embedding(chunk_id, vector, scale)`; vectors are **int8 symmetric-per-vector** quantized (`scale = max(|v|)/127`), scored directly against the float32 query vector (bit-identical to scoring the dequantized float32). `package fp32` is the fidelity escape hatch keeping float32.
- **Cross-refs pack (`crossrefs-pack-v1`, `docs/crossrefs-pack.md`).** `crossref` (per-source verse → vote-ranked targets, verse-key arithmetic `book_ord*1e6 + chapter*1e3 + verse`, signed zig-zag votes), `chunk_crossref` (per-pericope top-8 TSK-adjacent neighbors for the `hybridContext` graph boost), and `prooftext` (reverse index: "this verse is cited by WCF 11.1 / HC 60 / Dort h1.a1").

## Sources & content

Each source family lives under `sources/<family>/` as versioned snapshots (the raw bytes are gitignored per repo policy; only the provenance manifest is committed). `sources/manifest.json` records the four Bibles with URL + retrieval date + SHA-256. Content by family:

- **Bibles (`sources/{bsb,kjv,asv,web}`).** USFM format.
  - **Berean Standard Bible (BSB)** — bereanbible.com, snapshot 2026-06-10. The primary/bundled translation; carries 3,086 section headings and red-letter markup.
  - **KJV** — eBible.org `eng-kjv2006`. **ASV** — eBible.org `eng-asv` (1901). **WEB (World English Bible)** — eBible.org `eng-web` 2020 stable.
  - Canon: 66-book Protestant, KJV/traditional versification. A 16-reference "omitted-verse union" of critical-text variants is tracked and resolves in every translation.
- **Confessions / catechisms (`sources/confessions/`).** Seven Reformed documents, 895 sections total:
  - **WCF** (Westminster Confession) — original 1646/47 text from `andrewhwaller/westminster-json`, cross-checked against Wikisource Burges-1646, with the six 1788 American-revision loci marked.
  - **WLC / WSC** (Westminster Larger / Shorter Catechisms), **Heidelberg Catechism**, **Canons of Dort** — CCEL ThML.
  - **1689 LBCF** (Second London Baptist Confession) — `ParticularBaptists/lbcf-1689`, 32 chapters.
  - **Belgic Confession** — Wikisource 1840 RPDC translation, 37 articles.
- **Commentaries (`sources/commentaries/`).** Four public-domain sets, chunked per paragraph. Since v2.1.2, **pure structural-header chunks are dropped at build time** — passage labels and chapter marks with no exposition (Calvin's leading "Romans 15:25-29"-style headers, JFB's "CHAPTER 23" / "PSALM 134" marks) that wasted retrieval slots and rendered as empty detail sheets in the app. The filter is a deliberately conservative whole-string predicate: 3,238 chunks dropped with zero stray prose words, while 7,600+ terse-but-real chunks (JFB glosses, Calvin footnotes) that a naive length filter would have deleted are all kept. The drop propagates into the embeddings/BM25 index. The sets:
  - **Matthew Henry** (complete, 6 volumes), **Jamieson-Fausset-Brown (JFB)** (complete) — CCEL ThML.
  - **John Calvin** — CCEL, v1 scope = **Genesis + Psalms + New Testament only**; remaining OT volumes deferred.
  - **Charles Spurgeon — Treasury of David** (Psalms) — Internet Archive DjVu **OCR** (shipped in v2.1.2 under the architect's sign-off; OCR quality remains the flagged fidelity risk), because the CCEL edition is image-only. All seven volumes: six from the Google-digitized `*spurgoog` scans, with Psalms 104–118 gap-filled from an alternate PD scan.
- **Lexicons + tagged original text (`sources/lexicons/`).**
  - **Strong's Greek** — `morphgnt/strongs-dictionary-xml` (full G1–G5624 incl. "Not Used" placeholders).
  - **Strong's Hebrew** + **Brown-Driver-Briggs (BDB)** — `openscriptures/HebrewLexicon` (BDB keyed to Strong's via `LexicalIndex.xml`).
  - **TBESG** (Tyndale Brief lexicon of Extended Strong's for Greek) — `STEPBible/STEPBible-Data`; the architect-approved substitute for Thayer's, keyed by extended/disambiguated Strong's.
  - **OSHB** (Open Scriptures Hebrew Bible, WLC) — `openscriptures/morphhb`, Strong's + morphology per word.
  - **TAGNT** (Translators Amalgamated Greek NT) — `STEPBible/STEPBible-Data`, disambiguated Strong's + morphology for all 27 NT books.
  - **SBLGNT / MorphGNT** snapshotted **for provenance only** (no Strong's; under the SBLGNT EULA). **OpenGNT rejected** (CC-BY-SA copyleft).
- **Cross-references (`sources/crossrefs/`).** **Treasury of Scripture Knowledge (TSK)** via the **OpenBible.info** `cross_references.txt` dataset (~344,799 edges, ~29,364 source verses). Each edge carries a signed community relevance weight (−86 .. +1278; sign preserved). Refs are normalized to the KJV spine; non-resolving refs are flagged, never dropped.

**Guiding principle:** a less-canonical source is never substituted silently — it is always flagged for human review.

## Licensing & provenance

Licensing is treated as a first-class concern for a text corpus. The pipeline's `LICENSE` (MIT) explicitly covers **pipeline code only**, and states it does **not** cover the source texts (each carries its own PD status/license) nor the compiled artifacts (not distributed here). Key facts and sensitivities:

- **Repo code license.** MIT, © 2026 Amphora LLC.
- **Sourcing policy (`sources/README.md`).** Public-domain / open-licensed texts **only**; no restricted or modern-licensed content is ever placed in `sources/`. Every snapshot is recorded with source URL, fetch date, and SHA-256 (the checksum is the unit of reproducibility). Builds read from local snapshots, never live URLs.
- **Per-source license ledger (`corpus_manifest.json` → `acknowledgements[]`).** Every source records its license and required attribution:
  - **Public domain / CC0:** BSB (CC0), KJV/ASV/WEB (PD), all confessions (PD text), all four commentaries (PD — Spurgeon d.1892, Henry, JFB, Calvin), Strong's Greek (CC0 via `morphgnt/strongs-dictionary-xml`).
  - **CC-BY 4.0 (attribution required, share-alike-free):** Strong's Hebrew, BDB, OSHB, TBESG, TAGNT (all OpenScriptures / STEPBible editions), and the TSK cross-references (OpenBible.info).
- **Deliberate copyleft avoidance.** The project explicitly **replaced** prior CC-BY-**SA** Strong's dictionaries (`openscriptures/strongs` `.js`) with CC0 / CC-BY editions so **no copyleft term lands in the corpus**, and **rejected OpenGNT** (CC-BY-SA). This is an intentional, documented legal posture.
- **Required attribution strings that must render in the app's acknowledgements:**
  - TSK cross-references: *"Cross-reference data courtesy of www.openbible.info (CC-BY)."*
  - TAGNT / STEPBible: *"STEP Bible, www.STEPBible.org."*
  - OSHB: *"Original work of the Open Scriptures Hebrew Bible available at https://github.com/openscriptures/morphhb."*
- **Model licenses.** Query encoder `BAAI/bge-small-en-v1.5` — **MIT**. Reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` — **Apache-2.0** (chosen over the higher-scoring MIT `BAAI/bge-reranker-base` to fit the on-device size budget). BERT tokenizer vocab — Apache-2.0. All shipped as static-weight fp16 Core ML exports.

**Legally sensitive / flagged items to watch:**
- **KJV UK Crown patent.** The manifest notes the KJV is public domain *except* the UK Crown patent printing restriction. Relevant if LampStand distributes in the UK.
- **No pure-PD machine-readable Strong's *Hebrew* exists** in a source of record, so the best attribution-only option (CC-BY) is used and flagged. CC-BY carries an ongoing attribution obligation the app must honor.
- **Spurgeon Treasury of David is OCR** — shipped in v2.1.2 under the architect's spot-check; the underlying text is PD but OCR quality remains a data-quality risk (98 OCR flags in validation).
- **SBLGNT is under a EULA** and is snapshotted for provenance only (not shipped as data) — a boundary that must not be crossed.
- The confessions' underlying texts are PD, but the manifest records the **upstream repo licenses** separately (e.g. the WCF source repo is MIT) — the distinction between *text license* and *repo license* is tracked deliberately.

## Models & outputs

Two categories of output: **compiled data packs** (SQLite, see Data model) and **ML artifacts** (Core ML + eval reports). The heavy build artifacts under `output/` and `models/` are gitignored; only reports and the manifest are committed.

- **Embedding index (P6).** From `output/embeddings.sqlite`: **313,966 chunks** embedded at v2.1.2 (commentary 142,343 · scripture 138,167 · lexicon 32,561 · confession 895 — down 3,238 from v2.1.1 after the structural-header drop), plus a BM25 index (174,383 terms, ~14.2M postings). 3/3 cosine smoke queries pass. Model: BGE-small-en-v1.5, dim 384, pinned HF revision `5c38ec7c405e`.
- **On-device query encoder — `BGEQuery.mlpackage`.** Core ML fp16 export of BGE-small (via `coreml_export.py` / `coreml-export`). `RangeDim(1,512)` sequence length. Ships in the app binary with its tokenizer `vocab.txt`; the app loads it strictly from disk (no runtime download). Export provenance in `reports/coreml_export_m4.txt`; tokenizer-parity fixtures are regenerated from the real HuggingFace fast tokenizer to guarantee app/pipeline token-id agreement.
- **On-device reranker — `Reranker.mlpackage`.** Core ML fp16 export of `cross-encoder/ms-marco-MiniLM-L-6-v2`. Re-scores the fused top-30 `(query, passage)` candidate pairs (192-token pairs) **between RRF fusion and the tradition multiplier**; raw-classifier-logit score semantics (higher = more relevant; only order matters). The app measures on-device latency and **falls through to plain RRF on a budget miss**. Contract: `docs/reranker-pack.md`; export report: `reports/coreml_reranker_export.txt`.
- **Retrieval evaluation (F5 measurement foundation).** A zero-annotation **gold set** is derived from the corpus itself (`output/eval_gold_v1.json`, seed 613): prooftext (150 queries), crossref (150), commentary-anchor (152), plus a DRAFT hard-negative suite (60). Three arms — BM25-only, dense-only, hybrid RRF (fusing exactly as the app's `HybridRetriever` does with `k=60 perType=20 denseDepth=20`) — are measured for recall@k / MRR / nDCG@10 per category (`reports/retrieval_eval_v1.md`). Labels are conservative (corpus-native, lexically biased), so absolute numbers are floors, not user-experienced quality; **arm-vs-arm deltas on identical labels** are the meaningful signal. `sweep-retrieval` recommends fusion constants but never changes the app.
- **Reranker gate (`reports/reranker_eval_v1.md`).** Verdict: **SHIP.** The reranker cleared the gate (overall MRR +0.083; commentary-anchor MRR +0.206 / recall@10 +0.105; hard-negative pairwise 1.000). `ms-marco-MiniLM-L-6-v2` was chosen over the higher-scoring `bge-reranker-base` for the on-device size envelope.
- **Validation reports (`reports/`).** Every build phase emits a plaintext/Markdown report. The consolidated `corpus_validation_v1.md` rolls up all six per-phase validators: totals ~**13 errors / ~5,163 flags**, where errors and flags are both for **human adjudication**, never silent fixes (the rollup predates the v2.1.2 re-embed; the regenerated P6 embeddings report alone now stands at 1 error / 4,619 flags). ~98% of the large embeddings flag count is expected BDB lemma-only stubs + Strong's "Not Used" placeholders skipped from embedding. A pre-filled human spot-check worksheet lives at `reports/spotcheck_worksheet_v1.md`.

## Integrations & expected env

- **External services / sources.** The pipeline fetches source snapshots over HTTPS (via `requests`) from public, **no-auth** endpoints: `bereanbible.com`, `ebible.org`, `raw.githubusercontent.com` (OpenScriptures, STEPBible, morphgnt, ParticularBaptists, andrewhwaller repos), `ccel.org`, `en.wikisource.org` API, and `a.openbible.info`. Models are pulled from the **HuggingFace Hub** (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) at pinned revisions via `huggingface_hub.snapshot_download`. One **outbound** service: **Cloudflare R2** (bucket `lampstand-packs`), the pack-transport target for the upload/verify scripts.
- **No secrets or credentials.** There are **no API keys, tokens, or `.env` values** anywhere in the pipeline. Sources are public; model downloads are unauthenticated. Nothing reads a credential. The R2 upload authenticates via a one-time `npx wrangler login` browser OAuth on the build machine — nothing credential-shaped lands in the repo.
- **Environment variable NAMES the pipeline sets (internally, for determinism / caching — not read from a secret store):**
  - `HF_HUB_CACHE` — set (via `os.environ.setdefault`) to point HuggingFace downloads at the local gitignored `models/` cache dir.
  - `TOKENIZERS_PARALLELISM` — set to `false` for deterministic, single-threaded tokenization.
  - `PYTHONHASHSEED` — set to `0` for reproducibility.
  - (There is no external `*_TOKEN` / `*_API_KEY` / `*_SECRET` expected or consumed.)
- **Downstream integration.** The [[lampstand-ios]] app (private repo) is the sole consumer. Integration is **manifest-driven**: the app reads `corpus_manifest.json` and verifies each asset's `sha256`. The bundled packs and models still travel inside the app binary, but the **on-demand packs are now served over the network**: the app's `RemoteCorpusPackInstaller` downloads them at runtime from the R2 bucket at `<public-base-url>/<corpus_version>/<file>` (custom domain `packs.thelampstand.app`, or the bucket's r2.dev URL), checking every download against the baked manifest checksums. The versioned prefix lets corpora coexist so clients never see a mid-upgrade mix, and `scripts/verify-packs-remote.sh` is the pre-TestFlight proof the transport serves exactly the bytes the app will accept. The app's reader contracts are pinned in `docs/pack-diet.md`, `docs/crossrefs-pack.md`, `docs/reranker-pack.md`, and the re-sync checklist `docs/pack-handoff-v2.1.md`.

## Roadmap / TODOs

From code comments, docs, and reports:
- **Commentary expansion.** Calvin OT volumes (Exodus–Malachi minus Psalms) are **deferred** past v1 (surfaced in the report, never fetched). **John Gill** commentary is **deferred to v1.1** (no source defined yet).
- **Spurgeon gap.** Treasury of David is OCR and a candidate; historically the volume covering Psalms 104–118 was absent and flagged (later gap-filled from an alternate PD scan). OCR quality remains a manual-review item.
- **Proof-text coverage gap.** WCF/LBCF/Heidelberg/Dort carry proof apparatus; **WSC/WLC/Belgic source editions have none** — an architect decision on a supplemental proof-text source is **pending**.
- **Query expansion decision.** The `expansion`/synonym table SHIPS for the tap-to-gloss / lookup UX, but **query expansion for retrieval scoring stays OFF** (measured null-to-negative on the corpus-native gold set). Revisit only with a KJV-phrased user-query gold set.
- **DRAFT advisor gates.** `data/eval/hard_negatives_v1.json` and `data/eval/theological_synonyms_v1.json` are marked **DRAFT — pending theological-advisor review**; the 100 synonym pairs are folded into the gloss/expansion tables only after advisor approval.
- **Reranker → export.** Resolved — the SHIP verdict was carried through: the Core ML export and app wiring landed, and both reranker artifacts ship in the v2.1.2 manifest.
- **Size levers (carried through the v2.1.2 ship).** `ondemand_search.sqlite` (~275 MB) still exceeds the 250 MiB soft target — flagged as pre-existing and informational at ship; the next lever (architect decision) is to drop child display text and resolve from the per-resource packs. `Reranker.mlpackage` (~45.5 MB fp16) exceeds the ~20–25 MB back-of-envelope; the binary-size budget was architect-confirmed.
- **Ship gate.** Fired for the first time: `corpus-v2.1.2` carries `ship_ready: true`, architect-affirmed 2026-08-11 after the **23-point spot-check** (passed 2026-07-06). The gate itself is unchanged — the pipeline structurally never self-marks ship-ready and only emits `-candidate` manifests; the flip is a human commit, and there is no git tag for the ship (the manifest is the marker).
- **Sources-commit decision.** Whether raw snapshot bytes are committed (vs manifest-only + fetch-on-build, possibly git-lfs for larger texts) was flagged for the architect at P1.

## Open questions / risks

- **Data quality — OCR.** Spurgeon's Treasury of David is DjVu OCR (98 OCR flags); it shipped in v2.1.2 under the architect's sign-off, and OCR artifacts remain the biggest text-fidelity risk in the corpus.
- **Licensing — CC-BY attribution burden.** Multiple core sources (Strong's Hebrew, BDB, OSHB, TBESG, TAGNT, TSK) are CC-BY, imposing an ongoing, per-source attribution obligation the app **must** render. Any attribution regression is a compliance risk. The KJV UK-Crown-patent caveat is a jurisdictional edge case.
- **Reproducibility — float jitter.** Embedding determinism is not always bit-for-bit; CPU float reductions jitter at ~1e-7, accepted under the architect-approved "Option A" (cosine-tolerance) rather than enforced byte-identity. Reproducibility also depends on the pinned model revision and the local snapshot set — a moved/renamed upstream would break a fresh from-source rebuild.
- **Retrieval quality signal is a floor, not the truth.** The corpus-native gold set is lexically biased and conservatively labeled; a doctrinally-correct sibling hit scores as a miss. Real user-perceived quality lives in the app's own 46-case paraphrased-query eval, not in this repo. Eval numbers here should not be read as absolute quality.
- **Per-corpus-version integer ids.** The stable integer chunk ids renumber on any chunk add/remove; if the app ever persists them across a corpus update, cross-references and vectors would silently mis-join. The contract forbids it, but it's a fragile coupling.
- **Human-gated ship is a process dependency.** Nothing ships without the architect's manual 23-point spot-check + a theological advisor approving the DRAFT eval files. The gate has now exercised cleanly once (v2.1.2), but it remains a human bottleneck and a single point of judgment.
- **Two-repo drift.** The corpus and [[lampstand-ios]] are separate lanes coordinated only by `corpus_manifest.json` + the handoff docs. A reader-contract change (e.g. the parent-text-NULL migration) requires coordinated app-side work; skew between the emitted packs and the app's reader is a live risk (the handoff docs exist precisely to manage it). The R2 transport adds a network hop to that coupling — mitigated by per-version bucket prefixes and the `verify-packs-remote.sh` end-to-end check, but a stale or mis-uploaded bucket is a new failure surface the checksums must catch.
- **Missing content.** Calvin OT and Gill are absent; proof-texts are missing for WSC/WLC/Belgic. These are known gaps, transparently flagged, but they limit v1 coverage.

## Glossary

- **Corpus** — the full assembled body of processed texts + indexes + models this repo produces.
- **Manifest (`corpus_manifest.json`)** — the committed contract listing every pack/model with bytes, sha256, license, and tiering; the app reads it to fetch and verify assets.
- **Pack** — a compiled `.sqlite` asset the app bundles or downloads (bundled vs on-demand; roles: bibles / confessions / commentaries / lexicons / crossrefs / search / vectors).
- **Bundled vs on-demand** — bundled packs ship inside the app binary (public-domain/CC0 only); on-demand packs download at runtime from the versioned Cloudflare R2 bucket.
- **Chunk** — the atomic unit of retrieval: a pericope, commentary paragraph, confession section/Q&A, or lexicon entry, each with full provenance.
- **String chunk id** — content-addressed `sha256(resource_type · source · anchor · text_checksum)`; stable across corpus versions.
- **Integer chunk id** — a compact per-corpus-version id (1-based, ascending by string id); renumbers on any chunk change; never persist across versions.
- **Pericope** — a natural Scripture reading unit (~5–15 verses); the embedding granularity for scripture.
- **Parent / child (dual granularity)** — Scripture indexes at verse precision (children, rankable) but keeps pericope parents (`indexed=0`, `text=NULL`) for LLM context; the app reconstructs parent text from children.
- **Embedding / vector** — a 384-dim float32 (or int8-quantized) dense representation from BGE-small used for semantic search.
- **BM25** — the sparse lexical ranking function; the keyword arm of retrieval, with a documented deterministic tokenizer.
- **Hybrid retrieval / RRF** — Reciprocal Rank Fusion of the BM25 and dense arms, exactly as the app's `HybridRetriever` fuses (`k=60`, per-type limits, dense depth).
- **Reranker / cross-encoder** — an on-device model that re-scores the fused top-30 `(query, passage)` pairs for a semantic quality lift, with graceful fallback to plain RRF on a latency-budget miss.
- **Provenance** — the source/version/license/retrieved/url/checksum metadata carried on every chunk.
- **Strong's number** — a canonical index (G#### Greek / H#### Hebrew) keying original-language words to lexicon entries.
- **TSK (Treasury of Scripture Knowledge)** — the cross-reference dataset (via OpenBible.info) of signed, community-voted verse→verse edges.
- **Confession / catechism** — a Reformed doctrinal document (WCF, WLC, WSC, Heidelberg, Belgic, Dort, 1689 LBCF), chunked by section or Q&A.
- **Proof-text** — a Scripture citation attached to a confession/catechism section; indexed both forward (section→verses) and in reverse (`prooftext` table).
- **USFM / ThML / OSIS** — source markup formats: USFM/USX (Bibles), CCEL ThML (confessions/commentaries), OSIS-style verse refs on the normalized spine.
- **Candidate vs ship-ready** — the pipeline only ever emits a *candidate* (`ship_ready=false`); the flip to `true` is the architect's own commit after the 23-point human spot-check (first exercised for `corpus-v2.1.2`, affirmed 2026-08-11; manifest-based, no git tag).
- **Determinism / Option A** — the rule that identical source snapshots yield byte-identical output; "Option A" accepts ~1e-7 CPU float jitter under cosine tolerance (recorded, not ignored).
- **`.mlpackage`** — an Apple Core ML model bundle (the fp16 query encoder and reranker); its sha256 is a deterministic directory-tree hash.

Ingest note: drop into the Obsidian vault Clippings/ and ask Hermes to ingest.
