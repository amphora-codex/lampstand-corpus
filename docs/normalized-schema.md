# Normalized intermediate format

Every source — whatever its original format (USFM, USX, CCEL HTML, OpenScriptures JSON/XML) — converges on one normalized representation before anything is written to SQLite or embedded. This is what keeps provenance uniform and the output reproducible.

## Provenance (on every chunk, no exceptions)

| Field | Meaning |
|---|---|
| `source` | Canonical source id (e.g. `bsb`, `kjv`, `ccel:henry`, `morphgnt:strongs-greek`, `openscriptures:strongs-hebrew`) |
| `version` | Upstream version/edition identifier |
| `license` | License / public-domain status string |
| `retrieved` | ISO date the snapshot was fetched |
| `url` | Canonical upstream URL |
| `checksum` | SHA-256 of the source snapshot the chunk derives from |

## Verse reference

The spine of everything. Canonical form: `book` (stable id), `chapter`, `verseStart`, `verseEnd`. Commentary, cross-refs, and word-study anchor to this — not to translation-specific text — so they hold across translations.

## Resource types & chunking (per spec §4.3)

| Resource | Chunk granularity | SQLite output |
|---|---|---|
| Scripture | Pericope (natural reading unit, ~5–15 verses) for embedding; verse-addressable for display | `bibles.sqlite` |
| Commentary | Paragraph, mapped to a verse range | `commentaries.sqlite` |
| Confessions/catechisms | Section / Q&A | `confessions.sqlite` |
| Lexicons | Entry (by Strong's number / lemma) | `lexicons.sqlite` |
| Cross-references | Verse → verse(s) edges | `crossrefs.sqlite` |

## Reproducibility rules

- No timestamps written into output. No unfixed random seeds.
- Same source snapshots (by checksum) ⇒ bit-for-bit identical SQLite + indices.
- Embeddings computed at pipeline time with a pinned model + fixed precision/seed.

## Embeddings + BM25 (P6)

The retrieval index is built from the *already-built* per-resource DBs into a
single `embeddings.sqlite` (gitignored). Both indices are deliberately simple,
app-readable SQLite — no exotic formats.

**Model (provenance).** `BAAI/bge-small-en-v1.5`, pinned to HuggingFace revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, dim **384**. Weights are cached under
a gitignored `models/` dir and **never committed**; the manifest/`meta` records
the model name, revision, and a combined SHA-256 over the snapshot's files (the
model is an input to reproducibility). Vectors are L2-normalized float32,
little-endian, so cosine == dot product downstream. Query convention: the app
prepends BGE's `"Represent this sentence for searching relevant passages: "`
instruction; passage (chunk) vectors are stored bare.

**Chunk granularity (spec §4.3).**

| Resource | Chunk | How |
|---|---|---|
| Scripture | Pericope (~5-15 verses) | `bibles.sqlite` carries no paragraph markers, so a **deterministic verse-window fallback**: walk verses in canonical order, group into windows of `PERICOPE_TARGET=10` verses, **never crossing a chapter boundary**, folding a trailing remainder `< PERICOPE_MIN_TAIL=4` into the previous window. All 4 translations embedded. |
| Commentary | Paragraph | The rows already in `commentaries.sqlite` (one chunk per `comment` row). |
| Confessions | Section / Q&A | The rows already in `confessions.sqlite`; the section title is prepended to the body so headings are searchable. |
| Lexicons | Entry | One chunk per Strong's/BDB/TBESG entry, built from its English-bearing fields (definition / derivation / KJV gloss) with the lemma + transliteration as a light head. |

Cross-references are a graph, not prose — **excluded** from embedding (spec §4.3).

Every chunk carries a resolvable anchor (resource type, source id, and a VerseRef
or key) plus a SHA-256 of its embeddable text for chunk-stability auditing. Chunks
that have no embeddable text (e.g. BDB lemma-only stubs with no English gloss, or a
Scripture window made entirely of omitted critical-text verses) are **skipped and
flagged in the report — never silently dropped from their source DB.**

**BM25 keyword index.** Stored as `bm25_term` / `bm25_posting` / `bm25_doc` /
`bm25_stats` tables. Tokenizer is **deterministic and documented**: Unicode NFKC
normalize → casefold → `[a-z0-9]+('[a-z0-9]+)*` (internal apostrophes kept), **no
stemming, no stopword removal, no language rules**. The same text always yields the
same token stream. `bm25_stats` carries `N`, `avgdl`, and the `k1`/`b` constants so
the app scores identically.

**Determinism.** The committed artifact is encoded on **CPU**, single-threaded,
with `torch.manual_seed(0)`, `use_deterministic_algorithms`, and
`TOKENIZERS_PARALLELISM=false`. The build re-encodes a fixed deterministic sample
and requires a bit-for-bit byte match; if that ever fails it records
cosine-within-tolerance + the input/model checksums and **flags the deviation for
the architect** rather than accepting it silently.

## Validation (every build emits a report)

Missing books/chapters/verses · commentary chunks not mapping to valid verse refs · Strong's numbers without lexicon entries · cross-refs pointing to non-existent verses · statistical anomalies (unusually short chapters, unusually long commentary blocks). Residuals that a parser can't adjudicate are listed for **human review**, never silently resolved.

> Pipeline phases (see the app repo's build-log / M1 plan): P0 scaffold · P1 Bibles · P2 confessions · P3 commentaries · P4 lexicons · P5 cross-refs · P6 embeddings+BM25 · P7 packaging+snapshot · P8 validation+spot-check handoff.
