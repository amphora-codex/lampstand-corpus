# Normalized intermediate format

Every source — whatever its original format (USFM, USX, CCEL HTML, OpenScriptures JSON/XML) — converges on one normalized representation before anything is written to SQLite or embedded. This is what keeps provenance uniform and the output reproducible.

## Provenance (on every chunk, no exceptions)

| Field | Meaning |
|---|---|
| `source` | Canonical source id (e.g. `bsb`, `kjv`, `ccel:henry`, `openscriptures:strongs`) |
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

## Validation (every build emits a report)

Missing books/chapters/verses · commentary chunks not mapping to valid verse refs · Strong's numbers without lexicon entries · cross-refs pointing to non-existent verses · statistical anomalies (unusually short chapters, unusually long commentary blocks). Residuals that a parser can't adjudicate are listed for **human review**, never silently resolved.

> Pipeline phases (see the app repo's build-log / M1 plan): P0 scaffold · P1 Bibles · P2 confessions · P3 commentaries · P4 lexicons · P5 cross-refs · P6 embeddings+BM25 · P7 packaging+snapshot · P8 validation+spot-check handoff.
