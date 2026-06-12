# lampstand-corpus

The corpus pipeline for **LampStand** — a native iOS Bible study app by Amphora LLC. This repository is **public on purpose**: LampStand's promise is privacy and theological care, and open-sourcing the pipeline lets anyone verify exactly how the texts are sourced, normalized, and processed. It is a credibility asset, not a giveaway.

> This repo contains **pipeline and validation _code_ only.** Licensed or restricted texts and compiled artifacts (`.sqlite` databases, embedding indices) are never committed here.

## What it produces

From a set of versioned, public-domain source snapshots, the pipeline emits the read-only assets the app bundles or downloads:

- **Per-resource SQLite databases** — `bibles.sqlite`, `commentaries.sqlite`, `confessions.sqlite`, `lexicons.sqlite`, `crossrefs.sqlite`.
- **Embedding vector index** — computed offline (never at app runtime).
- **BM25 keyword index** — for hybrid (dense + sparse) retrieval.
- **Validation reports** — flagging missing books/chapters/verses, commentary that doesn't map to valid verse references, Strong's numbers without lexicon entries, cross-refs to non-existent verses, and statistical anomalies.

## Sources of record (canonical only)

- **Bibles:** Berean Standard Bible (bereanbible.com, CC0), KJV / ASV / WEB (eBible.org, public domain) — USFM.
- **Confessions/catechisms:** WCF — original 1646/47 text (`andrewhwaller/westminster-json`, MIT repo / PD text), prose cross-checked against Wikisource Burges-1646, with the six 1788 American-revision loci marked; 1689 LBCF (`ParticularBaptists/lbcf-1689`, CC0); Belgic Confession (Wikisource 1840 RPDC, PD); WLC / WSC / Heidelberg / Canons of Dort (CCEL ThML, PD). Underlying confessions are public domain; upstream repo licenses recorded in the source manifest.
- **Commentaries:** CCEL (each commentator verified against an authoritative edition) — Matthew Henry, Jamieson-Fausset-Brown, Calvin (Genesis + Psalms + NT for v1). **Spurgeon's *Treasury of David*** (Psalms) is ingested from the architect-approved Google-digitized Internet Archive `*spurgoog` DjVu OCR (PD; Spurgeon d.1892), since the CCEL edition is image-only. This is OCR (a *candidate*, not ship-ready); the volume covering Psalms 104-118 is absent from the scan set and is flagged for the architect to source.
- **Lexicons:** Strong's Greek (`morphgnt/strongs-dictionary-xml`, **CC0 / public domain**; full G1–G5624 span incl. explicit "Not Used" placeholders) and Strong's Hebrew (`openscriptures/HebrewLexicon` `HebrewStrong.xml`, **CC-BY 4.0** / PD text). These are **share-alike-free** editions — they replaced the prior `openscriptures/strongs` CC-BY-SA `.js` dictionaries so no copyleft term lands in the corpus; no CC0/pure-PD machine-readable Strong's *Hebrew* exists in a source of record, so the best attribution-only option (CC-BY) is used and flagged. Brown-Driver-Briggs comes from the same `openscriptures/HebrewLexicon` (CC-BY-4.0 / PD text, keyed to Strong's via the repo's `LexicalIndex.xml`). The Greek lexicon is supplemented by the **TBESG** — Tyndale Brief lexicon of Extended Strong's for Greek (`STEPBible/STEPBible-Data`, CC-BY 4.0; Abbott-Smith-based), the architect-approved substitute for Thayer's, keyed by extended/disambiguated Strong's. Strong's-tagged original text: the OSHB Hebrew (`openscriptures/morphhb`, CC-BY-4.0) is ingested per-word, and the Greek NT is tagged from the **STEPBible TAGNT** (Translators Amalgamated Greek NT, CC-BY 4.0), which carries disambiguated Strong's + morphology + edition membership for all 27 NT books. Required attribution: *STEP Bible, www.STEPBible.org*. **MorphGNT/SBLGNT** is snapshotted for provenance only (no Strong's; SBLGNT EULA); **OpenGNT** is rejected (CC-BY-SA copyleft).
- **Cross-references:** Treasury of Scripture Knowledge via the **OpenBible.info** `cross_references.txt` dataset (`a.openbible.info/data/cross-references.zip`, **CC-BY**). Required attribution: *Cross-reference data courtesy of www.openbible.info (CC-BY)*. Each edge carries a signed community relevance weight (range -86 .. +1278; the sign is preserved). Source + target OSIS refs are normalized to the canonical (KJV) spine; OpenBible already uses standard English/KJV versification, so no renumbering is needed. Refs that don't resolve to a real verse are flagged, never dropped.

A less-canonical source is never substituted silently — it's flagged for human review.

## Principles

- **Reproducible:** the same source snapshots produce bit-for-bit identical output. No timestamps in output, no unfixed random seeds.
- **Provenance on every chunk:** source, version, license, retrieval date.
- **Human-gated ship:** the pipeline produces a *candidate*; a manual spot-check (10 verses / 5 commentary passages / 3 confession sections / 5 Strong's lookups) gates any corpus that ships to an app build. The pipeline never marks a version ship-ready.
- **Versioned:** each shippable corpus is tagged (e.g. `corpus-v1.0.0`); the app builds against a specific version. Updates ship via App Store releases only.

## Layout

```
src/lampstand_corpus/   pipeline code (sources, schema, normalize, build, validate)
sources/                versioned public-domain snapshots + provenance manifest
docs/                   normalized-schema definition
tests/                  pipeline unit tests
output/                 compiled DBs + indices (gitignored)
```

## Develop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && pytest
```

The embeddings phase (P6) needs the heavy encoder, kept out of the default install
so the light snapshot/build/validate phases stay lean:

```bash
pip install -e ".[embeddings]"                       # sentence-transformers + torch
python -m lampstand_corpus.cli snapshot-model        # download BGE-small (pinned rev) -> models/ (gitignored)
python -m lampstand_corpus.cli build-embeddings      # chunk built DBs, encode (CPU, deterministic), write embeddings.sqlite + report
```

`embeddings.sqlite` carries the dense float32 vectors (dim 384) and a BM25 keyword
index in plain SQLite tables; both the model weights and the compiled index are
gitignored and never committed.

---

*LampStand · An Amphora Company.* See the app repo (`lampstand-ios`, private) for the consuming side.
