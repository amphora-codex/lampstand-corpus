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
- **Lexicons:** OpenScriptures `strongs`, `HebrewLexicon` (BDB), `morphgnt/sblgnt`; Thayer's.
- **Cross-references:** Treasury of Scripture Knowledge (OpenBible.info / CCEL).

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

---

*LampStand · An Amphora Company.* See the app repo (`lampstand-ios`, private) for the consuming side.
