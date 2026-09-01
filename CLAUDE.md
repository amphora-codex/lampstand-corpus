# CLAUDE.md

## What this repo is

`lampstand-corpus` is the offline Python (>= 3.11) pipeline that builds the read-only
corpus assets for **LampStand**, a native iOS Bible-study app by Amphora LLC. It ingests
versioned public-domain source snapshots (Bibles, Reformed confessions, commentaries,
lexicons, TSK cross-references), normalizes them, builds per-resource SQLite DBs plus a
dense (BGE-small, dim 384) and BM25 retrieval index, validates everything, and packages
checksummed `.sqlite` packs and Core ML `.mlpackage` models.

The sole consumer is the private iOS app repo at `~/Lampstand` (has its own CLAUDE.md).
Integration is file-based: the app reads the committed `corpus_manifest.json` (the
contract of record — every pack with bytes + sha256) and verifies each asset. Compiled
artifacts (`output/`, `models/`, pack files) are gitignored and synced out-of-band;
only the manifest and `reports/` are committed.

## Where the deep docs are

- `lampstand-corpus-knowledge.md` (repo root) — full knowledge export: architecture,
  P0–P8 phase model, data model, licensing ledger, risks. Read this first.
- `docs/pack-diet.md`, `docs/crossrefs-pack.md`, `docs/reranker-pack.md` — app-side
  reader contracts. `docs/pack-handoff-v2.1.md` — the app re-sync checklist.
- `docs/normalized-schema.md` — the normalized intermediate format.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # light install; extras: [embeddings] [coreml] [rerank]
ruff check . && pytest
```

Single CLI: `python -m lampstand_corpus.cli <command>`. Each resource family has a
`snapshot* / build* / validate*` triad (e.g. `snapshot-confessions`,
`build-confessions`, `validate-confessions`; bare `snapshot`/`build`/`validate` are the
Bibles). Other key commands: `snapshot-model`, `build-embeddings` (incremental by
default; `build-embeddings full` forces re-encode), `validate-embeddings`, `build-eval`,
`validate-retrieval`, `sweep-retrieval`, `rerank-eval`, `coreml-export`,
`coreml-export-reranker`, `package` (v2 packs; `package fp32` keeps float32 vectors).

Pack transport (Cloudflare R2, needs `jq` + `npx wrangler`):
- `scripts/upload-packs-r2.sh [packs-dir]` — uploads on-demand packs to
  `lampstand-packs/<corpus_version>/`; refuses unless manifest `ship_ready` is true.
  Packs dir defaults to `~/Lampstand/Resources/Corpus`.
- `scripts/verify-packs-remote.sh <public-base-url>` — re-downloads every pack (~610 MB)
  and verifies size + sha256 against the manifest. Run after upload, before TestFlight.

## Current status (2026-09-01)

- Manifest: `corpus-v2.1.2`, `ship_ready: true` — architect-affirmed 2026-08-11
  (commit a288dc5). Recent work: drop pure structural-header commentary chunks at build
  time, plus the R2 upload / remote-verification scripts above.
- Branches: `main` is default; work happens on `feature/<topic>` branches (current:
  `feature/commentary-header-filter`, also `feature/omitted-verses`,
  `feature/retrieval-eval`).

## Sharp edges

- **Human-gated ship.** The pipeline only ever emits a candidate; the architect's
  23-point spot-check gates `ship_ready` and the version tag. Never set `ship_ready`
  or bump `corpus_version` yourself, and don't casually regenerate the manifest — it
  currently carries the signed-off v2.1.2 checksums.
- **Determinism is a design constraint.** Same snapshots must yield byte-identical
  output: no timestamps in output, fixed seeds, CPU single-threaded encoding.
  ~1e-7 float jitter is accepted under the architect's "Option A" (cosine tolerance,
  recorded, never silently ignored).
- **Integer chunk ids are per-corpus-version** — they renumber on any chunk
  add/remove. Cross-version identity is the string chunk id; the app must never
  persist integer ids across corpus updates.
- **Never commit artifacts.** `.sqlite` DBs, embedding indexes, model weights, and
  source snapshot bytes are gitignored by policy; only code, docs, manifest, and
  plaintext reports land in git. The repo is deliberately public — no secrets,
  credentials, or licensed text may enter it.
- **Licensing is load-bearing.** PD / CC0 / CC-BY sources only, no copyleft; CC-BY
  sources carry attribution strings the app must render. Details and flagged items
  (KJV UK Crown patent, SBLGNT EULA, Spurgeon OCR) are in the knowledge file.
- **Validation errors and flags are for human adjudication** — the pipeline flags
  degraded or non-resolving content, never silently drops or substitutes it.
