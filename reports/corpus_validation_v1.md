# LampStand corpus — consolidated v1 validation report

**CANDIDATE ONLY — NOT SHIP-READY.** The pipeline never marks a corpus version
ship-ready. The architect's 23-point human spot-check (CLAUDE.md §Human spot-check)
gates ship. See `reports/spotcheck_worksheet_v1.md` for the pre-filled checklist.

Corpus version placeholder: **`corpus-v1.0.0-candidate`** (tagged only after the
spot-check passes). Source snapshots retrieved **2026-06-10 / 2026-06-11**.

This document rolls up the six per-phase validators (P1 Bibles, P2 confessions,
P3 commentaries, P4 lexicons, P5 cross-refs, P6 embeddings) into one summary, plus
the P7 packaging outcome, the reproducibility statement, and the license roll-up.
Every per-phase report remains authoritative for its own detail
(`reports/*_validation_p*.txt`); this is the index over them.

---

## 1. Per-resource summary

| Resource | Scope / counts | Errors | Flags | Source report |
|---|---|---:|---:|---|
| **Bibles** | 4 translations · 66/66 books each · ASV 31102 v · BSB 31086 v (+16 omitted rows) · KJV 31102 v · WEB 31103 v · 16-ref omitted-variant union resolves in every translation | 0 | 3 | `bible_validation_p1.txt` |
| **Confessions** | 7 docs · 895 sections · WCF 33 ch / WLC 196 Q / WSC 107 Q / 1689 32 ch / Belgic 37 art / Heidelberg 129 Q / Dort 5 heads · WCF prose cross-checked vs Wikisource Burges-1646 (0 divergence in 14 samples) | 1 | 54 | `confessions_validation_p2.txt` |
| **Commentaries** | Henry (66 bk) · JFB (66 bk) · Calvin (26 bk: Gen+Ps+NT) · Spurgeon (147/150 Ps, OCR candidate) · 128,885 comment chunks | 6 | 151 | `commentaries_validation_p3.txt` |
| **Lexicons** | Strong's Greek (CC0, 5624) · Strong's Hebrew (CC-BY, 8674) · BDB (CC-BY, 11845) · TBESG (CC-BY, 11035) · OSHB tagged Hebrew (306,785 w) · TAGNT tagged Greek (141,720 w) · 0 orphan Strong's | 0 | 318 | `lexicons_validation_p4.txt` |
| **Cross-references** | TSK / OpenBible (CC-BY) · 344,799 edges · 29,364 source verses · votes −86..1278 | 5 | 19 | `crossrefs_validation_p5.txt` |
| **Embeddings + BM25** | BGE-small-en-v1.5 (dim 384, pinned rev) · 175,442 chunks · BM25 145,281 terms / 10,831,300 postings · 3/3 smoke queries PASS | 1 | 4618 | `embeddings_validation_p6.txt` |
| **TOTAL** | | **13** | **5163** | |

> "Errors" and "flags" are both **for human adjudication**, never silent fixes. An
> "error" is an off-spec condition (e.g. an off-canon verse ref); a "flag" is a
> structural ambiguity or licensing/quality note. The large embeddings flag count
> (4618) is dominated by 4516 BDB lemma-only stubs + 101 Strong's "Not Used"
> placeholders skipped from embedding (kept in the lexicon DB) — these are expected,
> not defects.

### Error/flag totals by category

- **Total errors: 13** — confessions 1 (lbcf 5.4 off-canon proof-text), commentaries
  6 (Henry 2CH 27:16-27 off-canon anchor ×6), cross-refs 5 (1 non-resolving source +
  4 non-resolving targets, all in the 2-3 John tail), embeddings 1 (determinism not
  bit-for-bit — within cosine-tolerance, architect-approved Option A).
- **Total flags: 5163** — embeddings 4618 (≈98% are the BDB-stub / Not-Used skips),
  lexicons 318 (mostly "no `<def>` gloss; meaning in derivation" upstream quirks),
  commentaries 151 (Spurgeon OCR 98 + Henry 51 + Calvin/JFB anchor-key repeats),
  confessions 54 (WCF 34 chapter-only proof-text citations + lbcf 19 + Heidelberg 1),
  cross-refs 19 (18 book-crossing ranges + 1 review note), Bibles 3 (WEB Romans 16
  versification).

---

## 2. P7 packaging outcome (bundled vs on-demand split)

Architect-locked split implemented in `src/lampstand_corpus/package.py`
(`python -m lampstand_corpus.cli package`). Pack `.sqlite` files are written under
`output/packs/` (gitignored); only `corpus_manifest.json` (repo root) is committed.

### Bundled pack — ships in the app binary

Target: **well under ~150–200 MB.** Actual: **63.7 MB — PASS, no size flag.**

| File | Bytes | Contents |
|---|---:|---|
| `bundled_bibles.sqlite` | 6,483,968 (6.2 MB) | BSB only · 66 books · 31,102 verses |
| `bundled_confessions.sqlite` | 61,440 (0.06 MB) | WSC only · 107 Q&A |
| `bundled_search.sqlite` | 60,252,160 (57.5 MB) | BSB-scripture + WSC dense vectors (3,382 chunks) + BM25 **recomputed over the subset** (N=3382, vocab 14,507, 365,626 postings) |
| **Bundled total** | **66,797,568 (63.7 MB)** | |

The bundled search index **reuses the exact stored float32 vectors** (verified
byte-identical to the full index — no re-encode) and **recomputes BM25 statistics
over only the bundled corpus** (avgdl, document frequencies, N), so app-side scoring
is correct against the bundled subset rather than inheriting the global corpus stats.

### On-demand pack(s) — free, downloaded on first launch

| File | Bytes | Contents |
|---|---:|---|
| `ondemand_bibles.sqlite` | 20,291,584 (19.4 MB) | KJV + ASV + WEB |
| `ondemand_confessions.sqlite` | 1,163,264 (1.1 MB) | WCF, WLC, 1689, Belgic, Heidelberg, Dort |
| `ondemand_commentaries.sqlite` | 132,046,848 (125.9 MB) | Henry, JFB, Calvin, Spurgeon |
| `ondemand_lexicons.sqlite` | 68,857,856 (65.7 MB) | Strong's G/H, BDB, TBESG + OSHB/TAGNT tagged text |
| `ondemand_crossrefs.sqlite` | 25,362,432 (24.2 MB) | full TSK |
| `ondemand_embeddings.sqlite` | 1,990,508,544 (**1,898 MB ≈ 1.9 GB**) | full dense + BM25 index, 175,442 chunks |
| **On-demand total** | **2,238,230,528 (2,134 MB ≈ 2.1 GB)** | |

On-demand pack files are **byte-faithful filtered copies** of the built DBs (same
schema), so the on-demand format == the built format; nothing new needs revalidation.

### SIZE FLAG (FLAG-FOR-HUMAN — proposals only; nothing lossy implemented)

> **FLAG: `ondemand_embeddings.sqlite` is ~1.9 GB (full embeddings index).** This is
> the architect-anticipated ~2 GB pack. It is well within feasibility for a free
> first-launch download, but if the architect wants it smaller, here are options —
> **none implemented silently; all change retrieval fidelity or schema and need a
> decision.** The bundled pack is unaffected (63.7 MB, comfortably in target).
>
> 1. **float16 vector storage (proposal).** Storing the 384-d vectors as float16
>    instead of float32 halves the vector blobs from ~257 MB to ~128 MB — a ~128 MB
>    saving, bringing the pack to ~1.77 GB. **Caveat: this is the smaller lever.** The
>    vectors are only 257 MB of the 1.9 GB; cosine recall loss from float16 is
>    typically negligible but must be measured against the eval set before adoption.
> 2. **Integer chunk-id remap for the BM25 tables (proposal — the larger lever).** The
>    real bulk is the **BM25 posting table: 10,831,300 rows**, each storing a 28-char
>    TEXT `chunk_id`, plus the `(term_id, chunk_id)` PK index and `idx_posting_chunk`.
>    Remapping `chunk_id` to a 4-byte integer across `chunk`/`embedding`/`bm25_*` would
>    cut the dominant ~1.6 GB BM25/index portion substantially. This is a **shared
>    schema change** to `build_embeddings.py` affecting the app reader, so it is the
>    architect's + iOS side's call, not a corpus-side silent edit.
> 3. **Ship the on-demand search as two sub-packs** (dense-only vs BM25) so a
>    device can fetch dense vectors first and the keyword index lazily. Pure
>    packaging change, no fidelity loss, but adds app-side download orchestration.
>
> Recommendation: **leave as-is for the candidate** (1.9 GB is a one-time free
> download and the data is correct), and treat (2) as the highest-value future
> optimization if size becomes a constraint. Do not adopt (1) without an eval-set
> recall check.

---

## 3. Reproducibility statement

- **Per-resource DBs (P1–P5):** bit-for-bit reproducible. Each builder inserts rows
  in a fixed canonical order, writes no wall-clock timestamps, and uses no unfixed
  random seeds. Covered by `tests/test_reproducibility.py` (build twice → identical
  SHA-256).
- **Packaging (P7):** bit-for-bit reproducible. Verified by running `package` twice
  and comparing SHA-256 across all 9 pack files (identical), and by
  `tests/test_package.py::test_packaging_is_deterministic`. Pack files are
  deterministic filtered copies (fixed row order, schema copied verbatim) and the
  bundled BM25 is recomputed deterministically (term ids assigned by sorted term).
- **Embeddings (P6):** reproducible under the **architect-approved Option A
  (cosine-tolerance)**. Two CPU re-encodes of a fixed 400-chunk sample matched to
  `min cosine 1.00000000` with `max abs diff 1.49e-07` — not byte-identical, **flagged
  for the architect, not accepted silently** (CLAUDE.md rule 6). A reused vector is
  provably the embedding of byte-identical text under the pinned model revision
  (`5c38ec7c…`); CPU float reductions jitter at ~1e-7. The incremental re-encode reuses
  vectors only on a content-addressed id match (resource_type+source+anchor+text
  checksum), which guarantees identical input text.

---

## 4. License / attribution roll-up

The committed `corpus_manifest.json` carries the full per-source acknowledgements the
app renders (23 entries: name, resource type, license, attribution, source URL,
retrieval date, source checksum). Summary:

| License class | Sources |
|---|---|
| **CC0 / public domain** | BSB; ASV; KJV; WEB; WCF, WLC, WSC, 1689, Belgic, Heidelberg, Dort; Calvin, Henry, JFB, Spurgeon; Strong's Greek (morphgnt CC0) |
| **CC-BY 4.0 (attribution required)** | Strong's Hebrew (OpenScriptures); BDB (OpenScriptures); TBESG (STEPBible — *STEP Bible, www.STEPBible.org*); OSHB tagged Hebrew (*Open Scriptures Hebrew Bible, github.com/openscriptures/morphhb*); TAGNT tagged Greek (STEPBible — *STEP Bible, www.STEPBible.org*); TSK cross-refs (*Cross-reference data courtesy of www.openbible.info, used under CC-BY*) |
| **MIT (tool)** | BGE-small-en-v1.5 embedding model (weights gitignored; not redistributed in the corpus, recorded for provenance) |

- **No CC-BY-SA / copyleft** text is bundled: the prior CC-BY-SA Strong's `.js` Hebrew
  and OpenGNT were deliberately rejected; SBLGNT is snapshot-only (provenance), not
  ingested.
- **CC-BY attribution is mandatory in the app's acknowledgements screen** (CLAUDE.md
  memory: corpus-licensing). The required strings are in the manifest's
  `acknowledgements[*].attribution` fields.
- **OPEN licensing decisions for the architect** (carried from P4): (a) keep the CC-BY
  Hebrew Strong's or source a CC0/PD Hebrew later; (b) TBESG is the approved Thayer's
  substitute — a canonical Thayer's proper would still need approval.

---

## 5. Tests / lint

- `ruff check .` — clean.
- `pytest` — full suite green (incl. `tests/test_package.py`, 5 new P7 tests, and
  the reproducibility suite).

---

## 6. What still gates ship

This is a **candidate**. Before any corpus ships to an app build the architect must
run the 23-point spot-check in `reports/spotcheck_worksheet_v1.md` (10 verses /
5 commentary passages / 3 confession sections / 5 Strong's lookups), adjudicate the
flagged singletons listed there, and only then tag `corpus-v1.0.0`. The pipeline does
not tag and does not self-certify.
