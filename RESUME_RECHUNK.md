# RESUME — corpus-v2 re-chunk release (state as of session handoff)

**TEMPORARY working doc — delete before the release PR merges.**

## Running chain (survives session death)

- **Encode**: pid **72166** — `build-embeddings full`, 210,123 vectors, CPU,
  started ~20:10, expect ~4–6 h total. nohup'd + reparented; its stdout log
  path was session-temp and may vanish — the artifact write is what matters.
- **Watcher**: pid **75079** — `./.rechunk-resume.sh` (repo root, gitignored;
  nohup + disown). Waits for 72166, validates `output/embeddings.sqlite`
  (schema_version=2, n_chunks>300k), then runs
  `build-eval → validate-retrieval → sweep-retrieval → package`, logging to
  **`.rechunk-run.log`** (repo root, gitignored).
- **Killed-run caveat (verified)**: a killed encode resumes EXPENSIVELY, not
  cheaply — `write_embeddings` writes only at the END, and incremental reuse
  keys on content-addressed chunk ids, which ALL changed this release, so the
  prior DB offers 0 reuse. That is why 72166 was left running rather than
  relaunched. If it dies anyway: rerun
  `.venv/bin/python -m lampstand_corpus.cli build-embeddings full` (full cost).

## Check progress at 5am

```bash
cd ~/lampstand-corpus
tail -20 .rechunk-run.log             # pipeline stage log ("PIPELINE DONE" = all ran)
ps -p 72166 -o etime,cputime          # encode still running?
sqlite3 output/embeddings.sqlite "SELECT key,value FROM meta WHERE key IN ('schema_version','n_chunks','n_indexed','n_embedded')"
```

## 5am work items, in order

0. **NEW — ARCHITECT APPROVAL RECEIVED**: supplement WSC/WLC/Belgic proof-texts
   from the **official Westminster Standards repository** (already listed as an
   approved confession source in CLAUDE.md). Per pipeline rules: versioned
   source snapshot + checksum into `sources/confessions/` + manifest,
   extraction into `proof_texts`, validation counts, and an advisor spot-check
   note in the report. (Belgic may need its own approved edition — confirm
   what CLAUDE.md's approved list covers before touching anything beyond
   WSC/WLC.) Then rebuild confessions.sqlite; proof_texts changes do NOT
   change chunk ids (text unchanged) — no re-encode needed; rebuild the gold
   set + `package` afterwards so prooftext rows/labels pick it up.
1. If `.rechunk-run.log` shows PIPELINE DONE: review
   `reports/embeddings_validation_p6.txt` (determinism verdict),
   `reports/retrieval_eval_v1.md`, `reports/pack_diet_v1.md`,
   `reports/crossref_pack_v1.md`.
2. Parity checks (adapt the recipes; they lived in session scratchpad):
   - posting decode: `ondemand_search.sqlite` bm25_term blobs vs
     `output/embeddings.sqlite` bm25_posting rows (3 sampled terms).
   - doc_len vs bm25_doc (sample), vector bytes == quantize_int8(stored fp32)
     (sample), chunk counts match, int ids ascend with string ids.
   - crossref pack: edge parity for GEN 1:1 + PSA 23:1, expansion decode,
     prooftext rows for a WCF 11.1 verse, CC-BY attribution in meta.
3. Bit-identical rebuild: rerun `package`, compare pack SHAs in
   corpus_manifest.json against the first run (must match exactly).
4. Full `pytest` + `ruff check src tests` (incl. test_corpus_spotcheck against
   the new DBs).
5. Commit `corpus_manifest.json` + `reports/*.md` as
   `docs(reports): corpus-v2 re-chunk — measured before/after` including the
   BEFORE/AFTER harness table and final pack sizes. Then delete this file +
   the .gitignore entries for the run state in a final chore commit.

## BEFORE numbers (old chunking — cite from git history of reports/)

- Arms overall (452 q): bm25 r@5/10/20 .124/.186/.248, MRR .076, nDCG .032 ·
  dense .053/.082/.111/.031/.013 · hybrid .088/.157/.228/.052/.023.
- Per-cat r@20 (bm25): commentary-anchor .289 · crossref .393 · prooftext .060.
- Hardneg pairwise: bm25 58/60 · dense 60/60 · hybrid 60/60.
- int8 delta: dense r@20 −0.002 · hybrid −0.007 (others ≤0.002).
- Graph boost (hybrid-graph): prooftext r@20 .053→.087 · commentary .283→.217.
- Pack sizes: bundled_search 8.2 MB · bundled_crossrefs 2.9 MB ·
  ondemand_search 203.3 MB · ondemand_vectors 68.7 MB · totals 17.3 MB bundled
  / 508.3 MB on-demand. (v1 packs: ondemand_embeddings 1,990.5 MB,
  bundled_search 57.5 MB.)
- Old corpus profile: 175,442 chunks, all embedded. New: 317.7k chunks,
  303.4k BM25 docs, 210.1k vectors; 8,150 commentary chunks measured over the
  512-wordpiece window (now split into #s siblings); 13,795 pericope parents.

## Open flags (carry into the final report)

- DRAFT pending advisor: `data/eval/hard_negatives_v1.json`,
  `data/eval/theological_synonyms_v1.json` (written by the first package/
  harness run on the new corpus if not already present).
- Irregular archaisms (saith/shew/cometh) fail mining thresholds honestly —
  advisor candidates, do not force.
- Expansion arms (`bm25-expand`/`hybrid-expand`) + graph/int8 arms re-measure
  automatically in validate-retrieval; report their deltas vs the new baseline.
