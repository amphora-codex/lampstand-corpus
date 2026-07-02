# RESUME — corpus-v2 re-chunk release

**TEMPORARY working doc — delete before the release PR merges.**

## State: RE-CHUNK COMPLETE ✅ (encode 2.4 h; pipeline green; packs verified)

Everything ran and is committed: encode (210,123 vectors, determinism
cosine-tolerance 1.5e-7 — Option A, FLAGGED per rule 6), gold rebuild,
validate-retrieval (all 9 arms), sweep, package (expansion-dedupe fix landed),
full parity checks, and a **bit-identical rebuild across all 11 packs**.
306 tests + ruff green. Before/after numbers: `reports/retrieval_eval_v1.md`
+ the docs(reports) commit message.

## ONLY remaining work item (5am) — ARCHITECT-APPROVED

Supplement **WSC/WLC/Belgic proof-texts** from the **official Westminster
Standards repository** (already an approved confession source per CLAUDE.md;
confirm Belgic coverage before touching it — Belgic may need its own approved
edition). Per pipeline rules:
1. Versioned source snapshot + checksum into `sources/confessions/` + manifest.
2. Extraction into `section.proof_texts` (reuse the heidelberg/dort scripRef
   patterns in `confessions.py`).
3. `build-confessions` + validation counts; advisor spot-check note in report.
4. proof_texts changes do NOT re-key chunks (text unchanged — no re-encode).
   Then: `build-eval` → `validate-retrieval` → `package` so the prooftext
   reverse index + gold labels pick the new refs up; commit manifest/reports.

## Open flags

- `ondemand_search.sqlite` = 283.0 MB — **over the 250 MB display-text line;
  architect decision** (report §3 documents the fallback: drop chunk text,
  resolve display from per-resource packs; note Scripture text is now stored
  at BOTH child and parent granularity — dropping parent display text is a
  smaller lever worth pricing first).
- DRAFT pending advisor: `data/eval/hard_negatives_v1.json`,
  `data/eval/theological_synonyms_v1.json` (top-100 mined candidates).
- Irregular archaisms (saith/shew/cometh) fail mining thresholds honestly —
  advisor may hand-add to the synonyms file.
