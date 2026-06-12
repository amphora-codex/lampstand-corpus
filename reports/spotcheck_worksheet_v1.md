# LampStand corpus — human spot-check worksheet (v1 candidate)

**This is the architect's job, not the pipeline's.** The pipeline produced the
candidate (`corpus-v1.0.0-candidate`) + `reports/corpus_validation_v1.md`; this
worksheet is the 23-point manual gate (CLAUDE.md §Human spot-check). **No corpus
version is tagged or shipped until every item below passes and the flagged
singletons are adjudicated.**

How to use: each item names exactly **what to check**, **against what authority**,
and gives the corpus's current value so you can compare. Mark `PASS` / `FAIL` /
`NEEDS-FIX` in the result column and initial it.

The 23 points = **10 verses + 5 commentary passages + 3 confession sections +
5 Strong's lookups**. The flagged-singletons section after them lists the specific
ambiguities the validators surfaced — adjudicate those too before tagging.

Spread of the 23 points was chosen to land on the known risk areas: omitted-verse
variants, the BSB Psalm-superscription case, Spurgeon OCR + mis-anchored heads,
the Henry off-canon anchor, the WCF 1788 loci, and the lexicon edge cases.

---

## A. Ten verses across ten books (vs a printed/authoritative reference)

Compare each against a trusted print or a second digital edition. The omitted-verse
and superscription rows are deliberately included — confirm the framing, not just the
words.

| # | Reference | Translation | Corpus says | Check against | Result |
|---|---|---|---|---|---|
| 1 | **Genesis 1:1** | KJV | "In the beginning God created the heaven and the earth." | KJV print | ☐ |
| 2 | **John 3:16** | BSB | "For God so loved the world that He gave His one and only Son…" | bereanbible.com | ☐ |
| 3 | **Psalm 23:1** | BSB | superscription = "A Psalm of David."; v1 = "The LORD is my shepherd; I shall not want." | BSB print — confirm superscription is stored on the verse, not merged into body | ☐ |
| 4 | **Psalm 51:1** | BSB | superscription = "For the choirmaster. A Psalm of David. When Nathan the prophet came to him after his adultery with Bathsheba."; **v1 body present** ("Have mercy on me, O God…") | BSB — this is the **superscription-offset case**: confirm BSB folds the superscription into v1 and still carries a v1 row (the earlier dropped-v1 bug is fixed; offset vs KJV is 0) | ☐ |
| 5 | **Matthew 18:11** | BSB | **omitted=1**, body empty; source_note = "For the Son of Man came to save the lost" (recovered from BSB footnote) | Confirm BSB legitimately omits this critical-text verse and the tap-to-explain note reads irenically | ☐ |
| 6 | **Matthew 18:11** | KJV | present, body = "For the Son of man is come to save that which was lost." | KJV print — the SAME ref is a real verse in KJV; confirm the omitted-variant union resolves in BOTH translations | ☐ |
| 7 | **Romans 16:24** | ASV | **omitted=1**, body empty; source_note = "Some ancient authorities insert here v. 24 The grace of our Lord Jesus Christ…" | ASV — confirm this is a genuine ASV critical-text omission | ☐ |
| 8 | **Romans 16:25** | WEB | **empty / flagged** (WEB ROM 16 versification: max verse 25 vs reference 27; ROM 14 max 26 vs 23 — TR doxology placement) | WEB print — **adjudicate**: is the empty ROM 16:25 a parser miss or a real WEB versification quirk? (P1 flag) | ☐ |
| 9 | **Isaiah 53:5** | BSB | "But He was pierced for our transgressions, He was crushed for our iniquities…" | BSB print | ☐ |
| 10 | **Revelation 22:21** | KJV | "The grace of our Lord Jesus Christ be with you all. Amen." | KJV print (last verse — confirms canon tail intact) | ☐ |

---

## B. Five commentary passages (vs the canonical edition)

These deliberately include the OCR + mis-anchor cases. For the mis-anchored Spurgeon
heads, the check is **"is the missing psalm's content present in the PRECEDING psalm's
block?"** — content was absorbed, not deleted.

| # | Passage | Corpus state | Check against | Result |
|---|---|---|---|---|
| 11 | **Spurgeon — Treasury of David, Psalm 23** (OCR sample) | present (`PSA.23.0#titl1…`); note OCR garble in opening title ("I%ere U no intpired title…") | Printed *Treasury of David* — judge whether visible-but-readable OCR is v1-acceptable (the v1-vs-v1.1 gate) | ☐ |
| 12 | **Spurgeon — Psalm 44** | **0 chunks anchored to Ps 44**; content absorbed into **Ps 43** block (`Ps43` = 13 chunks, larger than neighbours) — OCR-lost head | Printed Treasury — confirm Ps 44's exposition is present inside the Ps 43 block and split it; verify it's a mis-anchor, not a deletion | ☐ |
| 13 | **Spurgeon — Psalm 133 & 143** | **0 chunks each**; absorbed into **Ps 132** (129 chunks) and **Ps 142** (90 chunks) respectively | Printed Treasury — same mis-anchor check for both | ☐ |
| 14 | **Spurgeon — gap-fill Psalms 104–118** (sample, e.g. Ps 110) | gap-filled from an ALTERNATE PD scan `treasuryofdavidc0005spur` (vol. 5, 1882), NOT the `*spurgoog` set | Printed Treasury **vol. 5** — confirm the alternate-scan psalms read correctly and match the rest in segmentation | ☐ |
| 15 | **Matthew Henry — 2 Chronicles 27** | anchors present but **6 chunks tagged off-canon** (`2CH.27.16-27`, end-verse 27 > chapter length 9) + ranges `2CH.27.6-15` overrun; Ps 27 also 35/36 ch partial | Henry print on 2 Chr 27 — confirm the CCEL verse-range labels overrun the chapter (anchored at start) and content is correct; decide whether to re-anchor | ☐ |
| 16 (bonus) | **Calvin — Psalm 119** | **363 chunks** for one psalm (longest single-chapter block); first chunk = "COMMENTARY" | Calvin's *Comm. on Psalms* — confirm the 363-chunk Ps 119 block is genuine (Ps 119 is 176 vv, Calvin is voluminous) and **not** a CCEL div-merge error | ☐ |

> (Item 16 is included as a sixth commentary check because Calvin Ps 119 is the single
> largest anomaly-by-size; pick any 5 of 11–16 to satisfy the formal count, but the
> mis-anchored heads 12–13 and the off-canon Henry 15 are the highest-risk and should
> not be skipped.)

---

## C. Three confession sections (vs the authoritative source)

| # | Section | Corpus state | Check against | Result |
|---|---|---|---|---|
| 17 | **WCF Ch. 23 (Of the Civil Magistrate), §1 + §3** | §1 original 1646/47 text present; **§3 carries BOTH** the original text AND `amendment_1788` note + verbatim `amendment_1788_text` ("Civil magistrates may not assume to themselves the administr…") | Printed WCF original **and** the 1788 American revision — confirm original is verbatim and the 1788 revised wording is correctly marked/separate | ☐ |
| 18 | **WCF Ch. 24 (Of Marriage and Divorce), §4** | original text present; `amendment_1788` note present BUT `amendment_1788_text` = **NULL** (justified gap: the CCEL American ch.24 is a later denominational rewrite, not the 1788 text — flagged for human reconstruction, NOT fabricated) | The 1788 American revision (original minus the dropped consanguinity clause) — **architect must source/reconstruct the 1788 verbatim separately**; confirm nothing was invented | ☐ |
| 19 | **1689 LBCF Ch. 26 (Of the Church), §1** | present ("The Catholick or universal Church, which (with respect to the internal…"); 1689 totals 32 ch / 160 sections | Printed 1677/89 LBCF | ☐ |
| 20 (bonus) | **Belgic Confession — article count** | **37/37 articles** | Confirm 37 is correct (1840 RPDC translation) — count check | ☐ |

> (Pick any 3 of 17–20 for the formal count; 17 and 18 — the WCF 1788 loci — are the
> highest-risk and should both be done.)

---

## D. Five Strong's word lookups (vs Strong's Concordance)

Deliberate mix: 2 Hebrew, 2 Greek, plus the two edge cases (a "Not Used" placeholder
and an Aramaic "corresponding-to" entry).

| # | Strong's | Corpus says | Check against | Result |
|---|---|---|---|---|
| 21 | **H430** (ʼĕlôhîym) | lemma אֱלֹהִים · "gods in the ordinary sense; but specifically used…" · KJV: "angels, × exceeding, God (gods)…" | Strong's Concordance H430 | ☐ |
| 22 | **G26** (agápē) | lemma ἀγάπη · "love, i.e. affection or benevolence…" · KJV: "(feast of) charity(-ably), dear, love." | Strong's Concordance G26 | ☐ |
| 23 | **G2424** (Iēsoûs) | lemma Ἰησοῦς · "Jesus (i.e. Jehoshua), the name of our Lord…" | Strong's Concordance G2424 | ☐ |
| 24 (edge) | **G2717** — "Not Used" placeholder | ALL English fields NULL, `strongs_linked=1`, kept so the G1–G5624 span is explicit (1 of 101 such placeholders in the CC0 Greek edition) | Strong's — confirm G2717 is a genuine unused/skipped number, not a parser drop | ☐ |
| 25 (edge) | **H426** (ʼĕlâhh) — Aramaic | lemma אֱלָהּ · "God" · derivation = "(Aramaic) corresponding to H433;" | Strong's H426 — confirm the Aramaic "corresponding-to" cross-pointer resolves to H433 | ☐ |

> (Pick any 5 of 21–25 for the formal count; do at least one of the two edge cases
> 24/25 — they exercise the placeholder-span and Aramaic-pointer handling.)

---

## E. Flagged singletons to adjudicate before tagging

These are the specific residuals the validators could not auto-resolve. Each needs a
human decision (keep / fix / re-source). They are **not** silently resolved.

| Item | What to decide | Where flagged |
|---|---|---|
| **lbcf 5.4 → PSA 1:21** | Genuine bad/typo proof-text ref (Psalm 1 has only 6 verses) — resolves in NO translation. Adjudicate the correct citation; do NOT renumber blindly. | P2 (confessions): the 1 confession error |
| **3 John 1:15 cross-ref edge** | `3JN 1:15 → JHN 10:3` source verse doesn't resolve on the KJV spine (3 John has 14 vv in KJV / 15 in critical text). Decide versification handling for the 2-3 John tail. | P5: 1 of 5 crossref errors |
| **4 non-resolving crossref targets** | All point into the `2JN 1:1–3JN 1:15` range (book-crossing, critical-text v15). Same 2-3 John tail decision. | P5: 4 of 5 crossref errors |
| **18 book-crossing TSK target ranges** | e.g. `NUM 3:1 → LEV 27:34-NUM 1:1`, `EZR 4:3 → 2CH 36:22-EZR 1:3`. Confirm these are genuine TSK spans, not digitization artifacts. | P5: structural-ambiguity flags |
| **Henry 2CH 27:16-27 off-canon (×6) + 5 range overruns** | CCEL end-verses exceed chapter length (e.g. 2CH 28:20-36 vs ch len 27; ACT 26:24-44 vs 32; ISA 7:17-28 vs 25; PRO 31:10-33 vs 31). Decide re-anchor vs keep-anchored-at-start. | P3: the 6 commentary errors + overrun flags |
| **Spurgeon mis-anchored heads 44, 133, 143** | Content absorbed into preceding psalm (items 12–13 above). Split against printed edition before ship. | P3 flags |
| **Spurgeon OCR quality (v1 vs v1.1 gate)** | Residual intra-word OCR substitution (garble mean 0.019, max 0.286); Hebrew/Greek quotations are OCR-garbage, NOT reconstructed. Decide: ship visible-but-readable OCR as v1, or defer Spurgeon to v1.1. | P3 OCR-quality section |
| **Calvin / JFB / Henry repeated anchor keys** | 45 Calvin + 1 JFB + 1 Henry verse-anchor keys repeat (CCEL split a chapter across sibling divs; later occurrence got a `~N` suffix). Verify benign split, not merge error. | P3 flags |
| **Truncated-chunk samples (485)** | 485 chunks hit the 6000-char cap before encoding (e.g. `calvin:GEN.1.1#p8`, `calvin:PSA.51.4#p5`). Confirm no long block lost retrieval-needed material. | P6 flag 3 |
| **BDB stub samples (4516)** | 4516 BDB lemma-only stubs have no `<def>` gloss and were skipped from embedding (kept in lexicons.sqlite). Confirm none should instead merge into a linked Strong's entry. | P4 / P6 flag 1 |
| **Heidelberg Lord's Day 2 vs 29** | LD marker reads 2 where 29 expected in sequence (likely CCEL source typo). Verify against printed Heidelberg. | P2 flag |
| **WCF chapter-only proof-texts (34)** | e.g. `2CH '29, 30'`, `HEB '8, 9, 10 chapters'` — chapter-level citations the resolver left out (expected for these source forms). Confirm none should be verse-resolved. | P2 flags |

---

## F. Packaging / licensing gates (confirm before tagging)

| Item | Check | Result |
|---|---|---|
| Bundled pack size | 63.7 MB — under the ~150–200 MB target. Confirm acceptable for binary. | ☐ |
| On-demand embeddings size | ~1.9 GB (`ondemand_embeddings.sqlite`). Decide whether to accept as-is or pursue the float16 / integer-chunk-id proposals in `corpus_validation_v1.md` §2 (proposals only — nothing lossy implemented). | ☐ |
| Bundled search correctness | BM25 stats recomputed over the BSB+WSC subset (N=3382); vectors byte-identical to full index. Confirm. | ☐ |
| CC-BY attribution strings | The app's acknowledgements screen must render the attribution for TBESG/TAGNT ("STEP Bible, www.STEPBible.org"), OSHB, and TSK ("courtesy of www.openbible.info"). Confirm wired into the iOS acknowledgements screen. | ☐ |
| CC-BY Hebrew Strong's | Decide: keep the CC-BY HebrewStrong.xml or source a CC0/PD Hebrew later (no CC0 machine-readable Hebrew Strong's exists in a source of record). | ☐ |
| Thayer's substitute | Confirm TBESG is the accepted Thayer's stand-in for v1 (a canonical Thayer's proper would need fresh approval). | ☐ |

---

## Sign-off

- [ ] All 23 points (A–D) PASS.
- [ ] All flagged singletons (E) adjudicated.
- [ ] Packaging/licensing gates (F) confirmed.
- [ ] **Only then:** tag `corpus-v1.0.0` and update `corpus_manifest.json`
      `corpus_version` from the `-candidate` placeholder.

Architect: _____________________  Date: __________
