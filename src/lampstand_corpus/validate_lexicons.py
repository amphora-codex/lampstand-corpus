"""Validation pass over normalized lexicon entries (+ optional tagged text).

Per CLAUDE.md pipeline rules, the report flags (never silently fixes):
  * Coverage: Greek Strong's (G1-G5624) and Hebrew Strong's (H1-H8674) — actual
    entry count vs. the expected span, with interior gaps listed.
  * Orphan Strong's: every Strong's number REFERENCED — by the OSHB tagged text,
    or by a BDB entry's Strong's link — that has NO Strong's-dictionary entry.
  * Statistical anomalies: entries with empty/missing definition; duplicate
    Strong's keys; unusually long definitions.
  * Tagged text (P4b): per-word rows whose VerseRef falls outside the canonical
    spine; words whose lemma yields no Strong's number (prefix-only particles are
    expected — counted, not error-flagged).

Residuals a human must adjudicate are listed verbatim. The pipeline never marks
output ship-ready — the architect's spot-check gates ship.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import books
from .lexicons import (
    STRONGS_GREEK_MAX,
    STRONGS_HEBREW_MAX,
    ParsedLexicon,
    ParsedTaggedText,
)

LONG_DEF_CHARS = 4_000


@dataclass
class LexiconReport:
    lexicon: str
    language: str = ""
    n_entries: int = 0
    n_linked: int = 0            # entries keyed by a Strong's number
    n_with_definition: int = 0
    expected_max: int = 0        # G5624 / H8674 span (Strong's dictionaries only)
    distinct_strongs: int = 0
    coverage_gaps: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)
    empty_definitions: list[str] = field(default_factory=list)
    long_definitions: list[str] = field(default_factory=list)
    parser_flags: list[str] = field(default_factory=list)

    @property
    def error_total(self) -> int:
        return len(self.duplicate_keys) + len(self.empty_definitions)

    @property
    def flag_total(self) -> int:
        return (len(self.coverage_gaps) + len(self.long_definitions)
                + len(self.parser_flags))


@dataclass
class TaggedReport:
    source: str
    language: str = ""
    n_words: int = 0
    n_verses: int = 0
    n_books: int = 0
    n_words_with_strongs: int = 0
    bad_refs: list[str] = field(default_factory=list)         # malformed -> ERROR
    masoretic_divergences: list[str] = field(default_factory=list)  # -> FLAG
    parser_flags: list[str] = field(default_factory=list)

    @property
    def error_total(self) -> int:
        return len(self.bad_refs)

    @property
    def flag_total(self) -> int:
        return len(self.parser_flags) + (
            1 if self.masoretic_divergences else 0
        )


@dataclass
class OrphanReport:
    """Strong's numbers referenced but missing a lexicon entry."""

    from_tagged: list[str] = field(default_factory=list)   # OSHB Hebrew -> Strong's
    from_greek: list[str] = field(default_factory=list)    # TAGNT Greek -> TBESG
    from_bdb: list[str] = field(default_factory=list)

    @property
    def error_total(self) -> int:
        return len(self.from_tagged) + len(self.from_greek) + len(self.from_bdb)


def _gaps(present: set[int], max_n: int, prefix: str, *, limit: int = 40) -> list[str]:
    """Return a compact list of missing Strong's numbers in 1..max_n."""
    missing = [n for n in range(1, max_n + 1) if n not in present]
    if not missing:
        return []
    # Collapse consecutive runs into ranges for a readable report.
    ranges: list[str] = []
    start = prev = missing[0]
    for n in missing[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(f"{prefix}{start}" if start == prev else f"{prefix}{start}-{prefix}{prev}")
        start = prev = n
    ranges.append(f"{prefix}{start}" if start == prev else f"{prefix}{start}-{prefix}{prev}")
    note = f"{len(missing)} missing Strong's numbers in 1..{max_n}"
    head = ranges[:limit]
    if len(ranges) > limit:
        head.append(f"... (+{len(ranges) - limit} more ranges)")
    return [f"{note}: " + ", ".join(head)]


def validate_lexicon(pl: ParsedLexicon) -> LexiconReport:
    rep = LexiconReport(lexicon=pl.id, language=pl.language)
    rep.parser_flags = list(pl.flags)
    rep.n_entries = len(pl.entries)
    present: set[int] = set()
    seen_keys: set[str] = set()
    bdb_no_def = 0
    for e in pl.entries:
        if e.strongs:
            rep.n_linked += 1
            key = e.strongs
            if key in seen_keys and pl.lexicon == "strongs":
                rep.duplicate_keys.append(f"{pl.id}: duplicate {key}")
            seen_keys.add(key)
            if e.strongs[1:].isdigit():
                present.add(int(e.strongs[1:]))
        if e.definition:
            rep.n_with_definition += 1
            if len(e.definition) > LONG_DEF_CHARS:
                rep.long_definitions.append(
                    f"{pl.id}:{e.strongs or e.raw_key} "
                    f"({len(e.definition)} chars)"
                )
        else:
            # An entry is a true ERROR only when it carries NO meaning at all (no
            # strongs_def, no kjv_def, no derivation). A known upstream Strong's
            # quirk: ~20 Greek entries fold the gloss into `derivation`/`kjv_def`
            # with an empty `strongs_def` (e.g. G976 βίβλος) — those are FLAGGED,
            # not errored, since the meaning is present. BDB has many
            # cross-reference-only stubs with no <def>; likewise flagged.
            label = f"{pl.id}:{e.strongs or e.raw_key}"
            has_other_meaning = bool(e.kjv_def or e.derivation)
            if pl.lexicon == "strongs" and not has_other_meaning:
                rep.empty_definitions.append(label)
            elif pl.lexicon == "strongs":
                rep.parser_flags.append(
                    f"{label} (no strongs_def gloss; meaning in "
                    f"derivation/kjv_def — upstream quirk)"
                )
            else:
                bdb_no_def += 1
    if bdb_no_def:
        rep.parser_flags.append(
            f"bdb: {bdb_no_def} entries have no <def> gloss (BDB cross-reference / "
            f"root-pointer stubs — expected, not errored)"
        )
    rep.distinct_strongs = len(present)
    if pl.lexicon == "strongs":
        rep.expected_max = (STRONGS_GREEK_MAX if pl.language == "greek"
                            else STRONGS_HEBREW_MAX)
        prefix = "G" if pl.language == "greek" else "H"
        rep.coverage_gaps = _gaps(present, rep.expected_max, prefix)
    return rep


def validate_tagged(pt: ParsedTaggedText) -> TaggedReport:
    """Validate the OSHB Strong's-tagged Hebrew words.

    OSHB is numbered in the **Masoretic (Hebrew) versification**, which differs
    from the English/KJV spine in well-known places (Joel has 4 Hebrew chapters
    vs 3 English; many chapters shift a verse where Hebrew counts a superscription
    or splits differently; Psalms count the superscription as v.1). Those are NOT
    corpus errors — they are versification differences the P5 canonical-map
    reconciles. So a verse that is malformed (unknown book, chapter/verse <= 0) is
    an ERROR, while a well-formed Masoretic ref that simply exceeds the English
    verse count is a FLAG (a divergence to reconcile), not an error.
    """
    rep = TaggedReport(source=pt.id, language=pt.language)
    rep.parser_flags = list(pt.flags)
    rep.n_words = len(pt.words)
    verses: set[tuple[str, int, int]] = set()
    divergent: set[tuple[str, int, int]] = set()
    for w in pt.words:
        ref = w.ref
        if not _well_formed_ref(ref.book, ref.chapter, ref.verse_start):
            rep.bad_refs.append(
                f"{pt.id}: {ref.book} {ref.chapter}:{ref.verse_start} "
                f"(malformed reference)"
            )
        elif not _valid_ref(ref.book, ref.chapter, ref.verse_start):
            divergent.add((ref.book, ref.chapter, ref.verse_start))
        verses.add((ref.book, ref.chapter, ref.verse_start))
        if w.strongs:
            rep.n_words_with_strongs += 1
    rep.n_verses = len(verses)
    rep.n_books = len({v[0] for v in verses})
    if divergent:
        sample = ", ".join(
            f"{b} {c}:{v}" for b, c, v in sorted(divergent)[:10]
        )
        scheme = "Masoretic" if pt.language == "hebrew" else "NRSV"
        rep.masoretic_divergences = [
            f"{pt.id}: {len(divergent)} distinct verses use {scheme} numbering "
            f"that diverges from the English/KJV spine (expected — reconciled by "
            f"the P5 canonical versification map; e.g. {sample})"
        ]
    return rep


def validate_orphans(
    lexicons: dict[str, ParsedLexicon],
    tagged: dict[str, ParsedTaggedText],
) -> OrphanReport:
    """Every Strong's number REFERENCED must resolve to a lexicon entry.

    Resolution targets differ by language:
      * Hebrew (OSHB) refs a base Strong's H-number -> the Strong's-Hebrew dict.
      * Greek (TAGNT) refs a *disambiguated/extended* Strong's (e.g. G2424G) ->
        the TBESG lexicon, which is keyed by exactly that extended Strong's. A
        TAGNT word whose extended key is not in TBESG, AND whose base Strong's
        (letter stripped) is not in Strong's-Greek either, is a true orphan.
    """
    rep = OrphanReport()
    # Strong's-dictionary keys (base numbers; Greek + Hebrew).
    have_strongs: set[str] = set()
    for pl in lexicons.values():
        if pl.lexicon == "strongs":
            have_strongs |= {e.strongs for e in pl.entries if e.strongs}
    # TBESG keys (extended/disambiguated Greek Strong's) — the Greek resolution set.
    have_tbesg: set[str] = set()
    for pl in lexicons.values():
        if pl.lexicon == "tbesg":
            have_tbesg |= {e.strongs for e in pl.entries if e.strongs}

    # Referenced by the tagged text. Hebrew (OSHB) keys resolve against the
    # Strong's dictionaries; Greek (TAGNT) extended keys resolve against TBESG
    # (falling back to the base Strong's-Greek dictionary if the extended sense is
    # absent but the base number exists).
    orphan_tagged: set[str] = set()
    orphan_greek: set[str] = set()
    for pt in tagged.values():
        for w in pt.words:
            for key in w.strongs:
                if pt.language == "greek":
                    if key in have_tbesg:
                        continue
                    if _base_strongs(key) in have_strongs:
                        continue
                    orphan_greek.add(key)
                else:
                    if key not in have_strongs:
                        orphan_tagged.add(key)
    rep.from_tagged = sorted(orphan_tagged, key=_strongs_num)
    rep.from_greek = sorted(orphan_greek, key=_strongs_num)

    # Referenced by BDB's Strong's links.
    referenced_bdb: set[str] = set()
    for pl in lexicons.values():
        if pl.lexicon == "bdb":
            referenced_bdb |= {e.strongs for e in pl.entries if e.strongs}
    rep.from_bdb = sorted(referenced_bdb - have_strongs, key=_strongs_num)
    return rep


def _base_strongs(key: str) -> str:
    """Strip a trailing disambiguation letter from a Greek extended Strong's.

    ``G2424G`` -> ``G2424``; ``G976`` -> ``G976``.
    """
    m = re.fullmatch(r"(G\d+)[A-Za-z]?", key)
    return m.group(1) if m else key


def _strongs_num(s: str) -> tuple[int, str]:
    return (int(s[1:]), s) if len(s) > 1 and s[1:].isdigit() else (1 << 30, s)


def _valid_ref(book: str, chapter: int, verse: int) -> bool:
    """True when (book, chapter, verse) is in range for the English/KJV spine."""
    if book not in books.CANON:
        return False
    counts = books.VERSE_COUNTS.get(book)
    if not counts or not (1 <= chapter <= len(counts)):
        return False
    return 1 <= verse <= counts[chapter - 1]


def _well_formed_ref(book: str, chapter: int, verse: int) -> bool:
    """True when the ref is structurally sane (known book, positive ch/verse).

    Looser than :func:`_valid_ref`: it does NOT require the verse to be within the
    English count, because OSHB is Masoretic-numbered. Used to separate genuinely
    malformed refs (errors) from versification divergences (flags). A chapter that
    is one beyond the English count (e.g. Joel 4) is allowed — Hebrew chapter
    counts legitimately differ.
    """
    if book not in books.CANON:
        return False
    return chapter >= 1 and verse >= 1


def render_lexicon_report(
    lex_reports: dict[str, LexiconReport],
    *,
    tagged_reports: dict[str, TaggedReport] | None = None,
    orphans: OrphanReport | None = None,
    thayers_flag: str | None = None,
    sblgnt_flag: str | None = None,
) -> str:
    tagged_reports = tagged_reports or {}
    lines: list[str] = []
    lines.append("LampStand corpus — lexicon validation report (M1-P4)")
    lines.append("=" * 64)
    lines.append("")
    lines.append("CANDIDATE ONLY. Not ship-ready: the architect's 23-point "
                 "spot-check gates ship (CLAUDE.md §Human spot-check).")
    lines.append("")

    total_err = sum(r.error_total for r in lex_reports.values())
    total_flag = sum(r.flag_total for r in lex_reports.values())
    total_err += sum(r.error_total for r in tagged_reports.values())
    total_flag += sum(r.flag_total for r in tagged_reports.values())
    if orphans:
        total_err += orphans.error_total

    lines.append("ERROR / FLAG SUMMARY")
    lines.append("-" * 64)
    lines.append(f"  total errors: {total_err}")
    lines.append(f"  total flags:  {total_flag}")
    lines.append("")

    lines.append("LEXICON DICTIONARIES")
    lines.append("-" * 64)
    for lid in sorted(lex_reports):
        r = lex_reports[lid]
        lines.append(f"  {lid} [{r.language}]")
        lines.append(f"    entries={r.n_entries} linked-to-strongs={r.n_linked} "
                     f"distinct-strongs={r.distinct_strongs} "
                     f"with-definition={r.n_with_definition}")
        if r.expected_max:
            lines.append(f"    coverage span: 1..{r.expected_max}")
        for g in r.coverage_gaps:
            lines.append(f"    GAP: {g}")
        for d in r.duplicate_keys:
            lines.append(f"    ERROR: {d}")
        if r.empty_definitions:
            lines.append(f"    ERROR: {len(r.empty_definitions)} entries with no "
                         f"definition (first: {r.empty_definitions[0]})")
        for ln in r.long_definitions[:10]:
            lines.append(f"    FLAG (long def): {ln}")
        for f in r.parser_flags[:8]:
            lines.append(f"    flag: {f}")
        if len(r.parser_flags) > 8:
            lines.append(f"    flag: ... (+{len(r.parser_flags) - 8} more)")
        lines.append("")

    if orphans:
        lines.append("ORPHAN STRONG'S (referenced, no lexicon entry)")
        lines.append("-" * 64)
        lines.append(f"  from OSHB Hebrew text (-> Strong's-Hebrew): "
                     f"{len(orphans.from_tagged)}")
        if orphans.from_tagged:
            lines.append("    " + ", ".join(orphans.from_tagged[:40]))
        lines.append(f"  from TAGNT Greek text (-> TBESG/Strong's-Greek): "
                     f"{len(orphans.from_greek)}")
        if orphans.from_greek:
            lines.append("    " + ", ".join(orphans.from_greek[:40]))
        lines.append(f"  from BDB links (-> Strong's-Hebrew): "
                     f"{len(orphans.from_bdb)}")
        if orphans.from_bdb:
            lines.append("    " + ", ".join(orphans.from_bdb[:40]))
        lines.append("")

    if tagged_reports:
        lines.append("TAGGED ORIGINAL-LANGUAGE TEXT (P4b)")
        lines.append("-" * 64)
        for sid in sorted(tagged_reports):
            r = tagged_reports[sid]
            lines.append(f"  {sid} [{r.language}]")
            lines.append(f"    words={r.n_words} verses={r.n_verses} "
                         f"books={r.n_books} words-with-strongs="
                         f"{r.n_words_with_strongs}")
            if r.bad_refs:
                lines.append(f"    ERROR: {len(r.bad_refs)} malformed refs "
                             f"(first: {r.bad_refs[0]})")
            for d in r.masoretic_divergences:
                lines.append(f"    FLAG: {d}")
            for f in r.parser_flags[:8]:
                lines.append(f"    flag: {f}")
        lines.append("")

    lines.append("FLAGGED FOR HUMAN REVIEW")
    lines.append("-" * 64)
    n = 1
    if thayers_flag:
        lines.append(f"  {n}. {thayers_flag}")
        n += 1
    if sblgnt_flag:
        lines.append(f"  {n}. {sblgnt_flag}")
        n += 1
    lines.append("")
    return "\n".join(lines) + "\n"
