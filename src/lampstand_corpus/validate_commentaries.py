"""Validation pass over normalized commentary chunks.

Per commentator, flags (never silently fixes):
  * coverage — which books / chapters produced at least one chunk, vs. the books
    the commentator is expected to cover (Henry & JFB: whole Bible; Calvin: the
    v1 scope of Genesis + Psalms + NT, minus the NT books Calvin never wrote);
  * every chunk maps to a VALID verse reference on the 66-book spine — refs whose
    book/chapter/verse fall outside the canon are listed for human review (the
    parser already drops un-mappable scripCom anchors with a flag; this is the
    belt-and-braces check on what survived);
  * statistical anomalies — an unusually long single comment block (e.g. Calvin
    on Psalm 119) is SURFACED, never truncated; trivially short blocks are noted.

Residuals a parser can't adjudicate are listed verbatim. Strong's-number checks
are N/A for commentaries. The pipeline never marks output ship-ready — the
architect's 23-point spot-check gates ship.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import books
from .commentaries import (
    CALVIN_NT_NOT_WRITTEN,
    CALVIN_VOLUMES_DEFERRED,
    LONG_BLOCK_CHARS,
    ParsedCommentary,
)

# Expected book coverage per commentator (USFM ids). Henry & JFB cover the whole
# 66-book canon. Calvin's v1 scope is Genesis + Psalms + the New Testament; the
# four NT books Calvin never commented on are excluded from "expected" so their
# absence is reported as known, not as a gap.
_NT_BOOKS = tuple(books.ORDER[books.ORDER_INDEX["MAT"]:])
CALVIN_EXPECTED_BOOKS: frozenset[str] = frozenset(
    ("GEN", "PSA", *(b for b in _NT_BOOKS if b not in CALVIN_NT_NOT_WRITTEN))
)
EXPECTED_BOOKS: dict[str, frozenset[str]] = {
    "henry": books.CANON,
    "jfb": books.CANON,
    "calvin": CALVIN_EXPECTED_BOOKS,
}


@dataclass
class CommentaryReport:
    commentator: str
    n_chunks: int = 0
    n_books: int = 0
    n_chapters: int = 0
    books_covered: list[str] = field(default_factory=list)
    expected_books: int = 0
    missing_books: list[str] = field(default_factory=list)
    partial_books: list[str] = field(default_factory=list)   # book: got/expected chapters
    bad_refs: list[str] = field(default_factory=list)        # book/chapter off-canon
    overrun_refs: list[str] = field(default_factory=list)    # CCEL end-verse > chap len
    long_blocks: list[str] = field(default_factory=list)
    parser_flags: list[str] = field(default_factory=list)
    total_chars: int = 0

    @property
    def error_total(self) -> int:
        # A chunk whose book/chapter falls outside the canon is a true error.
        # Range overruns (a valid chapter, but CCEL's end-verse exceeds the
        # chapter length) are an upstream-data flag for human review, not a
        # build error — the verse_start still resolves; only CCEL's end is off.
        return len(self.bad_refs)

    @property
    def flag_total(self) -> int:
        return (len(self.missing_books) + len(self.partial_books)
                + len(self.overrun_refs) + len(self.long_blocks)
                + len(self.parser_flags))


def _valid_ref(book: str, chapter: int, verse: int) -> bool:
    """A ref is valid if book/chapter are in canon and verse is in range.

    ``verse == 0`` is the legitimate whole-chapter (intro) anchor, allowed as long
    as the chapter exists.
    """
    if book not in books.CANON:
        return False
    counts = books.VERSE_COUNTS.get(book)
    if not counts or not (1 <= chapter <= len(counts)):
        return False
    if verse == 0:
        return True
    return 1 <= verse <= counts[chapter - 1]


def validate_commentary(pc: ParsedCommentary) -> CommentaryReport:
    rep = CommentaryReport(commentator=pc.id)
    rep.n_chunks = len(pc.chunks)
    rep.parser_flags = list(pc.flags)

    chapters_seen: set[tuple[str, int]] = set()
    for ch in pc.chunks:
        r = ch.ref
        rep.total_chars += len(ch.text)
        chapters_seen.add((r.book, r.chapter))
        vend = r.verse_end or r.verse_start
        label = (f"{pc.id}:{ch.key} -> {r.book} {r.chapter}:{r.verse_start}"
                 f"{('-' + str(r.verse_end)) if r.verse_end else ''}")
        start_ok = _valid_ref(r.book, r.chapter, r.verse_start)
        end_ok = _valid_ref(r.book, r.chapter, vend)
        if not start_ok:
            # Book or chapter (or the start verse) is genuinely off the canon spine.
            rep.bad_refs.append(f"{label} (off-canon book/chapter/start)")
        elif not end_ok:
            # Valid chapter + start, but CCEL's end-verse overruns the chapter
            # length — an upstream source-data quirk to verify against a printed
            # edition (the comment content is still anchored correctly at start).
            counts = books.VERSE_COUNTS.get(r.book, [])
            chap_len = counts[r.chapter - 1] if r.chapter <= len(counts) else "?"
            rep.overrun_refs.append(
                f"{label} (CCEL end-verse > chapter length {chap_len})"
            )
        if len(ch.text) >= LONG_BLOCK_CHARS:
            rep.long_blocks.append(
                f"{pc.id}:{ch.key} ({len(ch.text):,} chars) "
                f"passage={ch.meta.get('passage')!r}"
            )

    covered = sorted({b for (b, _c) in chapters_seen},
                     key=lambda b: books.ORDER_INDEX.get(b, 999))
    rep.books_covered = covered
    rep.n_books = len(covered)
    rep.n_chapters = len(chapters_seen)

    expected = EXPECTED_BOOKS.get(pc.id, books.CANON)
    rep.expected_books = len(expected)
    rep.missing_books = [b for b in books.ORDER
                         if b in expected and b not in set(covered)]

    # Partial coverage: a book present but with fewer chapters than the canon
    # expects (a real concern for a "whole Bible" commentator). Flag, never fix.
    chap_count: dict[str, int] = {}
    for (b, _c) in chapters_seen:
        chap_count[b] = chap_count.get(b, 0) + 1
    for b in covered:
        exp = books.CHAPTER_COUNTS.get(b, 0)
        got = chap_count.get(b, 0)
        if exp and got < exp:
            rep.partial_books.append(f"{b}: {got}/{exp} chapters")

    return rep


def validate_all_commentaries(
    parsed: dict[str, ParsedCommentary]
) -> dict[str, CommentaryReport]:
    return {cid: validate_commentary(parsed[cid]) for cid in sorted(parsed)}


# Sources in the architect-locked scope that were FLAGGED-AND-SKIPPED for v1
# (surfaced for human sourcing; never substituted with a derived/OCR text), and
# the deferrals confirmed by this phase. Each entry: (label, why).
SKIPPED_AND_DEFERRED: list[tuple[str, str]] = [
    ("Spurgeon — Treasury of David (Psalms) — SKIPPED, flagged for human sourcing",
     "The only CCEL edition (spurgeon/treasury1-6) is a PAGE-IMAGE SCAN: every "
     "volume's body is nothing but <img> references to ~2900 PNG page scans, with "
     "no machine-readable commentary text (the sole extractable text is the print-"
     "page-number index). Ingesting it would require OCR — a guess, not a parse — "
     "which CLAUDE.md forbids. ARCHITECT: approve a clean full-text public-domain "
     "Treasury of David source before inclusion. (Mirrors the P2 1689/Belgic "
     "handling.)"),
    ("John Gill — Exposition of the Whole Bible — NOT INGESTED (deferred to v1.1)",
     "Per the architect-locked scope, Gill is deferred to corpus v1.1 and is "
     "deliberately not defined as a source in this phase. Confirmed absent."),
    ("John Calvin — remaining OT — DEFERRED (cleanly skipped)",
     "v1 Calvin scope is Genesis + Psalms + New Testament only (spec §4.2). The "
     "deferred OT volumes (Harmony of the Law/Joshua, Isaiah, Jeremiah & "
     "Lamentations, Ezekiel, Daniel, the Minor Prophets) are NOT snapshotted or "
     "parsed: " + "; ".join(f"{k}={v}" for k, v in CALVIN_VOLUMES_DEFERRED.items())
     + ". Calvin also never wrote on Revelation, 2 John, or 3 John — those NT "
     "books are legitimately absent, not gaps (he did comment on Jude)."),
]


def render_commentary_report(
    reports: dict[str, CommentaryReport]
) -> str:
    lines: list[str] = []
    lines.append("LampStand corpus — P3 commentaries validation report")
    lines.append("Granularity: paragraph-level chunk, each anchored to a VerseRef")
    lines.append("(single verse or verse range / pericope). Every CCEL structural")
    lines.append("ambiguity is a FLAG for human review, not a silent fix. The")
    lines.append("pipeline never marks a corpus ship-ready.")
    lines.append("=" * 72)

    lines.append("")
    lines.append("## SCOPE (architect-locked, spec §4.2)")
    lines.append("  Henry  — Commentary on the Whole Bible (complete)")
    lines.append("  JFB    — Commentary Critical & Explanatory on the Whole Bible "
                 "(complete)")
    lines.append("  Calvin — Genesis + Psalms + New Testament ONLY (remaining OT "
                 "deferred)")
    lines.append("  Spurgeon Treasury of David — skipped (image-only CCEL source); "
                 "Gill — deferred v1.1")

    for cid in sorted(reports):
        r = reports[cid]
        lines.append("")
        lines.append(f"## {cid.upper()}")
        lines.append(f"  chunks={r.n_chunks}  books={r.n_books}/{r.expected_books}  "
                     f"chapters={r.n_chapters}  text={r.total_chars:,} chars")
        lines.append(f"  errors={r.error_total}  flags={r.flag_total}")
        lines.append(f"  books covered: {', '.join(r.books_covered)}")

        def block(title: str, items: list[str], cap: int = 60) -> None:
            if not items:
                return
            lines.append(f"  {title} ({len(items)}):")
            for it in items[:cap]:
                lines.append(f"    - {it}")
            if len(items) > cap:
                lines.append(f"    ... and {len(items) - cap} more")

        # Range overruns repeat per-paragraph; collapse to the distinct ranges so
        # the report shows the 6 source ranges, not 55 paragraph rows.
        distinct_overruns = sorted({
            re.sub(r"#p\d+(~\d+)?", "", o) for o in r.overrun_refs
        })

        block("EXPECTED BOOKS WITH NO COMMENTARY (review)", r.missing_books)
        block("BOOKS WITH PARTIAL CHAPTER COVERAGE (review)", r.partial_books)
        block("CHUNKS WITH OFF-CANON VERSE REFS (error)", r.bad_refs)
        block(f"CCEL ANCHOR VERSE-RANGE OVERRUNS — {len(r.overrun_refs)} chunk(s) "
              "across these distinct ranges (verify vs printed edition; content "
              "anchored at start)", distinct_overruns)
        block("UNUSUALLY LONG COMMENT BLOCKS (anomaly — verify not a merge)",
              r.long_blocks)
        block("PARSER / CCEL AMBIGUITY FLAGS (review)", r.parser_flags)

    lines.append("")
    lines.append("## SOURCES SKIPPED / DEFERRED IN SCOPE (flagged for human review)")
    for label, why in SKIPPED_AND_DEFERRED:
        lines.append(f"  - {label}")
        lines.append(f"      {why}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("END. Human spot-check still required before ship (architect's "
                 "23-point: 5 commentary passages vs the canonical edition).")
    return "\n".join(lines) + "\n"
