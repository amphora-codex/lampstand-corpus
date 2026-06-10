"""Validation pass over parsed Bibles.

Emits a structured report flagging, per translation:
  * missing books (vs. the 66-book canon)
  * missing/extra chapters (vs. expected chapter counts)
  * verse-count deviations per chapter (vs. KJV/traditional versification) — these
    are *flags*, not errors: BSB/ASV/WEB have known minor versification differences
  * empty verses (parsed to no text — often a textual-variant omission, e.g.
    ASV John 5:4)
  * statistical anomalies (e.g. a chapter with far fewer verses than reference)

Nothing is "fixed" silently. Residuals a human must adjudicate are listed
verbatim. The pipeline never marks output ship-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import books
from .usfm import ParsedBook


@dataclass
class TranslationReport:
    translation: str
    n_books: int = 0
    n_chapters: int = 0
    n_verses: int = 0
    n_red_letter_verses: int = 0
    n_bridges: int = 0
    missing_books: list[str] = field(default_factory=list)
    extra_books: list[str] = field(default_factory=list)
    chapter_count_mismatches: list[str] = field(default_factory=list)
    verse_count_deviations: list[str] = field(default_factory=list)
    empty_verses: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    parser_warnings: list[str] = field(default_factory=list)

    @property
    def error_total(self) -> int:
        """Hard problems (missing/extra books, missing chapters)."""
        return len(self.missing_books) + len(self.extra_books) + len(
            [m for m in self.chapter_count_mismatches])

    @property
    def flag_total(self) -> int:
        """Soft items for human review (deviations, empties, anomalies)."""
        return (len(self.verse_count_deviations) + len(self.empty_verses)
                + len(self.anomalies))


def validate_translation(tid: str, parsed: dict[str, ParsedBook]) -> TranslationReport:
    rep = TranslationReport(translation=tid)

    present = set(parsed)
    rep.missing_books = [b for b in books.ORDER if b not in present]
    rep.extra_books = sorted(b for b in present if b not in books.CANON)
    rep.n_books = len([b for b in books.ORDER if b in present])

    for book_id in books.ORDER:
        pb = parsed.get(book_id)
        if pb is None:
            continue
        rep.parser_warnings.extend(pb.warnings)

        chapters = sorted({v.chapter for v in pb.verses})
        rep.n_chapters += len(chapters)
        expected_ch = books.CHAPTER_COUNTS[book_id]
        if len(chapters) != expected_ch:
            rep.chapter_count_mismatches.append(
                f"{book_id}: {len(chapters)} chapters, expected {expected_ch}"
            )
        # missing chapter numbers in the 1..expected range
        missing_ch = [c for c in range(1, expected_ch + 1) if c not in chapters]
        if missing_ch:
            rep.chapter_count_mismatches.append(
                f"{book_id}: missing chapters {missing_ch}"
            )

        # verse bookkeeping
        by_chapter: dict[int, list] = {}
        for v in pb.verses:
            rep.n_verses += 1
            if v.has_red_letter:
                rep.n_red_letter_verses += 1
            if v.is_bridge:
                rep.n_bridges += 1
            if not v.text:
                rep.empty_verses.append(f"{book_id} {v.chapter}:{v.verse_start}")
            by_chapter.setdefault(v.chapter, []).append(v)

        # verse-count deviation vs. reference versification
        ref = books.VERSE_COUNTS.get(book_id, [])
        for ci, expected_vc in enumerate(ref, start=1):
            verses_here = by_chapter.get(ci, [])
            # Highest verse number addressed (accounting for bridges).
            max_v = max((v.verse_end for v in verses_here), default=0)
            if max_v != expected_vc:
                rep.verse_count_deviations.append(
                    f"{book_id} {ci}: max verse {max_v}, reference expects {expected_vc}"
                )
            # anomaly: chapter with no verses at all
            if not verses_here:
                rep.anomalies.append(f"{book_id} {ci}: chapter has no verses")

    return rep


def validate_all(parsed: dict[str, dict[str, ParsedBook]]) -> dict[str, TranslationReport]:
    return {tid: validate_translation(tid, parsed[tid]) for tid in sorted(parsed)}


def render_report(reports: dict[str, TranslationReport]) -> str:
    """Render a human-readable text report (written to reports/)."""
    lines: list[str] = []
    lines.append("LampStand corpus — P1 Bible validation report")
    lines.append("Canon: 66-book Protestant. Reference versification: KJV/traditional.")
    lines.append("Note: deviations and empty verses are FLAGS for human review,")
    lines.append("not silent fixes. Pipeline never marks a corpus ship-ready.")
    lines.append("=" * 72)

    for tid in sorted(reports):
        r = reports[tid]
        lines.append("")
        lines.append(f"## {tid.upper()}")
        lines.append(f"  books={r.n_books}/66  chapters={r.n_chapters}  "
                     f"verses={r.n_verses}  red-letter verses={r.n_red_letter_verses}  "
                     f"bridges={r.n_bridges}")
        lines.append(f"  errors={r.error_total}  flags={r.flag_total}")

        def block(title: str, items: list[str], cap: int = 50) -> None:
            if not items:
                return
            lines.append(f"  {title} ({len(items)}):")
            for it in items[:cap]:
                lines.append(f"    - {it}")
            if len(items) > cap:
                lines.append(f"    ... and {len(items) - cap} more")

        block("MISSING BOOKS", r.missing_books)
        block("EXTRA / NON-CANON BOOKS", r.extra_books)
        block("CHAPTER MISMATCHES", r.chapter_count_mismatches)
        block("EMPTY VERSES (textual variants?)", r.empty_verses)
        block("VERSE-COUNT DEVIATIONS (vs reference)", r.verse_count_deviations)
        block("STRUCTURAL ANOMALIES", r.anomalies)
        block("PARSER WARNINGS", r.parser_warnings)

    lines.append("")
    lines.append("=" * 72)
    lines.append("END. Human spot-check still required before ship (architect's 23-point).")
    return "\n".join(lines) + "\n"
