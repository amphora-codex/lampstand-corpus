"""Validation pass over normalized cross-references (CLAUDE.md pipeline rules).

The report FLAGS (never silently drops):
  * Cross-references whose **source verse** does not resolve to a real verse on the
    canonical (KJV) spine.
  * Cross-references whose **target** (either endpoint of a range) does not resolve.
  * Structural ambiguity surfaced for human review:
      - Target ranges that cross a *book* boundary (e.g. Lev.27.34-Num.1.1) — rare
        and worth a human eye to confirm they aren't a digitization artifact.
      - Target ranges whose end precedes its start in canonical order (inverted).
      - Negative-vote edges (community-downvoted; the sign is preserved, the count
        is surfaced so a human can decide a vote floor at app-render time).
      - Any line OpenBible emitted that we could not parse at all (``unparsed``).
  * Statistical context: total edges, distinct source verses, range share, vote
    distribution.

The pipeline never marks output ship-ready — the architect's 23-point spot-check
gates ship (CLAUDE.md §Human spot-check).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import books
from .crossrefs import (
    CROSSREFS_ATTRIBUTION,
    CROSSREFS_LICENSE,
    CROSSREFS_URL,
    CrossRef,
    ParsedCrossRefs,
    point_resolves,
)

# How many example flagged refs to print per category before truncating.
EXAMPLE_LIMIT = 25


@dataclass
class CrossRefReport:
    n_refs: int = 0
    n_distinct_sources: int = 0
    n_ranges: int = 0
    n_single: int = 0
    votes_min: int = 0
    votes_max: int = 0
    n_negative_votes: int = 0
    n_zero_votes: int = 0

    # Errors (CLAUDE.md: every non-resolving ref must be flagged, not dropped).
    nonresolving_source: list[str] = field(default_factory=list)
    nonresolving_target: list[str] = field(default_factory=list)

    # Structural ambiguity surfaced for the human.
    cross_book_ranges: list[str] = field(default_factory=list)
    inverted_ranges: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)

    @property
    def n_nonresolving_source(self) -> int:
        return len(self.nonresolving_source)

    @property
    def n_nonresolving_target(self) -> int:
        return len(self.nonresolving_target)

    @property
    def error_total(self) -> int:
        # A non-resolving source OR target is the CLAUDE.md error class to surface.
        return (len(self.nonresolving_source)
                + len(self.nonresolving_target)
                + len(self.unparsed))

    @property
    def flag_total(self) -> int:
        return (len(self.cross_book_ranges)
                + len(self.inverted_ranges)
                + (1 if self.n_negative_votes else 0))


def _src_str(cr: CrossRef) -> str:
    s = cr.source
    return f"{s.book} {s.chapter}:{s.verse}"


def _tgt_str(cr: CrossRef) -> str:
    ts, te = cr.target_start, cr.target_end
    if not cr.is_range:
        return f"{ts.book} {ts.chapter}:{ts.verse}"
    return (f"{ts.book} {ts.chapter}:{ts.verse}-"
            f"{te.book} {te.chapter}:{te.verse}")


def _canon_pos(book: str, chapter: int, verse: int) -> tuple[int, int, int]:
    return (books.ORDER_INDEX.get(book, 1 << 30), chapter, verse)


def validate_crossrefs(parsed: ParsedCrossRefs) -> CrossRefReport:
    rep = CrossRefReport()
    rep.unparsed = list(parsed.unparsed)
    rep.n_refs = len(parsed.refs)

    sources: set[tuple[str, int, int]] = set()
    if parsed.refs:
        rep.votes_min = min(cr.votes for cr in parsed.refs)
        rep.votes_max = max(cr.votes for cr in parsed.refs)

    for cr in parsed.refs:
        sources.add(cr.source.as_tuple())
        if cr.is_range:
            rep.n_ranges += 1
        else:
            rep.n_single += 1
        if cr.votes < 0:
            rep.n_negative_votes += 1
        elif cr.votes == 0:
            rep.n_zero_votes += 1

        # CLAUDE.md: flag every ref whose source OR target does not resolve.
        if not point_resolves(cr.source):
            rep.nonresolving_source.append(
                f"{_src_str(cr)} -> {_tgt_str(cr)} (votes={cr.votes})"
            )
        ts, te = cr.target_start, cr.target_end
        if not (point_resolves(ts) and point_resolves(te)):
            rep.nonresolving_target.append(
                f"{_src_str(cr)} -> {_tgt_str(cr)} (votes={cr.votes})"
            )

        # Structural ambiguity.
        if cr.is_range and ts.book != te.book:
            rep.cross_book_ranges.append(f"{_src_str(cr)} -> {_tgt_str(cr)}")
        if cr.is_range and _canon_pos(*te.as_tuple()) < _canon_pos(*ts.as_tuple()):
            rep.inverted_ranges.append(f"{_src_str(cr)} -> {_tgt_str(cr)}")

    rep.n_distinct_sources = len(sources)
    return rep


def render_crossref_report(rep: CrossRefReport) -> str:
    lines: list[str] = []
    lines.append("LampStand corpus — cross-reference validation report (M1-P5)")
    lines.append("=" * 64)
    lines.append("")
    lines.append("CANDIDATE ONLY. Not ship-ready: the architect's 23-point "
                 "spot-check gates ship (CLAUDE.md §Human spot-check).")
    lines.append("")

    lines.append("SOURCE")
    lines.append("-" * 64)
    lines.append("  dataset:     Treasury of Scripture Knowledge (OpenBible.info)")
    lines.append(f"  url:         {CROSSREFS_URL}")
    lines.append(f"  license:     {CROSSREFS_LICENSE}")
    lines.append(f"  attribution: {CROSSREFS_ATTRIBUTION}")
    lines.append("")

    lines.append("ERROR / FLAG SUMMARY")
    lines.append("-" * 64)
    lines.append(f"  total errors: {rep.error_total}")
    lines.append(f"  total flags:  {rep.flag_total}")
    lines.append("")

    lines.append("STATISTICS")
    lines.append("-" * 64)
    lines.append(f"  cross-references:      {rep.n_refs}")
    lines.append(f"  distinct source verses:{rep.n_distinct_sources:>8}")
    lines.append(f"  single-verse targets:  {rep.n_single}")
    lines.append(f"  range targets:         {rep.n_ranges}")
    lines.append(f"  votes range:           {rep.votes_min} .. {rep.votes_max}")
    lines.append(f"  negative-vote edges:   {rep.n_negative_votes}")
    lines.append(f"  zero-vote edges:       {rep.n_zero_votes}")
    lines.append("")

    lines.append("NON-RESOLVING REFERENCES (flagged, NOT dropped)")
    lines.append("-" * 64)
    lines.append(f"  source verse does not resolve: {rep.n_nonresolving_source}")
    for s in rep.nonresolving_source[:EXAMPLE_LIMIT]:
        lines.append(f"    {s}")
    if rep.n_nonresolving_source > EXAMPLE_LIMIT:
        lines.append(f"    ... (+{rep.n_nonresolving_source - EXAMPLE_LIMIT} more)")
    lines.append(f"  target verse/range does not resolve: "
                 f"{rep.n_nonresolving_target}")
    for t in rep.nonresolving_target[:EXAMPLE_LIMIT]:
        lines.append(f"    {t}")
    if rep.n_nonresolving_target > EXAMPLE_LIMIT:
        lines.append(f"    ... (+{rep.n_nonresolving_target - EXAMPLE_LIMIT} more)")
    lines.append("")

    lines.append("STRUCTURAL AMBIGUITY (for human review)")
    lines.append("-" * 64)
    lines.append(f"  target ranges crossing a book boundary: "
                 f"{len(rep.cross_book_ranges)}")
    for r in rep.cross_book_ranges[:EXAMPLE_LIMIT]:
        lines.append(f"    {r}")
    lines.append(f"  inverted ranges (end before start): "
                 f"{len(rep.inverted_ranges)}")
    for r in rep.inverted_ranges[:EXAMPLE_LIMIT]:
        lines.append(f"    {r}")
    if rep.unparsed:
        lines.append(f"  unparsed lines: {len(rep.unparsed)}")
        for u in rep.unparsed[:EXAMPLE_LIMIT]:
            lines.append(f"    {u}")
    lines.append("")

    lines.append("FLAGGED FOR HUMAN REVIEW")
    lines.append("-" * 64)
    n = 1
    if rep.n_negative_votes:
        lines.append(
            f"  {n}. {rep.n_negative_votes} cross-references carry NEGATIVE votes "
            f"(min {rep.votes_min}). The sign is preserved verbatim; decide whether "
            f"the app should apply a vote floor at render time (these are "
            f"community-downvoted, not deleted)."
        )
        n += 1
    if rep.cross_book_ranges:
        lines.append(
            f"  {n}. {len(rep.cross_book_ranges)} target ranges cross a book "
            f"boundary (e.g. {rep.cross_book_ranges[0]}). Confirm these are genuine "
            f"TSK spans and not a digitization artifact."
        )
        n += 1
    if rep.nonresolving_source or rep.nonresolving_target:
        lines.append(
            f"  {n}. {rep.n_nonresolving_source} source / "
            f"{rep.n_nonresolving_target} target refs do not resolve on the "
            f"canonical KJV spine. They are kept and flagged (never dropped); "
            f"verify whether any indicate a versification mismatch vs. a true bad "
            f"reference."
        )
        n += 1
    if n == 1:
        lines.append("  (none)")
    lines.append("")
    return "\n".join(lines) + "\n"
