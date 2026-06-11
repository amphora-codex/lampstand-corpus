"""Treasury of Scripture Knowledge cross-references (OpenBible.info dataset).

Source of record (CLAUDE.md §Sources of record): the OpenBible.info
``cross_references.txt`` dataset — a vote-ranked, machine-readable digitization of
the Treasury of Scripture Knowledge cross-reference network.

  * URL:     https://a.openbible.info/data/cross-references.zip
  * License: **CC-BY** (Creative Commons Attribution). Attribution required:
             "Cross-reference data courtesy of www.openbible.info (CC-BY)."
             The header line of the data file carries the same notice + a date.

Format
------
The file is tab-separated with a single header row::

    From Verse<TAB>To Verse<TAB>Votes<TAB>#www.openbible.info CC-BY <date>
    Gen.1.1<TAB>Eccl.12.1<TAB>26
    Gen.1.1<TAB>Eph.6.23-Eph.6.24<TAB>2

* ``From Verse`` is always a single OSIS verse (``Book.Chapter.Verse``).
* ``To Verse`` is an OSIS verse OR an OSIS verse range (``A-B``). A range's two
  endpoints may cross a chapter boundary (e.g. ``Gen.11.32-Gen.12.1``) and, in 18
  cases, a *book* boundary (e.g. ``Lev.27.34-Num.1.1``) — both are handled by
  storing the full start AND end coordinates of the target.
* ``Votes`` is a signed relevance weight. Range observed: **-86 .. +1278**. The
  sign is meaningful (negative = community-downvoted) and is preserved verbatim.

Normalization
-------------
OpenBible uses standard English/KJV versification — the SAME spine LampStand
adopted as canonical (see ``versification.py``, "Option A"). So OSIS refs map onto
our spine by book-code translation alone; no renumbering is needed for the spine
itself. Each ref is normalized to the canonical ``(USFM book, chapter, verse)``
coordinate. A ref that does not resolve to a real verse on that spine
(``books.VERSE_COUNTS``) is FLAGGED, never dropped (CLAUDE.md validation rule).

This module only *parses + normalizes*; ``build_crossrefs.py`` writes the SQLite,
``validate_crossrefs.py`` produces the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import books
from .schema import Provenance
from .sources import SOURCES_DIR

CROSSREFS_DIR = SOURCES_DIR / "crossrefs"

# OpenBible.info cross-reference dataset.
CROSSREFS_URL = "https://a.openbible.info/data/cross-references.zip"
CROSSREFS_ZIP = "cross-references.zip"
CROSSREFS_TXT = "cross_references.txt"
CROSSREFS_NAME = "Treasury of Scripture Knowledge cross-references (OpenBible.info)"
CROSSREFS_LICENSE = "CC-BY 4.0 (www.openbible.info)"
CROSSREFS_ATTRIBUTION = (
    "Cross-reference data courtesy of www.openbible.info, used under CC-BY."
)

# --- OSIS book code -> canonical USFM id (the 66-book Protestant canon) --------
# OpenBible uses OSIS abbreviations; our spine is the USFM ids in books.ORDER.
OSIS_TO_USFM: dict[str, str] = {
    "Gen": "GEN", "Exod": "EXO", "Lev": "LEV", "Num": "NUM", "Deut": "DEU",
    "Josh": "JOS", "Judg": "JDG", "Ruth": "RUT", "1Sam": "1SA", "2Sam": "2SA",
    "1Kgs": "1KI", "2Kgs": "2KI", "1Chr": "1CH", "2Chr": "2CH", "Ezra": "EZR",
    "Neh": "NEH", "Esth": "EST", "Job": "JOB", "Ps": "PSA", "Prov": "PRO",
    "Eccl": "ECC", "Song": "SNG", "Isa": "ISA", "Jer": "JER", "Lam": "LAM",
    "Ezek": "EZK", "Dan": "DAN", "Hos": "HOS", "Joel": "JOL", "Amos": "AMO",
    "Obad": "OBA", "Jonah": "JON", "Mic": "MIC", "Nah": "NAM", "Hab": "HAB",
    "Zeph": "ZEP", "Hag": "HAG", "Zech": "ZEC", "Mal": "MAL", "Matt": "MAT",
    "Mark": "MRK", "Luke": "LUK", "John": "JHN", "Acts": "ACT", "Rom": "ROM",
    "1Cor": "1CO", "2Cor": "2CO", "Gal": "GAL", "Eph": "EPH", "Phil": "PHP",
    "Col": "COL", "1Thess": "1TH", "2Thess": "2TH", "1Tim": "1TI", "2Tim": "2TI",
    "Titus": "TIT", "Phlm": "PHM", "Heb": "HEB", "Jas": "JAS", "1Pet": "1PE",
    "2Pet": "2PE", "1John": "1JN", "2John": "2JN", "3John": "3JN", "Jude": "JUD",
    "Rev": "REV",
}

_OSIS_RE = re.compile(r"^([0-9A-Za-z]+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class CanonicalPoint:
    """One endpoint of a reference, normalized to the canonical USFM spine."""

    book: str          # USFM id
    chapter: int
    verse: int

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.book, self.chapter, self.verse)


@dataclass(frozen=True)
class CrossRef:
    """One normalized cross-reference: source verse -> target (verse or range)."""

    source: CanonicalPoint
    target_start: CanonicalPoint
    target_end: CanonicalPoint   # == target_start for single-verse targets
    votes: int
    rank: int                    # 1-based rank of this target within its source verse

    @property
    def is_range(self) -> bool:
        return self.target_start.as_tuple() != self.target_end.as_tuple()


@dataclass
class ParsedCrossRefs:
    """All parsed cross-references plus parser-level flags + provenance."""

    refs: list[CrossRef] = field(default_factory=list)
    # OSIS tokens we could not structurally parse (kept verbatim for the report).
    unparsed: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    provenance: Provenance | None = None
    header: str = ""


def _parse_osis_point(token: str) -> CanonicalPoint | None:
    """Parse a single ``Book.Chapter.Verse`` OSIS token to a CanonicalPoint.

    Returns None when the token is structurally malformed or the book code is not
    one of the 66 canonical books (the caller records it as ``unparsed``).
    """
    m = _OSIS_RE.match(token)
    if not m:
        return None
    osis_book, chap, verse = m.group(1), int(m.group(2)), int(m.group(3))
    usfm = OSIS_TO_USFM.get(osis_book)
    if usfm is None:
        return None
    return CanonicalPoint(book=usfm, chapter=chap, verse=verse)


def _parse_target(token: str) -> tuple[CanonicalPoint, CanonicalPoint] | None:
    """Parse a target token — a single OSIS verse or an ``A-B`` OSIS range.

    Returns (start, end). For a single verse start == end. Returns None if either
    endpoint is unparseable.
    """
    if "-" in token:
        a, b = token.split("-", 1)
        start = _parse_osis_point(a)
        end = _parse_osis_point(b)
        if start is None or end is None:
            return None
        return start, end
    p = _parse_osis_point(token)
    return (p, p) if p is not None else None


def parse_crossrefs(text: str, provenance: Provenance) -> ParsedCrossRefs:
    """Parse the OpenBible ``cross_references.txt`` body into normalized refs.

    The first line is the header (``From Verse\\tTo Verse\\tVotes\\t#...``) and is
    captured for provenance, not treated as data. Each subsequent line is one
    cross-reference. Rank is assigned per source verse in *file order* (the file
    is already vote-sorted within each source verse by OpenBible), so the rank is
    deterministic and stable for a given snapshot.
    """
    result = ParsedCrossRefs(provenance=provenance)
    lines = text.splitlines()
    if not lines:
        result.flags.append("empty cross-reference file")
        return result

    result.header = lines[0]
    rank_by_source: dict[tuple[str, int, int], int] = {}

    for lineno, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) < 3:
            result.unparsed.append(f"line {lineno}: {raw!r} (too few columns)")
            continue
        from_tok, to_tok, vote_tok = parts[0], parts[1], parts[2]

        try:
            votes = int(vote_tok)
        except ValueError:
            result.unparsed.append(
                f"line {lineno}: non-integer votes {vote_tok!r}"
            )
            continue

        source = _parse_osis_point(from_tok)
        if source is None:
            result.unparsed.append(
                f"line {lineno}: unparseable source ref {from_tok!r}"
            )
            continue

        target = _parse_target(to_tok)
        if target is None:
            result.unparsed.append(
                f"line {lineno}: unparseable target ref {to_tok!r}"
            )
            continue
        tstart, tend = target

        key = source.as_tuple()
        rank = rank_by_source.get(key, 0) + 1
        rank_by_source[key] = rank

        result.refs.append(
            CrossRef(
                source=source,
                target_start=tstart,
                target_end=tend,
                votes=votes,
                rank=rank,
            )
        )
    return result


def point_resolves(point: CanonicalPoint) -> bool:
    """True iff ``point`` is a real verse on the canonical (KJV) spine.

    Uses ``books.VERSE_COUNTS`` — the same spine the rest of the corpus anchors
    to. OpenBible's TSK uses standard English/KJV versification, so a non-resolving
    point is a genuine out-of-range reference to surface, not a versification skew.
    """
    counts = books.VERSE_COUNTS.get(point.book)
    if counts is None or not (1 <= point.chapter <= len(counts)):
        return False
    return 1 <= point.verse <= counts[point.chapter - 1]


# A target verse that is a known textual-variant omission (e.g. Matt 18:11) is a
# REAL reference on the canonical spine — the bibles carry an omitted=1 row for
# each (books.OMITTED_VARIANTS) — so it must NOT be flagged as non-resolving even
# though some of them sit at the end of a chapter's count. They are within the KJV
# verse count already, so point_resolves accepts them; this set is kept for the
# report's note only.
OMITTED_TARGET_NOTE = (
    "Targets landing on a textual-variant omitted verse (e.g. Matt 18:11) resolve "
    "to the bibles' omitted=1 placeholder row — counted as resolving, not flagged."
)
