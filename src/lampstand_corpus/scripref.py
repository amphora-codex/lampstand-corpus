"""Parse human-written scripture citations into :class:`VerseRef`s.

The confession sources express proof-texts as prose citations rather than OSIS
ids: the WCF original (westminster-json) inlines them in parentheses
(``Eph. 1:11; Rom. 9:15, 18``), the 1689 lists them in a per-paragraph block
(``2 Timothy 3:15-17; Isaiah 8:20``). We convert those to the canonical
``(book, chapter, verse_start[, verse_end])`` spine so every ref can be
re-validated against ``bibles.sqlite``.

Design rules (CLAUDE.md):
  * Only emit a VerseRef when book + chapter + verse parse unambiguously. A token
    we can't resolve is returned in ``unparsed`` (the caller FLAGS it) — never
    guessed at.
  * Verse ranges (``3:15-17``) collapse to ``verse_start``/``verse_end``; comma
    verse lists (``9:15, 18``) become separate refs that share the chapter.
  * Chapter-only citations (no verse, e.g. ``Psalm 119``) are intentionally NOT
    emitted as verse refs (they have no verse to resolve) and are reported as
    ``unparsed`` so the human sees them rather than the pipeline inventing v.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import VerseRef

# --- book-name / abbreviation -> USFM id -------------------------------------
# Covers full names and the abbreviation styles seen across the WCF (westminster-
# json) and 1689 (ParticularBaptists) sources. Keys are lowercased, trailing
# period stripped, internal whitespace collapsed (see _book_key).
_BOOK_ALIASES: dict[str, str] = {}


def _alias(usfm: str, *names: str) -> None:
    for n in names:
        _BOOK_ALIASES[_book_key(n)] = usfm


def _book_key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().rstrip(".")).lower()


# fmt: off
_alias("GEN", "Genesis", "Gen")
_alias("EXO", "Exodus", "Exod", "Ex")
_alias("LEV", "Leviticus", "Lev")
_alias("NUM", "Numbers", "Num", "Numb")
_alias("DEU", "Deuteronomy", "Deut", "Deu")
_alias("JOS", "Joshua", "Josh", "Jos")
_alias("JDG", "Judges", "Judg", "Jdg")
_alias("RUT", "Ruth")
_alias("1SA", "1 Samuel", "1 Sam", "1Sam")
_alias("2SA", "2 Samuel", "2 Sam", "2Sam")
_alias("1KI", "1 Kings", "1 Kgs", "1Kings")
_alias("2KI", "2 Kings", "2 Kgs", "2Kings")
_alias("1CH", "1 Chronicles", "1 Chron", "1 Chr", "1Chron")
_alias("2CH", "2 Chronicles", "2 Chron", "2 Chr", "2Chron")
_alias("EZR", "Ezra")
_alias("NEH", "Nehemiah", "Neh")
_alias("EST", "Esther", "Esth")
_alias("JOB", "Job")
_alias("PSA", "Psalms", "Psalm", "Ps", "Pss", "Psa")
_alias("PRO", "Proverbs", "Prov", "Pro")
_alias("ECC", "Ecclesiastes", "Eccles", "Eccl", "Ecc")
_alias("SNG", "Song of Solomon", "Song of Songs", "Song", "Cant", "Canticles")
_alias("ISA", "Isaiah", "Isa")
_alias("JER", "Jeremiah", "Jer")
_alias("LAM", "Lamentations", "Lam")
_alias("EZK", "Ezekiel", "Ezek", "Eze")
_alias("DAN", "Daniel", "Dan")
_alias("HOS", "Hosea", "Hos")
_alias("JOL", "Joel")
_alias("AMO", "Amos")
_alias("OBA", "Obadiah", "Obad")
_alias("JON", "Jonah", "Jon")
_alias("MIC", "Micah", "Mic")
_alias("NAM", "Nahum", "Nah")
_alias("HAB", "Habakkuk", "Hab")
_alias("ZEP", "Zephaniah", "Zeph")
_alias("HAG", "Haggai", "Hag")
_alias("ZEC", "Zechariah", "Zech", "Zec")
_alias("MAL", "Malachi", "Mal")
_alias("MAT", "Matthew", "Matt", "Mat")
_alias("MRK", "Mark")
_alias("LUK", "Luke")
_alias("JHN", "John")
_alias("ACT", "Acts")
_alias("ROM", "Romans", "Rom")
_alias("1CO", "1 Corinthians", "1 Cor", "1Cor")
_alias("2CO", "2 Corinthians", "2 Cor", "2Cor")
_alias("GAL", "Galatians", "Gal")
_alias("EPH", "Ephesians", "Eph", "Ephes")
_alias("PHP", "Philippians", "Phil", "Php")
_alias("COL", "Colossians", "Col")
_alias("1TH", "1 Thessalonians", "1 Thess", "1 Thes", "1Thess")
_alias("2TH", "2 Thessalonians", "2 Thess", "2 Thes", "2Thess")
_alias("1TI", "1 Timothy", "1 Tim", "1Tim")
_alias("2TI", "2 Timothy", "2 Tim", "2Tim")
_alias("TIT", "Titus", "Tit")
_alias("PHM", "Philemon", "Phlm", "Philem")
_alias("HEB", "Hebrews", "Heb")
_alias("JAS", "James", "Jam", "Jas")
_alias("1PE", "1 Peter", "1 Pet", "1Pet")
_alias("2PE", "2 Peter", "2 Pet", "2Pet")
_alias("1JN", "1 John", "1John")
_alias("2JN", "2 John", "2John")
_alias("3JN", "3 John", "3John")
_alias("JUD", "Jude")
_alias("REV", "Revelation", "Rev")
# fmt: on


# One-chapter books: a citation like "Jude 4" or "2 John 10, 11" names a verse,
# not a chapter, so it resolves to chapter 1. (Multi-chapter chapter-only refs
# like "Lev. 18" stay unresolved — there is no single verse to point at.)
_ONE_CHAPTER_BOOKS: frozenset[str] = frozenset(
    {"OBA", "PHM", "2JN", "3JN", "JUD"}
)


@dataclass
class ScripRefResult:
    refs: list[VerseRef] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)


# A single citation token, e.g. "1 Cor. 2:13, 14" or "Romans 1:19-21". The book
# group allows a leading 1/2/3 and multi-word names ("Song of Solomon"); group 2
# is the chapter:verse tail (must begin with a digit).
_CITATION_RE = re.compile(
    r"^\s*([1-3]?\s*[A-Za-z][A-Za-z. ]*?)\s+(\d.*?)\s*$"
)
# chapter:verse(-verse)? possibly followed by extra ", v" verse-only continuations.
_CHAP_VERSE_RE = re.compile(r"^(\d+)\s*[:.]\s*(\d+)(?:\s*[-–]\s*(\d+))?$")
# Connectors that join multiple distinct citations inside one token, e.g.
# "Gen. 2:7 with Eccles. 12:7 & Luke 23:43 and Matt. 10:28". We split on these so
# each sub-citation is resolved independently (the connectors carry no verse data).
_CONNECTOR_RE = re.compile(r"\s+(?:with|and|&|&c\.?|etc\.?)\s+|\s*&\s*", re.IGNORECASE)
# Trailing prose tails the sources append to citations ("8:28 to the end",
# "Psalm 73 throughout"). Stripped before parsing; what remains parses or flags.
_TAIL_NOISE_RE = re.compile(
    r"\s+(?:to the end|throughout|&c\.?|etc\.?|ff\.?)\s*\.?\s*$", re.IGNORECASE
)


def _resolve_book(raw: str) -> str | None:
    return _BOOK_ALIASES.get(_book_key(raw))


def _parse_one(book_usfm: str, tail: str) -> tuple[list[VerseRef], list[str]]:
    """Parse the ``chapter:verse...`` tail of one citation into VerseRefs.

    Handles ``3:15``, ``3:15-17`` (range), and ``9:15, 18`` (verse list sharing a
    chapter). A bare ``119`` chapter-only tail yields no ref (returned unparsed).
    """
    refs: list[VerseRef] = []
    unparsed: list[str] = []
    # In a one-chapter book the running chapter is implicitly 1, so "Jude 4, 6, 7"
    # parses as verse-only continuations against chapter 1.
    cur_chapter: int | None = 1 if book_usfm in _ONE_CHAPTER_BOOKS else None
    for part in re.split(r",", tail):
        part = part.strip()
        if not part:
            continue
        m = _CHAP_VERSE_RE.match(part)
        if m:
            cur_chapter = int(m.group(1))
            vs = int(m.group(2))
            ve = int(m.group(3)) if m.group(3) else None
            refs.append(VerseRef(
                book=book_usfm, chapter=cur_chapter,
                verse_start=vs, verse_end=ve,
            ))
            continue
        # A verse-only continuation ("18" or "9-11") sharing the running chapter.
        vm = re.match(r"^(\d+)(?:\s*[-–]\s*(\d+))?$", part)
        if vm and cur_chapter is not None:
            vs = int(vm.group(1))
            ve = int(vm.group(2)) if vm.group(2) else None
            refs.append(VerseRef(
                book=book_usfm, chapter=cur_chapter,
                verse_start=vs, verse_end=ve,
            ))
            continue
        # A comma-part that re-states a book ("Job 15:14" inside "Job 14:4, Job
        # 15:14") is its own citation — resolve it and switch the running book.
        cm = _CITATION_RE.match(part)
        if cm:
            inner = _resolve_book(cm.group(1))
            if inner is not None:
                sub_refs, sub_unparsed = _parse_one(inner, cm.group(2))
                refs.extend(sub_refs)
                unparsed.extend(sub_unparsed)
                book_usfm = inner
                cur_chapter = (sub_refs[-1].chapter if sub_refs
                               else cur_chapter)
                continue
        # Anything else (chapter-only, ranges across chapters, malformed) — flag.
        unparsed.append(f"{book_usfm} {tail!r} (part {part!r})")
    return refs, unparsed


def parse_proof_block(block: str) -> ScripRefResult:
    """Parse a ``;``-separated citation block into VerseRefs.

    ``block`` is the inside of a proof-text group, e.g.
    ``"2 Timothy 3:15-17; Isaiah 8:20; Luke 16:29, 31"``. Citations that don't
    resolve (unknown book, chapter-only, malformed) land in ``unparsed`` for the
    caller to FLAG; nothing is guessed.
    """
    result = ScripRefResult()
    for raw_token in block.split(";"):
        # One ";"-group may chain several citations via "with"/"and"/"&". Split
        # them so each "Book ch:v" resolves on its own; a fragment without its own
        # book is a verse-only continuation we re-attach below.
        last_book: str | None = None
        for token in _CONNECTOR_RE.split(raw_token):
            token = _TAIL_NOISE_RE.sub("", token).strip().strip(".,").strip()
            # Drop a "ver."/"verse" word between book and number ("Jude ver. 4").
            token = re.sub(r"\b(?:ver|verse|vers)\.?\s+(?=\d)", "", token,
                           flags=re.IGNORECASE)
            if not token:
                continue
            cm = _CITATION_RE.match(token)
            if cm and _resolve_book(cm.group(1)) is not None:
                last_book = _resolve_book(cm.group(1))
                refs, unparsed = _parse_one(last_book, cm.group(2))
                result.refs.extend(refs)
                result.unparsed.extend(unparsed)
                continue
            # A bare "ch:v" fragment after a connector inherits the running book.
            if last_book is not None and _CHAP_VERSE_RE.match(token):
                refs, unparsed = _parse_one(last_book, token)
                result.refs.extend(refs)
                result.unparsed.extend(unparsed)
                continue
            result.unparsed.append(token)
    return result
