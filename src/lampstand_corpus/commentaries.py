"""P3 — Commentaries ingestion (CCEL ThML → paragraph-level NormalizedChunks).

Canonical source: CCEL (CLAUDE.md §Sources of record), public domain, each
commentator verified against a known authoritative edition. v1 scope is
architect-locked (spec §4.2):

* **Matthew Henry** — *Commentary on the Whole Bible* (6 CCEL volumes mhc1-6).
  Complete (whole Bible).
* **Jamieson-Fausset-Brown (JFB)** — *Commentary Critical and Explanatory on the
  Whole Bible* (one CCEL file). Complete.
* **John Calvin** — *Commentaries*, **NT + Psalms + Genesis ONLY** for v1. The
  remaining OT (Pentateuch harmony, Joshua, the major & minor prophets) is
  DEFERRED and cleanly skipped (the volumes are simply not snapshotted). Calvin
  wrote no commentary on Revelation, 2-3 John, or Jude — those NT books are
  legitimately absent and surfaced as expected-missing, not errors.
* **John Gill** — DEFERRED to v1.1. NOT ingested (no source defined here).

How the texts converge on the normalized schema
------------------------------------------------
Every CCEL commentary volume marks the start of comment on a passage with an
empty ``<scripCom>`` anchor carrying ``parsed="|OsisBook|chap|vstart|endchap|
vend|"`` and ``osisRef="Bible:Book.c.v-Book.c.v"``. The commentary prose for that
passage is the run of ``<p>`` elements following the anchor, up to the next
``<scripCom>`` (or the end of the chapter container). We emit **one paragraph-
level chunk per ``<p>``**, each anchored to the ``VerseRef`` (a single verse or a
verse *range* / pericope) of the governing ``scripCom``. A whole-chapter anchor
(``|Book|c|0|0|0``) carries the chapter introduction; its chunks anchor to the
chapter with ``verse_start=0`` flagged as a chapter-level note.

Every CCEL structural ambiguity is FLAGGED for human review, never silently
resolved (CLAUDE.md): scripCom anchors whose ``parsed`` can't be mapped to the
canonical 66-book spine, comment blocks that don't sit under a recognizable
verse anchor, and statistical anomalies (an unusually long single block — e.g.
Calvin on a long Psalm — is surfaced, not "fixed").

SPURGEON — *Treasury of David* — INGESTED from Internet Archive OCR
-------------------------------------------------------------------
The CCEL edition (spurgeon/treasury1-6) is a page-image scan with no machine
text, so the architect approved the Google-digitized ``*spurgoog`` DjVu scans on
archive.org instead. That ingestion is a separate path (substantial OCR cleaning,
not CCEL ThML parsing) and lives in the sibling ``spurgeon`` module. Spurgeon is
therefore NOT in ``COMMENTARY_SOURCES`` (which the CCEL snapshot/normalize loops
iterate); it joins ``all_commentary_sources()`` — the registry the shared build
resolves by id. The Treasury is a CANDIDATE only: it is OCR, the volume covering
Psalms 104-118 is absent from the scan set (flagged), and the architect's
spot-check decides v1-vs-v1.1. See ``spurgeon.py`` for the full account.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

from . import books
from .schema import NormalizedChunk, Provenance, ResourceType, VerseRef
from .sources import SOURCES_DIR

COMMENTARIES_DIR = SOURCES_DIR / "commentaries"


# --- OSIS book code -> USFM book id (shared spine with confessions) ----------
# CCEL scripCom anchors use OSIS book codes in both osisRef and the "parsed"
# pipe string. Reuse the canonical mapping; declared locally to avoid importing
# from the confessions module (keeps P3 independent of P2).
OSIS_TO_USFM: dict[str, str] = {
    "Gen": "GEN", "Exod": "EXO", "Lev": "LEV", "Num": "NUM", "Deut": "DEU",
    "Josh": "JOS", "Judg": "JDG", "Ruth": "RUT", "1Sam": "1SA", "2Sam": "2SA",
    "1Kgs": "1KI", "2Kgs": "2KI", "1Chr": "1CH", "2Chr": "2CH", "Ezra": "EZR",
    "Neh": "NEH", "Esth": "EST", "Job": "JOB", "Ps": "PSA", "Prov": "PRO",
    "Eccl": "ECC", "Song": "SNG", "Isa": "ISA", "Jer": "JER", "Lam": "LAM",
    "Ezek": "EZK", "Dan": "DAN", "Hos": "HOS", "Joel": "JOL", "Amos": "AMO",
    "Obad": "OBA", "Jonah": "JON", "Mic": "MIC", "Nah": "NAM", "Hab": "HAB",
    "Zeph": "ZEP", "Hag": "HAG", "Zech": "ZEC", "Mal": "MAL",
    "Matt": "MAT", "Mark": "MRK", "Luke": "LUK", "John": "JHN", "Acts": "ACT",
    "Rom": "ROM", "1Cor": "1CO", "2Cor": "2CO", "Gal": "GAL", "Eph": "EPH",
    "Phil": "PHP", "Col": "COL", "1Thess": "1TH", "2Thess": "2TH",
    "1Tim": "1TI", "2Tim": "2TI", "Titus": "TIT", "Phlm": "PHM", "Heb": "HEB",
    "Jas": "JAS", "1Pet": "1PE", "2Pet": "2PE", "1John": "1JN", "2John": "2JN",
    "3John": "3JN", "Jude": "JUD", "Rev": "REV",
}

# A single, unusually long extracted comment block is a statistical anomaly worth
# a human glance (CLAUDE.md cites "Calvin on Psalm 119" as the canonical example).
# This is a FLAG threshold, never a truncation — long blocks are kept verbatim.
LONG_BLOCK_CHARS = 12_000


@dataclass(frozen=True)
class CommentarySource:
    """One commentator's CCEL snapshot definition (may span multiple volumes)."""

    id: str               # 'henry','jfb','calvin'
    name: str
    shortcode: str        # citation prefix: MH / JFB / "Calvin Comm."
    author: str
    work: str
    volumes: tuple[str, ...]   # CCEL volume stems, e.g. ('mhc1',...,'mhc6')
    author_slug: str           # CCEL author path segment, e.g. 'henry'
    version: str
    license: str

    def url(self, volume: str) -> str:
        return f"https://ccel.org/ccel/{self.author_slug}/{volume}.xml"

    def dest(self, volume: str) -> Path:
        return COMMENTARIES_DIR / self.id / f"{volume}.xml"


# Calvin volume -> in/out of v1 scope. v1 = Genesis + Psalms + the New Testament.
# The remaining OT (Harmony of the Law, Joshua, Isaiah, Jeremiah/Lamentations,
# Ezekiel, Daniel, the Minor Prophets) is DEFERRED — those volumes are simply not
# listed here, so they are never fetched or parsed. (Mapping verified against each
# volume's CCEL DC.Title + scripCom osisRefs.)
CALVIN_VOLUMES_IN_SCOPE: tuple[str, ...] = (
    "calcom01", "calcom02",                              # Genesis (2 vols)
    "calcom08", "calcom09", "calcom10", "calcom11", "calcom12",  # Psalms (5 vols)
    # New Testament (Gospels harmony, John, Acts, Pauline + Catholic Epistles):
    "calcom31", "calcom32", "calcom33",                  # Matthew/Mark/Luke (3)
    "calcom34", "calcom35",                              # John (2)
    "calcom36", "calcom37",                              # Acts (2)
    "calcom38",                                          # Romans
    "calcom39", "calcom40",                              # Corinthians (2)
    "calcom41",                                          # Galatians, Ephesians
    "calcom42",                                          # Phil, Col, Thess
    "calcom43",                                          # Tim, Titus, Philemon
    "calcom44",                                          # Hebrews
    "calcom45",                                          # Catholic Epistles
)

# Calvin OT volumes DEFERRED for v1 (surfaced in the report; never fetched).
CALVIN_VOLUMES_DEFERRED: dict[str, str] = {
    "calcom03-06": "Harmony of the Law (Exodus-Deuteronomy, 4 vols)",
    "calcom07": "Joshua",
    "calcom13-16": "Isaiah (4 vols)",
    "calcom17-21": "Jeremiah & Lamentations (5 vols)",
    "calcom22-23": "Ezekiel (2 vols)",
    "calcom24-25": "Daniel (2 vols)",
    "calcom26": "Hosea",
    "calcom27": "Joel, Amos, Obadiah",
    "calcom28": "Jonah, Micah, Nahum",
    "calcom29": "Habakkuk, Zephaniah, Haggai",
    "calcom30": "Zechariah, Malachi",
}

# Books Calvin never commented on (legitimately absent from his NT corpus, so
# NOT an ingestion error — surfaced as expected-missing in the report). Calvin's
# Catholic-Epistles volume (calcom45) DOES cover Jude (147 chunks, verified in the
# build), so Jude is NOT in this list; he left only 2 John, 3 John, and Revelation
# without a commentary.
CALVIN_NT_NOT_WRITTEN: tuple[str, ...] = ("2JN", "3JN", "REV")


COMMENTARY_SOURCES: dict[str, CommentarySource] = {
    "henry": CommentarySource(
        id="henry",
        name="Matthew Henry's Commentary on the Whole Bible",
        shortcode="MH",
        author="Matthew Henry",
        work="An Exposition of the Old and New Testaments (Commentary on the Whole Bible)",
        volumes=("mhc1", "mhc2", "mhc3", "mhc4", "mhc5", "mhc6"),
        author_slug="henry",
        version="CCEL ThML (complete; 6 volumes mhc1-mhc6)",
        license="Public domain (CCEL)",
    ),
    "jfb": CommentarySource(
        id="jfb",
        name="Jamieson-Fausset-Brown Commentary on the Whole Bible",
        shortcode="JFB",
        author="Robert Jamieson, A. R. Fausset, David Brown",
        work="Commentary Critical and Explanatory on the Whole Bible",
        volumes=("jfb",),
        author_slug="jamieson",
        version="CCEL ThML (complete; single file)",
        license="Public domain (CCEL)",
    ),
    "calvin": CommentarySource(
        id="calvin",
        name="John Calvin's Commentaries (NT + Psalms + Genesis)",
        shortcode="Calvin Comm.",
        author="John Calvin",
        work="Calvin's Commentaries",
        volumes=CALVIN_VOLUMES_IN_SCOPE,
        author_slug="calvin",
        version=(
            "CCEL ThML (v1 scope: Genesis + Psalms + New Testament; remaining OT "
            "deferred to a later corpus version)"
        ),
        license="Public domain (CCEL)",
    ),
}


# Spurgeon's *Treasury of David* is no longer flagged-and-skipped: the architect
# approved the Internet Archive Google-OCR scans (the CCEL edition is image-only).
# Its source descriptor + parser live in the sibling ``spurgeon`` module (the OCR
# cleaning is substantial and kept separate from the CCEL ThML path). It is NOT in
# COMMENTARY_SOURCES (which the CCEL snapshot/normalize loops iterate) — instead it
# joins ALL_COMMENTARY_SOURCES, the registry the shared build resolves by id.
def all_commentary_sources() -> dict:
    """CCEL commentators + Spurgeon, keyed by id, for the shared build/writer.

    Lazy import of the Spurgeon source avoids a circular import (spurgeon imports
    schema + sources, not commentaries).
    """
    from .spurgeon import SPURGEON_SOURCE

    combined: dict = dict(COMMENTARY_SOURCES)
    combined[SPURGEON_SOURCE.id] = SPURGEON_SOURCE
    return combined


@dataclass
class ParsedCommentary:
    """One commentator's normalized chunks plus any ambiguities to flag."""

    id: str
    chunks: list[NormalizedChunk]
    flags: list[str]
    # Per-(book) coverage: book_id -> set of chapters that produced >=1 chunk.
    coverage: dict[str, set[int]] = field(default_factory=dict)


def _norm_ws(text: str) -> str:
    return re.sub(r"[ \t\n\r ]+", " ", text).strip()


def _parse_scripcom(parsed: str, osis: str) -> tuple[VerseRef | None, bool, str | None]:
    """Map a scripCom anchor to a VerseRef.

    Returns ``(ref, chapter_level, reason_if_none)``.

    ``parsed`` is CCEL's ``|OsisBook|chap|vstart|endchap|vend|`` pipe string;
    ``osis`` (``Bible:Book.c.v-Book.c.v``) is the cross-check. We trust ``parsed``
    for the numeric range and require the book to resolve to the 66-book spine.
    A ``vstart`` of 0 marks a whole-chapter (chapter-introduction) anchor.
    """
    fields = parsed.strip("|").split("|") if parsed else []
    if len(fields) != 5:
        return None, False, f"unparseable scripCom parsed={parsed!r} osis={osis!r}"
    osis_book, chap_s, vstart_s, endchap_s, vend_s = fields
    usfm = OSIS_TO_USFM.get(osis_book)
    if usfm is None:
        # Not in the canonical 66 (e.g. an Apocrypha ref inside a comment anchor).
        return None, False, f"scripCom book {osis_book!r} outside the 66-book canon (osis={osis!r})"
    if not chap_s.isdigit():
        return None, False, f"scripCom non-numeric chapter in parsed={parsed!r}"
    chapter = int(chap_s)
    vstart = int(vstart_s) if vstart_s.isdigit() else 0
    vend = int(vend_s) if vend_s.isdigit() else 0
    if vstart == 0:
        # Whole-chapter anchor (chapter introduction). Anchor verse_start=0.
        return (
            VerseRef(book=usfm, chapter=chapter, verse_start=0, verse_end=None),
            True,
            None,
        )
    verse_end = vend if vend and vend >= vstart else None
    return (
        VerseRef(book=usfm, chapter=chapter, verse_start=vstart, verse_end=verse_end),
        False,
        None,
    )


def _make_chunk(
    src: CommentarySource, prov: Provenance, *, key: str, ref: VerseRef,
    text: str, chapter_level: bool, para_index: int, volume: str,
    passage_label: str,
) -> NormalizedChunk:
    return NormalizedChunk(
        id=f"{src.id}:{key}",
        resource_type=ResourceType.COMMENTARY,
        ref=ref,
        key=key,
        text=text,
        meta={
            "author": src.author,
            "work": src.work,
            "shortcode": src.shortcode,
            "volume": volume,
            "passage": passage_label,       # CCEL's human-readable passage label
            "chapter_level": chapter_level,  # True for whole-chapter intro notes
            "para_index": para_index,        # paragraph ordinal within the anchor
        },
        provenance=prov,
    )


def _ref_key(src_id: str, ref: VerseRef, para_index: int) -> str:
    """Deterministic, collision-free chunk key.

    ``<src>:<BOOK>.<chap>.<vstart>[-<vend>]#p<n>`` — the paragraph ordinal keeps
    the many paragraphs that share one verse-anchor distinct and stable-ordered.
    """
    base = f"{ref.book}.{ref.chapter}.{ref.verse_start}"
    if ref.verse_end and ref.verse_end != ref.verse_start:
        base += f"-{ref.verse_end}"
    return f"{base}#p{para_index}"


def _iter_chapter_containers(soup: BeautifulSoup):
    """Yield the div elements that each hold exactly one Bible chapter's comment.

    A chapter container is the deepest ``div*`` whose own ``scripCom`` anchors all
    belong to a single (book, chapter). We detect it as a div that contains
    ``scripCom`` anchors but whose child divs do NOT (so book-level div1/div2 that
    merely nest chapter divs are skipped, and we descend to the chapter level).
    This makes the parser robust across the differing nestings CCEL uses:
    Henry div1=book/div2=chapter, JFB div2=book/div3=chapter, Calvin div2=chapter.
    """
    body = soup.find("ThML.body") or soup
    for div in body.find_all(["div1", "div2", "div3", "div4"]):
        if not div.find("scripCom"):
            continue
        # If ANY descendant div also carries scripComs, this is a higher-level
        # (book/section) wrapper, not the leaf chapter container — skip it; we'll
        # reach the descendant(s) on their own iteration. Checking *any* descendant
        # (not just the first child) prevents double-parsing a chapter whose
        # wrapper div's first child happens to be non-chapter front matter.
        if any(d.find("scripCom") is not None
               for d in div.find_all(["div1", "div2", "div3", "div4"])):
            continue
        yield div


def parse_commentary(
    src: CommentarySource, prov_by_volume: dict[str, Provenance],
    content_by_volume: dict[str, str],
) -> ParsedCommentary:
    """Parse all of a commentator's CCEL volumes into paragraph-level chunks."""
    flags: list[str] = []
    chunks: list[NormalizedChunk] = []
    coverage: dict[str, set[int]] = {}

    for volume in src.volumes:
        content = content_by_volume[volume]
        prov = prov_by_volume[volume]
        soup = BeautifulSoup(content, "lxml-xml")

        for chap_div in _iter_chapter_containers(soup):
            # Walk the chapter container's <p> and <scripCom> in document order.
            # Prose before the first scripCom (the chapter-introduction lede that
            # Henry places ahead of the whole-chapter anchor) is buffered and
            # attached to the first anchor encountered — never dropped.
            cur_ref: VerseRef | None = None
            cur_chapter_level = False
            cur_passage = ""
            pending_pre: list[str] = []
            para_index = 0

            elements = chap_div.find_all(["scripCom", "p"])
            for el in elements:
                if el.name == "scripCom":
                    ref, chap_level, reason = _parse_scripcom(
                        el.get("parsed", ""), el.get("osisRef", "")
                    )
                    if ref is None:
                        flags.append(f"{src.id}/{volume}: {reason} — review")
                        # Keep the current anchor in force (don't lose following
                        # prose); the bad anchor's own prose stays with prior ref.
                        continue
                    # Reset the paragraph ordinal for the new anchor, then flush any
                    # pre-anchor lede (chapter-intro prose Henry places ahead of the
                    # whole-chapter anchor) onto it — never dropped, never colliding.
                    para_index = 0
                    if cur_ref is None and pending_pre:
                        passage = _norm_ws(el.get("passage", ""))
                        for ptxt in pending_pre:
                            para_index += 1
                            chunks.append(_emit(
                                src, prov, ref, ptxt, chap_level,
                                para_index, volume, passage, coverage,
                            ))
                        pending_pre = []
                    cur_ref = ref
                    cur_chapter_level = chap_level
                    cur_passage = _norm_ws(el.get("passage", ""))
                    continue
                # An ordinary <p>. Skip the empty <p> wrappers that merely hold a
                # scripCom anchor (handled above) so anchor wrappers never count.
                if el.find("scripCom") is not None:
                    continue
                txt = _norm_ws(el.get_text(" ", strip=True))
                if not txt:
                    continue
                if cur_ref is None:
                    # Pre-anchor chapter-introduction lede; buffer it.
                    pending_pre.append(txt)
                    continue
                para_index += 1
                chunks.append(_emit(
                    src, prov, cur_ref, txt, cur_chapter_level,
                    para_index, volume, cur_passage, coverage,
                ))

            # A chapter container whose lede never met a scripCom (no anchor at
            # all) would strand text — flag it rather than guessing a ref.
            if cur_ref is None and pending_pre:
                title = _norm_ws(chap_div.get("title") or "")
                flags.append(
                    f"{src.id}/{volume}: {len(pending_pre)} prose paragraph(s) under "
                    f"{title!r} have no scripCom verse anchor — unmapped, not guessed "
                    "— review"
                )

    chunks.sort(key=lambda c: _sort_key(c))

    # Disambiguate any keys that legitimately repeat within a commentator. This
    # happens when CCEL splits one Bible chapter across two sibling div blocks
    # (e.g. Calvin's Genesis 7 / Henry's 2 Chronicles 27 — a page break opens a
    # second container for the same chapter) or encodes a single-chapter book's
    # whole-chapter anchor and its verse-1 anchor to the same ref (e.g. JFB
    # Obadiah: "Obadiah 1" + "Ob 1:1"). We keep BOTH blocks (never drop content),
    # append a deterministic "~N" occurrence suffix to the later one(s), and FLAG
    # so a human can confirm it is a benign source split, not a merge error.
    base_counts: dict[str, int] = {}
    repeated_bases: set[str] = set()
    for c in chunks:
        base = c.key
        n = base_counts.get(base, 0)
        base_counts[base] = n + 1
        if n:
            repeated_bases.add(base)
            c.key = f"{base}~{n + 1}"
            c.id = f"{src.id}:{c.key}"
    if repeated_bases:
        flags.append(
            f"{src.id}: {len(repeated_bases)} verse-anchor key(s) repeat within the "
            "commentator (CCEL split one chapter across sibling div blocks, or a "
            "single-chapter book's whole-chapter and verse-1 anchors coincide). "
            "Both blocks kept; later occurrence(s) given a deterministic '~N' key "
            "suffix — verify this is a benign source split, not a merge error — review"
        )

    return ParsedCommentary(id=src.id, chunks=chunks, flags=flags, coverage=coverage)


def _emit(
    src: CommentarySource, prov: Provenance, ref: VerseRef, text: str,
    chapter_level: bool, para_index: int, volume: str, passage: str,
    coverage: dict[str, set[int]],
) -> NormalizedChunk:
    coverage.setdefault(ref.book, set()).add(ref.chapter)
    key = _ref_key(src.id, ref, para_index)
    return _make_chunk(
        src, prov, key=key, ref=ref, text=text,
        chapter_level=chapter_level, para_index=para_index, volume=volume,
        passage_label=passage,
    )


def _sort_key(c: NormalizedChunk) -> tuple:
    r = c.ref
    return (
        books.ORDER_INDEX.get(r.book, 999), r.chapter, r.verse_start,
        r.verse_end or r.verse_start, c.meta.get("para_index", 0),
    )
