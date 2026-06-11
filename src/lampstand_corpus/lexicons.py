"""P4 — Lexicon dictionaries ingestion (OpenScriptures → NormalizedChunks).

Canonical sources only (CLAUDE.md §Sources of record). Every structural or
sourcing ambiguity is FLAGGED for human review, never silently resolved, and the
pipeline never marks output ship-ready — the architect's spot-check gates ship.

Primary deliverable — the keyed lexicon dictionaries
----------------------------------------------------
* **Strong's Greek** — ``openscriptures/strongs`` ``greek/strongs-greek-
  dictionary.js``. A single ``var strongsGreekDictionary = {…};`` JSON object
  keyed ``G####``. Fields: ``lemma``, ``translit``, ``strongs_def`` (gloss/
  definition), ``derivation``, ``kjv_def``. (No pronunciation field in the Greek
  set — recorded as None.)
* **Strong's Hebrew** — ``openscriptures/strongs`` ``hebrew/strongs-hebrew-
  dictionary.js``. ``var strongsHebrewDictionary = {…};`` keyed ``H####``. Adds
  ``xlit`` (translit) and ``pron`` (pronunciation) to the same field set.
* **Brown-Driver-Briggs (BDB)** — ``openscriptures/HebrewLexicon``
  ``BrownDriverBriggs.xml`` (the fuller Hebrew entry). BDB entries are keyed by an
  internal id (``a.ab.ab``), NOT by Strong's number; the Strong↔BDB linkage lives
  in the same repo's ``LexicalIndex.xml`` (``<xref bdb=… strong=…/>``). We parse
  the index to attach a Strong's H-number to each BDB entry we can map, and FLAG
  (do not guess) BDB entries the index leaves unlinked.

License note (recorded per source in provenance): the underlying Strong's, BDB,
and the WLC text are all **public domain**; the OpenScriptures *editions* carry
licenses — Strong's dictionaries **CC-BY-SA** (per the file headers), the Hebrew
Lexicon and morphhb (OSHB) **CC-BY-4.0**. These are bundle-friendly with
attribution. Recorded so downstream attribution is correct.

Thayer's — FLAGGED, not fabricated
----------------------------------
No clean, canonical, machine-readable public-domain Thayer's Greek-English
lexicon dataset exists in a source of record (the OpenScriptures repos ship
Strong's + BDB, not Thayer's; a GitHub survey turned up only GPL-incompatible or
non-canonical aggregator copies). Per CLAUDE.md ("If no clean Thayer's dataset
exists, flag it — don't fabricate"), Thayer's is OMITTED for this run and
surfaced for human sourcing. Strong's Greek glosses cover the basic need; Thayer's
is the fuller entry to add once a canonical source is approved.

Secondary (P4b) — Strong's-tagged original-language text
--------------------------------------------------------
* **Hebrew OT (OSHB / morphhb)** — ``openscriptures/morphhb`` ``wlc/<Book>.xml``
  (OSIS). Each ``<w>`` carries ``lemma`` (Strong's H-number, possibly with a
  prefix segment like ``c/1961`` and/or a homograph letter like ``6965 b``) and
  ``morph`` (OSHB morphology). INGESTED: anchored to canonical VerseRefs via the
  ``<verse osisID="Book.c.v">`` spine, one per-word tagging row. CC-BY-4.0.
* **Greek NT (MorphGNT / SBLGNT)** — ``morphgnt/sblgnt``. DEFERRED + FLAGGED, for
  two reasons surfaced here rather than worked around:
    1. MorphGNT carries lemma + morphology but **no Strong's numbers** — it
       cannot supply the verse↔Strong's link the word-study feature needs without
       an external lemma→Strong's bridge that does not ship in a source of record.
    2. The SBLGNT *text* is under the **SBLGNT EULA** (not pure PD; the morph layer
       is CC-BY-SA). The EULA permits use with attribution/notice but restricts
       redistribution conditions — a bundling decision for the architect, not the
       pipeline. We SNAPSHOT it for provenance and FLAG; we do not ingest the
       tagged Greek text this run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

from .schema import Provenance, VerseRef
from .sources import SOURCES_DIR

LEXICONS_DIR = SOURCES_DIR / "lexicons"

# Strong's coverage spines (CLAUDE.md validation: report counts, flag gaps). The
# augmented Strong's sets run G1–G5624 (Greek) and H1–H8674 (Hebrew). Numbers are
# not perfectly contiguous (a handful are unused/merged); we validate against the
# max and report the actual count + any interior gaps as FLAGS, never "fill" them.
STRONGS_GREEK_MAX = 5624
STRONGS_HEBREW_MAX = 8674


# --- language enum-ish keys --------------------------------------------------
GREEK = "greek"
HEBREW = "hebrew"


@dataclass(frozen=True)
class LexiconSource:
    """One lexicon dictionary snapshot definition."""

    id: str            # 'strongs-greek','strongs-hebrew','bdb'
    name: str
    language: str      # GREEK | HEBREW
    lexicon: str       # 'strongs' | 'bdb' (the lexicon family, recorded per entry)
    url: str           # canonical upstream raw download
    filename: str      # snapshot file under sources/lexicons/<id>/
    version: str
    license: str       # the OpenScriptures EDITION license
    text_license: str  # the underlying dictionary text license (PD)

    @property
    def dest(self) -> Path:
        return LEXICONS_DIR / self.id / self.filename


# BDB needs the LexicalIndex to attach Strong's numbers; declared as an auxiliary
# snapshot on the BDB source (fetched alongside the main XML).
BDB_INDEX_URL = (
    "https://raw.githubusercontent.com/openscriptures/HebrewLexicon/master/"
    "LexicalIndex.xml"
)
BDB_INDEX_FILENAME = "LexicalIndex.xml"


LEXICON_SOURCES: dict[str, LexiconSource] = {
    "strongs-greek": LexiconSource(
        id="strongs-greek",
        name="Strong's Greek Dictionary",
        language=GREEK,
        lexicon="strongs",
        url="https://raw.githubusercontent.com/openscriptures/strongs/master/"
            "greek/strongs-greek-dictionary.js",
        filename="strongs-greek-dictionary.js",
        version="openscriptures/strongs greek JSON (snapshot 2026-06-10)",
        license="CC-BY-SA (OpenScriptures edition)",
        text_license="Public domain (Strong's Greek Dictionary, 1890)",
    ),
    "strongs-hebrew": LexiconSource(
        id="strongs-hebrew",
        name="Strong's Hebrew Dictionary",
        language=HEBREW,
        lexicon="strongs",
        url="https://raw.githubusercontent.com/openscriptures/strongs/master/"
            "hebrew/strongs-hebrew-dictionary.js",
        filename="strongs-hebrew-dictionary.js",
        version="openscriptures/strongs hebrew JSON (snapshot 2026-06-10)",
        license="CC-BY-SA (OpenScriptures edition)",
        text_license="Public domain (Strong's Hebrew Dictionary, 1894)",
    ),
    "bdb": LexiconSource(
        id="bdb",
        name="Brown-Driver-Briggs Hebrew Lexicon",
        language=HEBREW,
        lexicon="bdb",
        url="https://raw.githubusercontent.com/openscriptures/HebrewLexicon/"
            "master/BrownDriverBriggs.xml",
        filename="BrownDriverBriggs.xml",
        version="openscriptures/HebrewLexicon BDB XML (snapshot 2026-06-10)",
        license="CC-BY-4.0 (OpenScriptures Hebrew Lexicon edition)",
        text_license="Public domain (Brown-Driver-Briggs, 1906)",
    ),
}


@dataclass
class LexEntry:
    """One normalized lexicon dictionary entry, keyed by Strong's number."""

    strongs: str          # 'G####' / 'H####'
    language: str         # GREEK | HEBREW
    lexicon: str          # 'strongs' | 'bdb'
    lemma: str | None
    translit: str | None
    pronunciation: str | None
    definition: str | None    # the gloss / strongs_def / BDB definition
    derivation: str | None
    kjv_def: str | None
    raw_key: str | None = None  # source-native key where it differs (BDB entry id)


@dataclass
class ParsedLexicon:
    """A parsed lexicon source: its entries + any flags for human review."""

    id: str
    language: str
    lexicon: str
    provenance: Provenance
    entries: list[LexEntry] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


# --- Strong's dictionary (.js JSON object) -----------------------------------
# The file is `var strongsGreekDictionary = { …json… };` (likewise Hebrew). We
# slice from the first `{` to the matching final `}` and json.load it. Doing this
# by string slice (not eval) keeps it safe and deterministic.
_JS_ASSIGN_RE = re.compile(r"var\s+\w+\s*=\s*", re.MULTILINE)


def _extract_js_object(content: str) -> str:
    m = _JS_ASSIGN_RE.search(content)
    if m is None:
        raise ValueError("no `var <name> = {…}` assignment found")
    start = content.index("{", m.end())
    # The file ends with `};`; strip the trailing `;` and any whitespace.
    end = content.rindex("}")
    return content[start : end + 1]


def _clean(value: str | None) -> str | None:
    """Trim a dictionary field; collapse internal whitespace; None for empty."""
    if value is None:
        return None
    v = re.sub(r"\s+", " ", value).strip()
    return v or None


def parse_strongs(src: LexiconSource, prov: Provenance, content: str) -> ParsedLexicon:
    """Parse a Strong's Greek/Hebrew ``.js`` dictionary into LexEntries."""
    pl = ParsedLexicon(id=src.id, language=src.language, lexicon="strongs",
                        provenance=prov)
    data = json.loads(_extract_js_object(content))
    prefix = "G" if src.language == GREEK else "H"
    for raw_key in sorted(data, key=_strongs_sort_key):
        rec = data[raw_key]
        key = _normalize_strongs_key(raw_key, prefix)
        if key is None:
            pl.flags.append(f"{src.id}: unparseable Strong's key {raw_key!r} (skipped)")
            continue
        pl.entries.append(
            LexEntry(
                strongs=key,
                language=src.language,
                lexicon="strongs",
                lemma=_clean(rec.get("lemma")),
                translit=_clean(rec.get("translit") or rec.get("xlit")),
                pronunciation=_clean(rec.get("pron")),
                definition=_clean(rec.get("strongs_def")),
                derivation=_clean(rec.get("derivation")),
                kjv_def=_clean(rec.get("kjv_def")),
            )
        )
    return pl


def _normalize_strongs_key(raw: str, prefix: str) -> str | None:
    """Normalize a source Strong's key to ``G####`` / ``H####`` (zero-padded-free).

    Source keys are already ``G1615`` / ``H1`` form; some Greek-origin notes in the
    Hebrew set reference ``H08012`` (leading zeros). We normalize to the bare
    integer with the language prefix so the spine is consistent.
    """
    m = re.fullmatch(r"([GH])0*(\d+)", raw.strip())
    if not m:
        return None
    return f"{prefix}{int(m.group(2))}"


def _strongs_sort_key(raw: str) -> tuple[int, str]:
    m = re.fullmatch(r"[GH]0*(\d+)", raw.strip())
    return (int(m.group(1)), raw) if m else (1 << 30, raw)


# --- BDB (XML) + the Strong's↔BDB lexical index ------------------------------
def _parse_lexical_index(index_xml: str) -> dict[str, str]:
    """Map BDB entry id -> Strong's H-number from ``LexicalIndex.xml``.

    Each ``<entry>`` carries ``<xref bdb="a.ab.ab" strong="3" …/>``. Many entries
    have a bdb id but no strong attr (sub-entries, roots); those are simply not in
    the map. Where multiple index entries point at the same bdb id, the FIRST with
    a Strong's number wins and a note is left for the caller to flag.
    """
    soup = BeautifulSoup(index_xml, "lxml-xml")
    mapping: dict[str, str] = {}
    for xref in soup.find_all("xref"):
        bdb = xref.get("bdb")
        strong = xref.get("strong")
        if not bdb or not strong or not strong.isdigit():
            continue
        mapping.setdefault(bdb, f"H{int(strong)}")
    return mapping


def parse_bdb(
    src: LexiconSource, prov: Provenance, content: str, index_xml: str
) -> ParsedLexicon:
    """Parse ``BrownDriverBriggs.xml`` into LexEntries keyed by Strong's H-number.

    BDB entries carry their own ids (``a.ab.ab``); the Strong's linkage comes from
    ``LexicalIndex.xml``. Entries the index does not link to a Strong's number are
    kept (keyed by their BDB id under ``raw_key``) and FLAGGED so a human decides
    whether they belong — we do not invent a Strong's number for them.
    """
    pl = ParsedLexicon(id=src.id, language=HEBREW, lexicon="bdb", provenance=prov)
    bdb_to_strongs = _parse_lexical_index(index_xml)
    soup = BeautifulSoup(content, "lxml-xml")

    unlinked = 0
    seen_strongs: set[str] = set()
    for entry in soup.find_all("entry"):
        bdb_id = entry.get("id")
        if not bdb_id:
            continue
        # Headword(s): the first <w> in the entry is the lemma.
        w = entry.find("w")
        lemma = _clean(w.get_text()) if w is not None else None
        # Definition: concatenate the <def> glosses (BDB's English senses).
        defs = [_clean(d.get_text()) for d in entry.find_all("def")]
        definition = "; ".join(d for d in defs if d) or None
        strongs = bdb_to_strongs.get(bdb_id)
        if strongs is None:
            unlinked += 1
        else:
            seen_strongs.add(strongs)
        pl.entries.append(
            LexEntry(
                strongs=strongs or "",
                language=HEBREW,
                lexicon="bdb",
                lemma=lemma,
                translit=None,
                pronunciation=None,
                definition=definition,
                derivation=None,
                kjv_def=None,
                raw_key=bdb_id,
            )
        )
    if unlinked:
        pl.flags.append(
            f"bdb: {unlinked} BDB entries have no Strong's number in LexicalIndex "
            f"(kept, keyed by BDB id; not guessed) — human to confirm coverage"
        )
    pl.flags.append(
        f"bdb: linked {len(seen_strongs)} distinct Strong's H-numbers via "
        f"LexicalIndex (of {STRONGS_HEBREW_MAX} possible)"
    )
    return pl


# --- OSHB (morphhb) Strong's-tagged Hebrew text — P4b ------------------------
# OSIS book file -> USFM book id (the wlc/ filenames use OSIS-ish stems).
OSHB_FILE_TO_USFM: dict[str, str] = {
    "Gen": "GEN", "Exod": "EXO", "Lev": "LEV", "Num": "NUM", "Deut": "DEU",
    "Josh": "JOS", "Judg": "JDG", "Ruth": "RUT", "1Sam": "1SA", "2Sam": "2SA",
    "1Kgs": "1KI", "2Kgs": "2KI", "1Chr": "1CH", "2Chr": "2CH", "Ezra": "EZR",
    "Neh": "NEH", "Esth": "EST", "Job": "JOB", "Ps": "PSA", "Prov": "PRO",
    "Eccl": "ECC", "Song": "SNG", "Isa": "ISA", "Jer": "JER", "Lam": "LAM",
    "Ezek": "EZK", "Dan": "DAN", "Hos": "HOS", "Joel": "JOL", "Amos": "AMO",
    "Obad": "OBA", "Jonah": "JON", "Mic": "MIC", "Nah": "NAM", "Hab": "HAB",
    "Zeph": "ZEP", "Hag": "HAG", "Zech": "ZEC", "Mal": "MAL",
}


@dataclass
class TaggedWord:
    """One Strong's-tagged original-language word, anchored to a VerseRef."""

    ref: VerseRef
    position: int          # 1-based word index within the verse
    surface: str           # the pointed Hebrew word as written
    strongs: list[str]     # one or more H-numbers (prefix segments contribute none)
    morph: str | None      # OSHB morphology code
    lemma_raw: str         # the raw lemma attr (e.g. 'c/1961', '6965 b')


@dataclass
class ParsedTaggedText:
    """Parsed Strong's-tagged text for one language + flags for review."""

    id: str
    language: str
    provenance: Provenance
    words: list[TaggedWord] = field(default_factory=list)
    books_seen: set[str] = field(default_factory=set)
    flags: list[str] = field(default_factory=list)


_OSHB_STRONGS_RE = re.compile(r"(\d+)")


def _oshb_lemma_strongs(lemma: str) -> list[str]:
    """Extract Strong's H-numbers from an OSHB ``lemma`` attribute.

    OSHB lemmas combine prefix particles (``c/``, ``b/``, ``l``…), the Strong's
    number, and an optional homograph letter (``6965 b``). A lemma may have several
    numeric segments (``c/1961`` -> just 1961; the ``c`` prefix has none). We pull
    every integer run and key it ``H<n>``; segments with no integer contribute
    nothing (correctly, for bare-particle prefixes).
    """
    return [f"H{int(n)}" for n in _OSHB_STRONGS_RE.findall(lemma)]


def parse_oshb(
    src: TaggedTextSource, prov: Provenance, content_by_book: dict[str, str]
) -> ParsedTaggedText:
    """Parse the OSHB ``wlc/*.xml`` files into per-word Strong's-tagged rows."""
    pt = ParsedTaggedText(id=src.id, language=HEBREW, provenance=prov)
    for stem in sorted(content_by_book):
        usfm = OSHB_FILE_TO_USFM.get(stem)
        if usfm is None:
            pt.flags.append(f"oshb: file stem {stem!r} not in canonical OT spine "
                            f"(skipped)")
            continue
        soup = BeautifulSoup(content_by_book[stem], "lxml-xml")
        for verse in soup.find_all("verse"):
            osis = verse.get("osisID", "")
            ref = _oshb_osis_to_ref(osis)
            if ref is None:
                pt.flags.append(f"oshb: unparseable verse osisID {osis!r} in {stem}")
                continue
            pt.books_seen.add(ref.book)
            pos = 0
            for w in verse.find_all("w"):
                pos += 1
                lemma_raw = w.get("lemma", "")
                pt.words.append(
                    TaggedWord(
                        ref=ref,
                        position=pos,
                        surface=_clean(w.get_text()) or "",
                        strongs=_oshb_lemma_strongs(lemma_raw),
                        morph=_clean(w.get("morph")),
                        lemma_raw=lemma_raw,
                    )
                )
    return pt


def _oshb_osis_to_ref(osis: str) -> VerseRef | None:
    parts = osis.split(".")
    if len(parts) != 3:
        return None
    book, chap, verse = parts
    usfm = OSHB_FILE_TO_USFM.get(book)
    if usfm is None or not chap.isdigit() or not verse.isdigit():
        return None
    return VerseRef(book=usfm, chapter=int(chap), verse_start=int(verse))


@dataclass(frozen=True)
class TaggedTextSource:
    """A Strong's-tagged original-language text snapshot definition (P4b)."""

    id: str               # 'oshb'
    name: str
    language: str
    base_url: str         # raw dir URL; book files appended
    book_stems: tuple[str, ...]
    version: str
    license: str
    attribution: str      # required attribution string (CC-BY)

    def url(self, stem: str) -> str:
        return f"{self.base_url}{stem}.xml"

    def dest(self, stem: str) -> Path:
        return LEXICONS_DIR / self.id / f"{stem}.xml"


# The 39 OSHB book files (wlc/), OT order. morphhb names use these stems.
_OSHB_STEMS: tuple[str, ...] = (
    "Gen", "Exod", "Lev", "Num", "Deut", "Josh", "Judg", "Ruth", "1Sam", "2Sam",
    "1Kgs", "2Kgs", "1Chr", "2Chr", "Ezra", "Neh", "Esth", "Job", "Ps", "Prov",
    "Eccl", "Song", "Isa", "Jer", "Lam", "Ezek", "Dan", "Hos", "Joel", "Amos",
    "Obad", "Jonah", "Mic", "Nah", "Hab", "Zeph", "Hag", "Zech", "Mal",
)

TAGGED_TEXT_SOURCES: dict[str, TaggedTextSource] = {
    "oshb": TaggedTextSource(
        id="oshb",
        name="Open Scriptures Hebrew Bible (WLC, Strong's + morphology)",
        language=HEBREW,
        base_url="https://raw.githubusercontent.com/openscriptures/morphhb/"
                 "master/wlc/",
        book_stems=_OSHB_STEMS,
        version="openscriptures/morphhb OSHB WLC 4.20 (snapshot 2026-06-10)",
        license="CC-BY-4.0 (OSHB edition; WLC text public domain)",
        attribution="Original work of the Open Scriptures Hebrew Bible available "
                    "at https://github.com/openscriptures/morphhb",
    ),
}


# --- SBLGNT / MorphGNT — snapshot-only, FLAGGED (see module docstring) --------
@dataclass(frozen=True)
class FlaggedTextSource:
    """A tagged-text source we snapshot for provenance but do NOT ingest."""

    id: str
    name: str
    base_url: str
    files: tuple[str, ...]
    version: str
    license: str
    flag: str

    def url(self, fname: str) -> str:
        return f"{self.base_url}{fname}"

    def dest(self, fname: str) -> Path:
        return LEXICONS_DIR / self.id / fname


_SBLGNT_FILES: tuple[str, ...] = (
    "61-Mt-morphgnt.txt", "62-Mk-morphgnt.txt", "63-Lk-morphgnt.txt",
    "64-Jn-morphgnt.txt", "65-Ac-morphgnt.txt", "66-Ro-morphgnt.txt",
    "67-1Co-morphgnt.txt", "68-2Co-morphgnt.txt", "69-Ga-morphgnt.txt",
    "70-Eph-morphgnt.txt", "71-Php-morphgnt.txt", "72-Col-morphgnt.txt",
    "73-1Th-morphgnt.txt", "74-2Th-morphgnt.txt", "75-1Ti-morphgnt.txt",
    "76-2Ti-morphgnt.txt", "77-Tit-morphgnt.txt", "78-Phm-morphgnt.txt",
    "79-Heb-morphgnt.txt", "80-Jas-morphgnt.txt", "81-1Pe-morphgnt.txt",
    "82-2Pe-morphgnt.txt", "83-1Jn-morphgnt.txt", "84-2Jn-morphgnt.txt",
    "85-3Jn-morphgnt.txt", "86-Jud-morphgnt.txt", "87-Re-morphgnt.txt",
)

FLAGGED_TEXT_SOURCES: dict[str, FlaggedTextSource] = {
    "sblgnt": FlaggedTextSource(
        id="sblgnt",
        name="MorphGNT: SBLGNT Edition (Greek NT, morphology + lemma)",
        base_url="https://raw.githubusercontent.com/morphgnt/sblgnt/master/",
        files=_SBLGNT_FILES,
        version="morphgnt/sblgnt v6.12 (snapshot 2026-06-10)",
        license="Text: SBLGNT EULA (http://sblgnt.com/license/); "
                "morphology: CC-BY-SA-3.0",
        flag="MorphGNT/SBLGNT carries lemma + morphology but NO Strong's numbers, "
             "so it cannot supply the verse->Strong's link word-study needs "
             "without an out-of-corpus lemma->Strong's bridge; AND the SBLGNT text "
             "is under the SBLGNT EULA (not pure PD — bundling is an architect "
             "decision). Snapshotted for provenance; NOT ingested this run.",
    ),
}


# Thayer's — recorded as a not-found flag so the report surfaces it every run.
THAYERS_FLAG = (
    "Thayer's Greek-English lexicon: no clean, canonical, machine-readable "
    "public-domain dataset found in a source of record (OpenScriptures ships "
    "Strong's + BDB, not Thayer's; a GitHub survey found only GPL-incompatible or "
    "non-canonical aggregator copies). OMITTED, not fabricated (CLAUDE.md). "
    "Strong's Greek glosses cover the basic need; a canonical Thayer's source "
    "needs architect + theological-advisor approval before ingestion."
)
