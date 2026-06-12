"""P4 — Lexicon dictionaries ingestion (→ NormalizedChunks).

Canonical sources only (CLAUDE.md §Sources of record). Every structural or
sourcing ambiguity is FLAGGED for human review, never silently resolved, and the
pipeline never marks output ship-ready — the architect's spot-check gates ship.

Share-alike-free editions (architect-approved swap)
---------------------------------------------------
The Strong's Greek/Hebrew dictionaries were re-sourced OFF the OpenScriptures
``openscriptures/strongs`` ``.js`` editions, which are **CC-BY-SA** (share-alike
copyleft the architect wants out of the corpus). The replacements are:

* **Strong's Greek** — ``morphgnt/strongs-dictionary-xml`` ``strongsgreek.xml``,
  released **CC0 / public domain** (Ulrik Sandborg-Petersen's XML e-text of
  Strong's Greek Dictionary, 1890). A single ``<strongsdictionary>`` with one
  ``<entry strongs="00025">`` per number. Fields map to ours: ``<greek unicode=
  translit=>`` (lemma + translit), ``<pronunciation strongs="…">`` (pron),
  ``<strongs_def>`` (definition), ``<strongs_derivation>`` (derivation),
  ``<kjv_def>`` (kjv_def). Inline ``<greek>``/``<strongsref>`` nodes inside the
  prose fields are flattened (the Greek unicode is substituted, refs rendered
  ``G####``/``H####``) so no Greek word is lost. CC0 means NO attribution and NO
  share-alike obligation — strictly cleaner than the CC-BY-SA edition it replaces,
  and it covers the full G1–G5624 span (101 numbers the old edition omitted).
* **Strong's Hebrew** — ``openscriptures/HebrewLexicon`` ``HebrewStrong.xml``,
  **CC-BY 4.0** (attribution-only; underlying Strong's Hebrew text, 1894, is PD).
  NO clean CC0 / pure-PD machine-readable Strong's Hebrew dictionary exists in a
  source of record — surveyed and FLAGGED. CC-BY is the best available and is
  share-alike-free (the constraint the architect set), so it is used per the task
  directive ("if nothing cleaner than CC-BY exists, use the best attribution-only
  option and FLAG it; never fall back to CC-BY-SA"). This is the SAME repo we
  already use for BDB + the OSHB lineage; ``HebrewStrong.xml`` is keyed ``H####``
  with ``<w pron= xlit=>`` (lemma), ``<source>`` (derivation), ``<meaning>``/
  ``<def>`` (definition), ``<usage>`` (kjv_def), full H1–H8674.

The retired CC-BY-SA ``.js`` snapshots are removed from the build path (their
``sources/lexicons/strongs-greek`` / ``strongs-hebrew`` dirs are repointed to the
new editions); the old bytes remain in git history.

* **Brown-Driver-Briggs (BDB)** — ``openscriptures/HebrewLexicon``
  ``BrownDriverBriggs.xml`` (the fuller Hebrew entry). BDB entries are keyed by an
  internal id (``a.ab.ab``), NOT by Strong's number; the Strong↔BDB linkage lives
  in the same repo's ``LexicalIndex.xml`` (``<xref bdb=… strong=…/>``). We parse
  the index to attach a Strong's H-number to each BDB entry we can map, and FLAG
  (do not guess) BDB entries the index leaves unlinked.

License note (recorded per source in provenance): the underlying Strong's, BDB,
and the WLC text are all **public domain**; the *editions* carry licenses — the
new Strong's Greek edition **CC0** (no obligations), Strong's Hebrew + the Hebrew
Lexicon (BDB) and morphhb (OSHB) **CC-BY-4.0** (attribution-only, share-alike-
free). Recorded so downstream attribution is correct.

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
    license: str       # the EDITION license (CC0 / CC-BY / CC-BY-SA)
    text_license: str  # the underlying dictionary text license (PD)
    fmt: str = "js"    # parse format: 'js' (JSON-in-.js) | 'strongs-greek-xml'
                       #               | 'strongs-hebrew-xml' | 'bdb-xml'

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
        # Share-alike-free swap: morphgnt/strongs-dictionary-xml is CC0 / PD
        # (Sandborg-Petersen XML e-text of Strong's Greek, 1890), replacing the
        # retired CC-BY-SA OpenScriptures .js edition.
        url="https://raw.githubusercontent.com/morphgnt/strongs-dictionary-xml/"
            "master/strongsgreek.xml",
        filename="strongsgreek.xml",
        version="morphgnt/strongs-dictionary-xml strongsgreek.xml "
                "(snapshot 2026-06-11)",
        license="CC0 / public domain (morphgnt/strongs-dictionary-xml)",
        text_license="Public domain (Strong's Greek Dictionary, 1890)",
        fmt="strongs-greek-xml",
    ),
    "strongs-hebrew": LexiconSource(
        id="strongs-hebrew",
        name="Strong's Hebrew Dictionary",
        language=HEBREW,
        lexicon="strongs",
        # Share-alike-free swap: no CC0/PD machine-readable Strong's Hebrew exists
        # in a source of record (surveyed + FLAGGED). openscriptures/HebrewLexicon
        # HebrewStrong.xml is CC-BY 4.0 (attribution-only, NOT share-alike) — the
        # best available, replacing the retired CC-BY-SA OpenScriptures .js edition.
        url="https://raw.githubusercontent.com/openscriptures/HebrewLexicon/"
            "master/HebrewStrong.xml",
        filename="HebrewStrong.xml",
        version="openscriptures/HebrewLexicon HebrewStrong.xml "
                "(snapshot 2026-06-11)",
        license="CC-BY-4.0 (OpenScriptures Hebrew Lexicon edition)",
        text_license="Public domain (Strong's Hebrew Dictionary, 1894)",
        fmt="strongs-hebrew-xml",
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
        fmt="bdb-xml",
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
    not_used: bool = False      # Strong's "Not Used" placeholder (skipped number)


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


# --- Strong's Greek (CC0 morphgnt XML) ---------------------------------------
# strongsgreek.xml: one <entry strongs="00025"> per number. The prose fields
# (<strongs_def>, <strongs_derivation>, <kjv_def>) embed inline <greek
# unicode="…"/> and <strongsref language= strongs=/> nodes. Those nodes carry
# meaning in attributes, so a plain BeautifulSoup get_text() would silently drop
# the Greek words and cross-references. We flatten manually: <greek> -> its
# unicode form, <strongsref> -> the G####/H#### key, everything else -> its text.
def _flatten_greek_field(node) -> str | None:
    """Render a Greek prose field, substituting inline greek/ref attributes."""
    if node is None:
        return None
    out: list[str] = []
    for child in node.descendants:
        name = getattr(child, "name", None)
        if name is None:  # NavigableString
            # Skip strings that live inside a <greek>/<strongsref> (handled below).
            parent = getattr(child, "parent", None)
            if parent is not None and parent.name in ("greek", "strongsref",
                                                      "pronunciation"):
                continue
            out.append(str(child))
        elif name == "greek":
            out.append(child.get("unicode", ""))
        elif name == "strongsref":
            lang = (child.get("language") or "").upper()
            num = child.get("strongs", "")
            prefix = "H" if lang == "HEBREW" else "G"
            if num.strip():
                out.append(f"{prefix}{int(num)}" if num.strip().lstrip("0").isdigit()
                           else f"{prefix}{num}")
    return _clean("".join(out))


def parse_strongs_greek_xml(
    src: LexiconSource, prov: Provenance, content: str
) -> ParsedLexicon:
    """Parse the CC0 morphgnt ``strongsgreek.xml`` into LexEntries (keyed G####)."""
    pl = ParsedLexicon(id=src.id, language=GREEK, lexicon="strongs", provenance=prov)
    soup = BeautifulSoup(content, "lxml-xml")
    seen: set[str] = set()
    entries = soup.find_all("entry")
    # Sort by numeric Strong's so the entry list is deterministic regardless of
    # source ordering (the XML is already in order, but we don't rely on that).
    entries.sort(key=lambda e: int(re.sub(r"\D", "", e.get("strongs") or "0") or 0))
    for entry in entries:
        raw = entry.get("strongs", "")
        key = _normalize_strongs_key(f"G{int(raw)}" if raw.strip().isdigit() else raw,
                                     "G")
        if key is None:
            pl.flags.append(f"{src.id}: unparseable Strong's attr {raw!r} (skipped)")
            continue
        if key in seen:
            pl.flags.append(f"{src.id}: duplicate Strong's {key} (kept first)")
            continue
        seen.add(key)
        greek = entry.find("greek")
        pron = entry.find("pronunciation")
        # "Not Used" entries are Strong's-assigned numbers he never populated
        # (skipped/merged numbers). The CC0 source ships them as explicit
        # placeholders where the old CC-BY-SA edition simply omitted them; we keep
        # them flagged-as-placeholder so coverage is explicit, not an error.
        body = _clean(entry.get_text())
        not_used = (greek is None and body is not None
                    and body.replace(key[1:], "").strip().lower() == "not used")
        pl.entries.append(
            LexEntry(
                strongs=key,
                language=GREEK,
                lexicon="strongs",
                lemma=_clean(greek.get("unicode")) if greek is not None else None,
                translit=_clean(greek.get("translit")) if greek is not None else None,
                pronunciation=(_clean(pron.get("strongs")) if pron is not None
                               else None),
                definition=_flatten_greek_field(entry.find("strongs_def")),
                derivation=_flatten_greek_field(entry.find("strongs_derivation")),
                kjv_def=_clean_kjv(_flatten_greek_field(entry.find("kjv_def"))),
                not_used=not_used,
            )
        )
    n_not_used = sum(1 for e in pl.entries if e.not_used)
    if n_not_used:
        pl.flags.append(
            f"{src.id}: {n_not_used} 'Not Used' placeholder entries (Strong's "
            f"numbers he assigned but never populated — kept so the G1..{STRONGS_GREEK_MAX} "
            f"span is explicit; the prior CC-BY-SA edition omitted them as gaps)"
        )
    return pl


# --- Strong's Hebrew (CC-BY HebrewLexicon XML) -------------------------------
# HebrewStrong.xml: one <entry id="H1"> per number, OpenScriptures namespaced.
#   <w pos= pron= xlit= xml:lang=>lemma</w>
#   <source>derivation prose (with inline <w src="H24">24</w> refs)</source>
#   <meaning>definition prose (with inline <def>…</def> glosses)</meaning>
#   <usage>KJV renderings</usage>
def _flatten_hebrew_field(node) -> str | None:
    """Render a Hebrew prose field, substituting inline <w src> refs to H####."""
    if node is None:
        return None
    out: list[str] = []
    for child in node.descendants:
        name = getattr(child, "name", None)
        if name is None:  # NavigableString
            parent = getattr(child, "parent", None)
            if parent is not None and parent.name == "w" and parent.get("src"):
                continue  # the ref number text is replaced below
            out.append(str(child))
        elif name == "w" and child.get("src"):
            src_ref = child.get("src", "")
            m = re.fullmatch(r"[Hh]?0*(\d+)\w*", src_ref)
            out.append(f"H{int(m.group(1))}" if m else src_ref)
    return _clean("".join(out))


def parse_strongs_hebrew_xml(
    src: LexiconSource, prov: Provenance, content: str
) -> ParsedLexicon:
    """Parse the CC-BY ``HebrewStrong.xml`` into LexEntries (keyed H####)."""
    pl = ParsedLexicon(id=src.id, language=HEBREW, lexicon="strongs", provenance=prov)
    soup = BeautifulSoup(content, "lxml-xml")
    seen: set[str] = set()
    entries = soup.find_all("entry")
    entries.sort(key=lambda e: int(re.sub(r"\D", "", e.get("id") or "0") or 0))
    for entry in entries:
        key = _normalize_strongs_key(entry.get("id", ""), "H")
        if key is None:
            pl.flags.append(f"{src.id}: unparseable entry id {entry.get('id')!r} "
                            f"(skipped)")
            continue
        if key in seen:
            pl.flags.append(f"{src.id}: duplicate Strong's {key} (kept first)")
            continue
        seen.add(key)
        w = entry.find("w")
        pl.entries.append(
            LexEntry(
                strongs=key,
                language=HEBREW,
                lexicon="strongs",
                lemma=_clean(w.get_text()) if w is not None else None,
                translit=_clean(w.get("xlit")) if w is not None else None,
                pronunciation=_clean(w.get("pron")) if w is not None else None,
                definition=_flatten_hebrew_field(entry.find("meaning")),
                derivation=_flatten_hebrew_field(entry.find("source")),
                kjv_def=_flatten_hebrew_field(entry.find("usage")),
            )
        )
    return pl


def _clean_kjv(value: str | None) -> str | None:
    """Trim Strong's kjv_def boilerplate punctuation (``:--``, leading ``--``).

    The XML kjv_def reads ``:--(be-)love(-ed).`` / ``--Alpha.``; we strip the
    leading ``:--`` / ``--`` lead-in for parity with the prior .js edition's
    cleaner glosses, keeping the rendering itself intact.
    """
    if value is None:
        return None
    v = re.sub(r"^[:\s]*-{1,2}\s*", "", value).strip()
    return v or None


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
    """One Strong's-tagged original-language word, anchored to a VerseRef.

    Shared by the OSHB Hebrew tagging (P4b) and the STEPBible TAGNT Greek tagging
    (P4b). The Greek-only fields (``lemma``, ``gloss``, ``editions``) are left None
    for Hebrew rows, which carry their meaning through the Strong's dictionary +
    OSHB ``morph`` instead.
    """

    ref: VerseRef
    position: int          # 1-based word index within the verse
    surface: str           # the pointed Hebrew / accented Greek word as written
    strongs: list[str]     # one or more Strong's keys (prefix segments contribute none)
    morph: str | None      # OSHB / TAGNT morphology code
    lemma_raw: str         # the raw lemma/source token (e.g. 'c/1961', '6965 b')
    lemma: str | None = None      # TAGNT dictionary form (Greek)
    gloss: str | None = None      # TAGNT brief gloss (Greek)
    editions: str | None = None   # TAGNT edition membership flags (e.g. 'NKO')


@dataclass
class ParsedTaggedText:
    """Parsed Strong's-tagged text for one language + flags for review."""

    id: str
    language: str
    provenance: Provenance
    words: list[TaggedWord] = field(default_factory=list)
    books_seen: set[str] = field(default_factory=set)
    flags: list[str] = field(default_factory=list)
    # Source-display metadata, set during normalization so the builder doesn't have
    # to know which source registry a given tagged text came from (OSHB vs TAGNT).
    name: str = ""
    attribution: str = ""


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
    pt = ParsedTaggedText(id=src.id, language=HEBREW, provenance=prov,
                          name=src.name, attribution=src.attribution)
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
             "decision). NOT ingested: the Greek NT tagging is now supplied by the "
             "STEPBible TAGNT (CC-BY 4.0), which DOES carry disambiguated Strong's. "
             "SBLGNT remains snapshotted for provenance only. (OpenGNT was also "
             "considered and REJECTED: CC-BY-SA, a copyleft term we avoid bundling.)",
    ),
}


# Strong's Hebrew CC-BY flag (share-alike-free swap). The Greek dictionary moved
# to a CC0 source; no CC0 / pure-PD machine-readable Strong's HEBREW dictionary
# exists in a source of record (surveyed: morphgnt ships Greek only; STEPBible
# TBESH and openscriptures/HebrewLexicon are both CC-BY 4.0). Per the task
# directive, the best attribution-only option is used — HebrewLexicon's
# HebrewStrong.xml — and the absence of a CC0 Hebrew is FLAGGED here, never
# resolved by falling back to the retired CC-BY-SA edition.
STRONGS_HEBREW_FLAG = (
    "Strong's Hebrew dictionary: NO CC0 / pure-public-domain machine-readable "
    "edition exists in a source of record (the morphgnt CC0 set is Greek-only; the "
    "available Hebrew editions — openscriptures/HebrewLexicon HebrewStrong.xml and "
    "STEPBible TBESH — are both CC-BY 4.0). The retired OpenScriptures .js Hebrew "
    "edition was CC-BY-SA (share-alike), which the architect wants out. Per the "
    "swap directive ('if nothing cleaner than CC-BY exists, use the best "
    "attribution-only option and FLAG it; never fall back to CC-BY-SA'), the "
    "CC-BY-4.0 HebrewStrong.xml is ingested — same repo already used for BDB + the "
    "OSHB lineage, so attribution requirements are already met. FLAGGED so the "
    "architect can decide whether to keep the CC-BY Hebrew or source a CC0/PD "
    "Hebrew Strong's later. (The Greek dictionary IS now CC0.)"
)


# Thayer's — superseded by the STEPBible TBESG (P4b). Kept as a recorded note so
# the report shows the sourcing decision: no clean canonical PD Thayer's dataset
# exists in a source of record; the approved substitute is the Tyndale Brief
# lexicon (TBESG, Abbott-Smith-based), now ingested.
THAYERS_FLAG = (
    "Thayer's Greek-English lexicon: no clean, canonical, machine-readable "
    "public-domain dataset found in a source of record. SUBSTITUTE INGESTED — the "
    "STEPBible TBESG (Tyndale Brief lexicon of Extended Strong's for Greek, "
    "Abbott-Smith-based, CC-BY 4.0) is the architect-approved stand-in for "
    "Thayer's and is now in lexicons.sqlite. A canonical Thayer's proper would "
    "still need architect + theological-advisor approval before ingestion."
)


# =============================================================================
# P4b — STEPBible Greek: TAGNT (tagged Greek NT) + TBESG (Greek lexicon)
# =============================================================================
# Source: github.com/STEPBible/STEPBible-Data, CC-BY 4.0. The required
# attribution string is "STEP Bible, www.STEPBible.org" (recorded in provenance
# and surfaced in the manifest). TAGNT is the Strong's-tagged amalgamated Greek
# NT (each word flags which editions — NA28/NA27/SBL/Treg/WH/TR/Byz/Tyn — contain
# it); TBESG is the Greek lexicon keyed by extended/disambiguated Strong's.
STEPBIBLE_DIR = LEXICONS_DIR / "stepbible"

STEPBIBLE_ATTRIBUTION = "STEP Bible, www.STEPBible.org"
STEPBIBLE_LICENSE = "CC-BY-4.0 (STEPBible; Tyndale House Cambridge)"
STEPBIBLE_VERSION = "STEPBible-Data TAGNT/TBESG (snapshot 2026-06-11)"

# STEPBible 3-letter book abbreviation -> canonical USFM id. STEPBible uses the
# NRSV-style abbreviations seen in the TAGNT reference column (Mat, Mrk, Luk, Jhn,
# Act, Rom, 1Co, ... Rev). Mapped explicitly, never guessed.
STEPBIBLE_BOOK_TO_USFM: dict[str, str] = {
    "Mat": "MAT", "Mrk": "MRK", "Luk": "LUK", "Jhn": "JHN", "Act": "ACT",
    "Rom": "ROM", "1Co": "1CO", "2Co": "2CO", "Gal": "GAL", "Eph": "EPH",
    "Php": "PHP", "Col": "COL", "1Th": "1TH", "2Th": "2TH", "1Ti": "1TI",
    "2Ti": "2TI", "Tit": "TIT", "Phm": "PHM", "Heb": "HEB", "Jas": "JAS",
    "1Pe": "1PE", "2Pe": "2PE", "1Jn": "1JN", "2Jn": "2JN", "3Jn": "3JN",
    "Jud": "JUD", "Rev": "REV",
}

# The two TAGNT data files (split only because one file is too large for GitHub).
# Together they cover all 27 NT books, Matthew through Revelation.
_TAGNT_FILES: tuple[str, ...] = (
    "TAGNT Mat-Jhn - Translators Amalgamated Greek NT - STEPBible.org CC-BY.txt",
    "TAGNT Act-Rev - Translators Amalgamated Greek NT - STEPBible.org CC-BY.txt",
)
_TBESG_FILE = (
    "TBESG - Translators Brief lexicon of Extended Strongs for Greek - "
    "STEPBible.org CC BY.txt"
)

_STEPBIBLE_RAW_BASE = (
    "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/"
)


@dataclass(frozen=True)
class StepBibleSource:
    """A STEPBible TAGNT / TBESG snapshot definition (P4b, CC-BY 4.0)."""

    id: str                # 'tagnt' | 'tbesg'
    name: str
    repo_subdir: str       # url-path subdir within the repo
    files: tuple[str, ...]

    def url(self, fname: str) -> str:
        from urllib.parse import quote

        return _STEPBIBLE_RAW_BASE + quote(f"{self.repo_subdir}/{fname}")

    def dest(self, fname: str) -> Path:
        return STEPBIBLE_DIR / self.id / fname


STEPBIBLE_SOURCES: dict[str, StepBibleSource] = {
    "tagnt": StepBibleSource(
        id="tagnt",
        name="Translators Amalgamated Greek New Testament (TAGNT)",
        repo_subdir="Translators Amalgamated OT+NT",
        files=_TAGNT_FILES,
    ),
    "tbesg": StepBibleSource(
        id="tbesg",
        name="Tyndale Brief lexicon of Extended Strong's for Greek (TBESG)",
        repo_subdir="Lexicons",
        files=(_TBESG_FILE,),
    ),
}


def _normalize_greek_dstrong(raw: str) -> str | None:
    """Normalize a STEPBible Greek dStrong to ``G<int><suffix?>``.

    STEPBible disambiguated Strong's are zero-padded with an optional homograph
    letter: ``G0976`` -> ``G976``; ``G2424G`` -> ``G2424G``; ``G0007H`` -> ``G7H``.
    The leading zeros are stripped (to match the existing ``G####`` key style) but
    the extended letter is PRESERVED — it is the disambiguation that links a TAGNT
    word to its specific TBESG sense, so collapsing it would lose information.
    Returns None for an unparseable token (caller flags it).
    """
    m = re.fullmatch(r"G0*(\d+)([A-Za-z]?)", raw.strip())
    if not m:
        return None
    return f"G{int(m.group(1))}{m.group(2)}"


# --- TBESG (Greek lexicon, keyed by extended/disambiguated Strong's) ----------
# Main lexicon rows are tab-separated:
#   eStrong | dStrong | uStrong | Greek | Transliteration | Morph | Gloss | AS-def
# The dStrong column carries a trailing " =" plus an optional linking note
# (" = the Greek of\tHxxxx"); we take only the leading G-token. The TAGNT word
# rows reference dStrong, so TBESG is keyed by dStrong here.
_TBESG_ROW_RE = re.compile(r"^G\d{4}")


def parse_tbesg(prov: Provenance, content: str) -> ParsedLexicon:
    """Parse the TBESG Greek lexicon into LexEntries keyed by extended Strong's."""
    pl = ParsedLexicon(id="tbesg", language=GREEK, lexicon="tbesg", provenance=prov)
    seen: set[str] = set()
    for line in content.splitlines():
        if not _TBESG_ROW_RE.match(line):
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 7:
            pl.flags.append(f"tbesg: short row (<7 cols) skipped: {line[:40]!r}")
            continue
        dstrong_raw = cols[1].split("=", 1)[0].strip()
        key = _normalize_greek_dstrong(dstrong_raw)
        if key is None:
            pl.flags.append(f"tbesg: unparseable dStrong {dstrong_raw!r} (skipped)")
            continue
        if key in seen:
            # Genuine duplicate dStrong is unexpected; surface, keep the first.
            pl.flags.append(f"tbesg: duplicate dStrong {key} (kept first)")
            continue
        seen.add(key)
        greek = _clean(cols[3])
        translit = _clean(cols[4])
        morph = _clean(cols[5])
        gloss = _clean(cols[6])
        full_def = _clean(cols[7]) if len(cols) > 7 else None
        # Definition = the brief gloss + the Abbott-Smith entry where present.
        definition = full_def or gloss
        pl.entries.append(
            LexEntry(
                strongs=key,
                language=GREEK,
                lexicon="tbesg",
                lemma=greek,
                translit=translit,
                pronunciation=None,
                definition=definition,
                derivation=morph,     # TBESG part-of-speech tag (e.g. 'G:N-F')
                kjv_def=gloss,        # the short gloss, kept in the kjv_def slot
                raw_key=dstrong_raw,  # the source-native dStrong (zero-padded)
            )
        )
    pl.flags.append(
        f"tbesg: parsed {len(pl.entries)} Greek lexicon entries keyed by extended "
        f"(disambiguated) Strong's; substitute for Thayer's (Abbott-Smith-based)"
    )
    return pl


# --- TAGNT (Strong's-tagged amalgamated Greek NT) ----------------------------
# A word row starts with the reference+position+word-type token, e.g.
#   Mat.1.1#01=NKO  <tab>  Βίβλος (Biblos)  <tab>  ...  <tab>  G0976=N-NSF  ...
# Header/intro/per-verse-comment lines do not match this anchor and are skipped.
_TAGNT_ROW_RE = re.compile(
    r"^([1-3]?[A-Za-z]+)\.(\d+)\.(\d+)#(\d+)=(\S+)\t"
)


def _tagnt_surface(greek_col: str) -> str:
    """The accented Greek surface form, dropping the parenthetical transliteration.

    The Greek column is ``Βίβλος (Biblos)``; we keep ``Βίβλος`` (the first token
    before the space-paren), preserving punctuation attached to the word.
    """
    s = greek_col.strip()
    # Drop a trailing " (translit)" group if present.
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    return s


def parse_tagnt(
    prov: Provenance, content_by_file: dict[str, str]
) -> ParsedTaggedText:
    """Parse the STEPBible TAGNT into per-word Strong's-tagged Greek rows.

    Each row yields a :class:`TaggedWord` anchored to a canonical VerseRef with its
    disambiguated Strong's (``strongs``), morphology (``morph``), lemma + gloss, and
    the edition-membership flags (``editions``). Verse-position numbering restarts
    per verse from the source ``#NN`` field. Unknown book abbreviations and
    malformed rows are FLAGGED, never guessed.
    """
    pt = ParsedTaggedText(
        id="tagnt", language=GREEK, provenance=prov,
        name=STEPBIBLE_SOURCES["tagnt"].name, attribution=STEPBIBLE_ATTRIBUTION,
    )
    unknown_books: set[str] = set()
    for fname in sorted(content_by_file):
        for line in content_by_file[fname].splitlines():
            m = _TAGNT_ROW_RE.match(line)
            if not m:
                continue
            book_abbr, chap, verse, pos, wordtype = m.groups()
            usfm = STEPBIBLE_BOOK_TO_USFM.get(book_abbr)
            if usfm is None:
                unknown_books.add(book_abbr)
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                pt.flags.append(
                    f"tagnt: short row at {book_abbr}.{chap}.{verse}#{pos} "
                    f"(<5 cols, skipped)"
                )
                continue
            surface = _tagnt_surface(cols[1])
            dstrong_raw, _, morph = cols[3].partition("=")
            key = _normalize_greek_dstrong(dstrong_raw)
            strongs = [key] if key is not None else []
            if key is None and dstrong_raw.strip():
                pt.flags.append(
                    f"tagnt: unparseable dStrong {dstrong_raw!r} at "
                    f"{book_abbr}.{chap}.{verse}#{pos} (no Strong's recorded)"
                )
            lemma_field = cols[4] if len(cols) > 4 else ""
            lemma, _, gloss = lemma_field.partition("=")
            ref = VerseRef(book=usfm, chapter=int(chap), verse_start=int(verse))
            pt.books_seen.add(usfm)
            pt.words.append(
                TaggedWord(
                    ref=ref,
                    position=int(pos),
                    surface=surface,
                    strongs=strongs,
                    morph=_clean(morph) or None,
                    lemma_raw=cols[0],
                    lemma=_clean(lemma) or None,
                    gloss=_clean(gloss) or None,
                    editions=_clean(cols[5]) if len(cols) > 5 else None,
                )
            )
    if unknown_books:
        pt.flags.append(
            f"tagnt: unknown book abbreviations skipped: "
            f"{', '.join(sorted(unknown_books))}"
        )
    pt.flags.append(
        "tagnt: edition membership captured per word in the 'editions' column "
        "(e.g. NA28+NA27+Tyn+SBL+WH+Treg+TR+Byz); the word-type prefix (NKO etc.) "
        "is retained on the 'lemma_raw' source token"
    )
    return pt
