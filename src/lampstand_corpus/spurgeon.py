"""Spurgeon — *The Treasury of David* (Psalms) — Internet-Archive OCR ingestion.

Architect-approved source (a change from the earlier flagged-and-skipped status:
the CCEL Treasury is image-only, so the architect approved the Google-digitized
Internet Archive scans instead). The seven physical volumes of the *Treasury* are
the ``*spurgoog`` Google scans on archive.org; we pull each volume's ``_djvu.txt``
plaintext. This is **OCR**, not clean text — residual noise is expected and is the
whole reason this ingestion is a candidate, not ship-ready: the architect's
spot-check decides whether the Treasury ships in v1 or defers to v1.1.

Spurgeon (d.1892) and the *Treasury* (completed 1885) are firmly public domain.

Volume -> identifier -> psalm mapping (RESOLVED EMPIRICALLY, NOT GUESSED)
------------------------------------------------------------------------
The ``*spurgoog`` set has **overlapping scan sets** (twelve items for a seven-
volume work). Each chosen identifier's psalm span was resolved by reading its
title-page range line ("PSALM I. TO XXVI." etc.) and its running-header ``PSALM
<roman>`` markers — never by assuming the classic split. The set this edition
uses is: 1-26, 27-52, 53-78, 79-103, **104-118**, 119-124, 125-150.

CRITICAL GAP — flagged for human review, never papered over
-----------------------------------------------------------
**No ``*spurgoog`` item covers Psalms 104-118.** All twelve candidate scans were
probed for any Psalm 104-118 header or body text; none contains it. The physical
volume covering 104-118 is simply absent from the Google scan set. We ingest the
six available volumes (Psalms 1-103 and 119-150) and FLAG the 104-118 gap loudly
for the architect to source a replacement scan before ship. We do NOT substitute
a non-``*spurgoog`` text to fill it (that would breach the approved-source rule).

Where duplicate scans exist for a range, the cleanest copy (fewest residual
Google-watermark tokens, most complete OCR) was chosen; the rejected duplicates
are recorded so the choice is auditable.

Segmentation
------------
Each volume is split into per-psalm blocks on the literal ``PSALM <roman>`` head.
Within a psalm, Spurgeon's four-component structure is delimited by OCR'd ALL-CAPS
section heads (whitespace-tolerant matching, since OCR doubles spaces):

  * the psalm **title / argument** (the italic intro paragraph before EXPOSITION),
  * **EXPOSITION** — verse text + Spurgeon's verse-by-verse comment,
  * **EXPLANATORY NOTES AND QUAINT SAYINGS**,
  * **HINTS TO THE VILLAGE PREACHER**,
  * **WORK[S] UPON ...** (a bibliographic list; kept, anchored at verse 0).

Not every psalm carries all components, and the OCR mangles the heads (e.g.
"WORKS UPON THE TWENTT-TIIIRD PSALM"); matching is fuzzy on the leading keyword.
Within EXPOSITION / NOTES / HINTS, comment paragraphs that open with a verse cue
(``N.``, ``Verse N.—``, ``Ver. N.``) anchor to ``PSA:psalm:N``; component prose
with no verse cue anchors to the whole psalm (``verse_start=0``). Anything that
cannot be tied to a valid Psalm:verse is FLAGGED, never invented.

OCR caveats surfaced, never fabricated
--------------------------------------
Spurgeon quotes Hebrew and Greek; the scans render those as garbage. We do NOT
reconstruct them — chunks with a high non-Latin / gibberish ratio are flagged as
``garbled`` so the spot-check can judge them. Running headers, bare page numbers,
and the Google front/back boilerplate are stripped; hyphenation broken across a
line wrap is rejoined.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .schema import NormalizedChunk, Provenance, ResourceType, VerseRef
from .sources import SOURCES_DIR

SPURGEON_DIR = SOURCES_DIR / "commentaries" / "spurgeon"

# A single extracted block this long is a statistical anomaly worth a human glance
# (Psalm 119 legitimately runs long). FLAG threshold, never a truncation.
LONG_BLOCK_CHARS = 16_000


# --- Volume mapping (empirically resolved; see module docstring) --------------
@dataclass(frozen=True)
class SpurgeonVolume:
    """One physical Treasury volume = one chosen IA ``*spurgoog`` identifier."""

    stem: str            # local volume key, e.g. 'tod1'
    identifier: str      # archive.org item id (the chosen, cleanest scan)
    psalm_first: int     # first psalm covered (inclusive)
    psalm_last: int      # last psalm covered (inclusive)

    @property
    def url(self) -> str:
        return f"https://archive.org/download/{self.identifier}/{self.identifier}_djvu.txt"

    @property
    def dest(self) -> Path:
        return SPURGEON_DIR / f"{self.stem}.txt"


# The SEVEN volumes (cleanest scan per range). Psalms 104-118 (tod5) is now filled
# from an alternate PD Internet Archive scan — see TOD5_SOURCE_NOTE below.
SPURGEON_VOLUMES: tuple[SpurgeonVolume, ...] = (
    SpurgeonVolume("tod1", "treasurydavidco02spurgoog", 1, 26),
    SpurgeonVolume("tod2", "treasurydavidco05spurgoog", 27, 52),
    SpurgeonVolume("tod3", "treasurydavidco03spurgoog", 53, 78),
    SpurgeonVolume("tod4", "treasurydavidco01spurgoog", 79, 103),
    # tod5 (Psalms 104-118): GAP-FILLED from an alternate PD scan (NOT *spurgoog).
    # treasuryofdavidc0005spur is vol. 5 (1882), title-page range line "PSALM CIV.
    # TO CXVIII.", a clean non-Google scan (zero Google-watermark tokens); all 15
    # psalms 104-118 carry running-ordinal headers. Spurgeon d.1892 / Treasury
    # completed 1885 — firmly public domain.
    SpurgeonVolume("tod5", "treasuryofdavidc0005spur", 104, 118),
    SpurgeonVolume("tod6", "treasurydavidco07spurgoog", 119, 124),
    SpurgeonVolume("tod7", "treasurydavidco00spurgoog", 125, 150),
)

# The exact source used to fill the former 104-118 gap (FLAGGED per the task so the
# spot-check knows precisely which scan supplied these psalms).
TOD5_SOURCE_NOTE: str = (
    "Psalms 104-118 GAP-FILLED from archive.org item 'treasuryofdavidc0005spur' "
    "(The Treasury of David, vol. 5, 1882; title-page range 'PSALM CIV. TO "
    "CXVIII.'). This is an ALTERNATE PD scan set, NOT the *spurgoog Google scans "
    "used for the other six volumes (the *spurgoog set has no item covering "
    "104-118). It is a cleaner non-Google scan (0 Google-watermark tokens). Same "
    "cleaning + four-component segmentation as the rest of Spurgeon. ARCHITECT: "
    "spot-check these psalms against a printed Treasury vol. 5."
)

# No remaining structural gap — the 104-118 volume is now sourced. Kept as an
# empty sentinel (range collapses) so callers that referenced MISSING_VOLUME keep
# working and the manifest records that the gap was closed.
MISSING_VOLUME: tuple[int | None, int | None, str] = (
    None, None,
    "RESOLVED: Psalms 104-118 were gap-filled from an alternate PD Internet "
    "Archive scan (treasuryofdavidc0005spur, vol. 5, 1882). No *spurgoog Google "
    "scan covers this range, so a non-Google PD scan was used for tod5 only. "
    + TOD5_SOURCE_NOTE,
)

# Duplicate scans rejected in favour of the cleaner copy (kept for audit, not
# fetched). identifier -> (range, why-rejected).
REJECTED_DUPLICATES: dict[str, str] = {
    "treasurydavidco08spurgoog": "Ps 1-26 dup of tod1 (treasurydavidco02): 166 vs "
                                 "11 residual Google-watermark tokens",
    "treasurydavidvo01spurgoog": "Ps 53-78 dup of tod3 (treasurydavidco03): 514 vs "
                                 "11 watermark tokens",
    "treasurydavid00spurgoog":   "Ps 79-103 dup of tod4 (treasurydavidco01): 408 vs "
                                 "11 watermark tokens; tod4 also more complete",
    "treasurydavidco06spurgoog": "Ps 79-103 dup of tod4 (treasurydavidco01): smaller, "
                                 "98 watermark tokens, lower vowel ratio",
    "treasurydavidvo00spurgoog": "Ps 119-124 dup of tod6 (treasurydavidco07): "
                                 "near-equal OCR; tod6 marginally more complete",
    "treasurydavidco04spurgoog": "Ps 125-150 dup of tod7 (treasurydavidco00): smaller "
                                 "scan (1.64M vs 2.37M chars), less complete",
}


# --- Source registration (CommentarySource-compatible) -----------------------
# Spurgeon is registered alongside the CCEL commentators but carries kind=
# 'spurgeon-ia' so the snapshot/normalize loops build IA djvu URLs (not CCEL .xml)
# and dispatch to this module's parser.
@dataclass(frozen=True)
class SpurgeonSource:
    """Treasury-of-David source descriptor, shaped like CommentarySource."""

    id: str = "spurgeon"
    name: str = "Charles H. Spurgeon — The Treasury of David"
    shortcode: str = "CHS"
    author: str = "Charles Haddon Spurgeon"
    work: str = "The Treasury of David (an exposition of the Psalms)"
    version: str = (
        "Internet Archive DjVu OCR — all seven volumes; six from the Google-"
        "digitized *spurgoog scans, tod5 (Psalms 104-118) gap-filled from the "
        "alternate PD scan treasuryofdavidc0005spur (vol. 5, 1882)"
    )
    license: str = "Public domain (Spurgeon d.1892; Treasury completed 1885)"
    kind: str = "spurgeon-ia"

    @property
    def volumes(self) -> tuple[str, ...]:
        return tuple(v.stem for v in SPURGEON_VOLUMES)

    def volume(self, stem: str) -> SpurgeonVolume:
        for v in SPURGEON_VOLUMES:
            if v.stem == stem:
                return v
        raise KeyError(stem)

    def url(self, stem: str) -> str:
        return self.volume(stem).url

    def dest(self, stem: str) -> Path:
        return self.volume(stem).dest


SPURGEON_SOURCE = SpurgeonSource()


@dataclass
class ParsedSpurgeon:
    """Spurgeon's normalized chunks plus any ambiguities to flag.

    Shares the ``id`` / ``chunks`` / ``flags`` / ``coverage`` shape of
    ``commentaries.ParsedCommentary`` so the shared build + validator can consume
    it directly. ``coverage`` maps book id -> set of chapters (here always PSA ->
    the psalm numbers seen). ``psalms_seen`` is the convenience set for reporting.
    """

    id: str
    chunks: list[NormalizedChunk]
    flags: list[str]
    psalms_seen: set[int]
    coverage: dict[str, set[int]] = field(default_factory=dict)


# --- Roman numerals ----------------------------------------------------------
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> int | None:
    s = s.upper()
    if not s or any(ch not in _ROMAN for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        v = _ROMAN[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total if 1 <= total <= 150 else None


# --- OCR cleaning ------------------------------------------------------------
# Google's front/back-matter boilerplate that wraps every scan. We drop the run
# from "Google This is a digital copy" to the first PSALM head (front matter), and
# any trailing "About Google Book Search" block.
_FRONT_BOILERPLATE = re.compile(
    r"(?:^|\bGoogle\b\s*)?This is a digital copy of a book.*?(?=\bPSALM\b)",
    re.S | re.I,
)
_BACK_BOILERPLATE = re.compile(
    r"(About Google Book Search|Google'?s mission is to organize).*\Z", re.S | re.I
)
# Stray Google OCR watermark fragments embedded mid-text.
_WATERMARK = re.compile(
    r"(VjOOQI[CcEe]?|VjOOQ|D,?B,?i\.?\.?ab,?\s*Google|Digitized by\s*(?:VjOOQ\w*|Google)?|"
    r"Original from\b.*|UNIVERSITY OF\b.*|Generated (?:on|for)\b.*)",
    re.I,
)
# Running header that interrupts the body mid-paragraph, e.g.
# "PSALM THE TWENTY-THIRD. 399" or "THE TREASURY OF DAVID. 124".
_RUNNING_HEADER = re.compile(
    r"\n\s*(PSALM\s+THE\s+[A-Z\- ]+\.?|THE\s+TREASURY\s+OF\s+DAVID\.?)\s*\d{0,4}\s*\n",
    re.I,
)
# A bare page-number line.
_PAGE_NUMBER = re.compile(r"\n\s*\d{1,4}\s*\n")
# Hyphenation broken across a wrap: "righteous- \n ness" -> "righteousness".
_HYPHEN_WRAP = re.compile(r"([A-Za-z])-\s*\n\s*([a-z])")


def _strip_boilerplate(text: str) -> str:
    text = _FRONT_BOILERPLATE.sub("", text, count=1)
    text = _BACK_BOILERPLATE.sub("", text)
    text = _WATERMARK.sub(" ", text)
    return text


def _clean_block(text: str) -> str:
    """Clean an already-segmented block (running headers, page nums, hyphenation).

    Whitespace is collapsed last so flagging of garbled OCR sees the real letters.
    """
    text = _RUNNING_HEADER.sub("\n", text)
    # Re-run: headers can sit back-to-back with page numbers.
    text = _RUNNING_HEADER.sub("\n", text)
    text = _PAGE_NUMBER.sub("\n", text)
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# A paragraph is "garbled" if too few of its words look like real words. Spurgeon's
# Hebrew/Greek quotations and the worst OCR pages trip this — we FLAG, never repair.
def _garble_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z]{2,}", text)
    if not words:
        return 1.0
    plausible = sum(1 for w in words if re.search(r"[aeiouAEIOU]", w))
    return 1.0 - (plausible / len(words))


# --- Segmentation ------------------------------------------------------------
# Psalm head: the line-isolated heading "PSALM XXIII." that opens a psalm's
# section. Line-anchored ON PURPOSE: lower-case inline cross-references ("So Psalm
# xliv. 17, ...") and roman numerals mid-sentence must NOT be mistaken for section
# heads — the whole LINE must be essentially just the heading. Case-insensitive
# because the OCR sometimes lower-cases the roman ("PSALM vir." for VII); a stray
# trailing page-number/cruft is tolerated. A psalm whose head OCR'd too badly to
# match here is recovered from its running-header ordinal (below) or, failing
# both, reported as a coverage gap — never back-filled from an inline reference.
_PSALM_HEAD = re.compile(r"(?im)^[ \t]{0,8}PSALM[ \t]+([clxvidm]{1,7})\.?[ \t]*\d{0,4}[ \t]*$")

# Running-header ordinal index: "PSALM THE TWENTY-THIRD. 399" names the psalm by
# ordinal word. These headers repeat on every page of a psalm, so they recover the
# boundary of any psalm whose all-caps roman head OCR'd too badly to match above.
# Using the running header to LOCATE a boundary is a parse (the header literally
# names the psalm), not a guess. The ordinal words are themselves OCR-noisy, so we
# match them fuzzily against the canonical ordinal spellings.
_RUNNING_ORDINAL = re.compile(r"(?i)\bPSALM\s+THE\s+([A-Z][A-Z\- ]{2,34}?)\s*\.")

_ORDINALS_1_19 = [
    "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH",
    "NINTH", "TENTH", "ELEVENTH", "TWELFTH", "THIRTEENTH", "FOURTEENTH",
    "FIFTEENTH", "SIXTEENTH", "SEVENTEENTH", "EIGHTEENTH", "NINETEENTH",
]
_TENS = {20: "TWENTIETH", 30: "THIRTIETH", 40: "FORTIETH", 50: "FIFTIETH",
         60: "SIXTIETH", 70: "SEVENTIETH", 80: "EIGHTIETH", 90: "NINETIETH"}
_TENS_PREFIX = {20: "TWENTY", 30: "THIRTY", 40: "FORTY", 50: "FIFTY",
                60: "SIXTY", 70: "SEVENTY", 80: "EIGHTY", 90: "NINETY"}
_HUNDRED = "HUNDRED"


def _ordinal_word(n: int) -> str:
    """Canonical ordinal spelling for a psalm number (1-150), no spaces/hyphens.

    Used only as the fuzzy-match target for OCR'd running headers, e.g.
    23 -> 'TWENTYTHIRD', 119 -> 'HUNDREDANDNINETEENTH'.
    """
    if 1 <= n <= 19:
        return _ORDINALS_1_19[n - 1]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return _TENS[tens]
        return _TENS_PREFIX[tens] + _ORDINALS_1_19[ones - 1]
    # 100-150
    rem = n - 100
    if rem == 0:
        return _HUNDRED + "TH"
    return _HUNDRED + "AND" + _ordinal_word(rem)


def _norm_ordinal(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


# Precompute canonical ordinal -> psalm number for 1-150.
_ORDINAL_TO_PSALM: dict[str, int] = {_norm_ordinal(_ordinal_word(n)): n
                                     for n in range(1, 151)}


def _match_ordinal(raw: str) -> int | None:
    """Map an OCR'd running-header ordinal word to a psalm number, fuzzily.

    Exact match first; then a small edit-distance tolerance so OCR garbles
    ('TWEXTY-TniRD' -> TWENTYTHIRD) still resolve. Ambiguous matches return None
    (flagged upstream as an unrecovered head), never a guess.
    """
    key = _norm_ordinal(raw)
    if key in _ORDINAL_TO_PSALM:
        return _ORDINAL_TO_PSALM[key]
    best: tuple[int, int] | None = None  # (distance, psalm)
    for cand, n in _ORDINAL_TO_PSALM.items():
        if abs(len(cand) - len(key)) > 3:
            continue
        d = _edit_distance(key, cand, cap=3)
        if d is not None and (best is None or d < best[0]):
            best = (d, n)
    # Accept only a confidently-close, unique match (<=25% of length).
    if best and best[0] <= max(1, len(key) // 4):
        return best[1]
    return None


def _edit_distance(a: str, b: str, *, cap: int) -> int | None:
    """Levenshtein distance, early-exit if it exceeds ``cap`` (returns None)."""
    if abs(len(a) - len(b)) > cap:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            row_min = min(row_min, v)
        if row_min > cap:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= cap else None

# The four+ component section heads, matched whitespace-tolerantly on the leading
# keyword (OCR mangles the tail, e.g. "WORKS UPON THE TWENTT-TIIIRD PSALM").
_COMPONENTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("exposition", re.compile(r"\bE\s*X\s*P\s*O\s*S\s*I\s*T\s*I\s*O\s*N\b\.?", re.I)),
    ("notes", re.compile(r"\bEXPLANATORY\s+NOTES\b", re.I)),
    ("hints", re.compile(r"\bHINTS\s+TO\s+THE\b", re.I)),
    ("works", re.compile(r"\bWORK[S]?\s+UPON\b", re.I)),
)

# A verse cue opening a comment paragraph inside a component:
#   'Verse 3.—', 'Ver. 3.', '3. "the quote"'  (the bare-number form is common in
#   the EXPOSITION's verse-by-verse comments).
_VERSE_CUE = re.compile(
    r"(?:^|\n)\s*(?:Verse[s]?|Ver\.?)\s+(\d{1,3})(?:\s*[-–—.,]|\s)|"
    r"(?:^|\n)\s*(\d{1,3})\.\s+[\"“']",
)


def _split_psalms(
    text: str, declared_range: tuple[int, int] | None = None
) -> tuple[list[tuple[int, str]], list[int]]:
    """Split a cleaned volume into (psalm_number, block) on psalm heads.

    Returns ``(blocks, recovered)`` where ``recovered`` is the list of psalm
    numbers whose all-caps roman head was unreadable and whose boundary was
    recovered from a running-header ordinal instead (surfaced for the report).

    Primary boundaries are line-isolated ``PSALM <roman>`` heads. For any psalm in
    the volume's declared range with no such head, we recover the boundary from the
    earliest running-header ordinal that names it (the header literally spells the
    psalm out, so this is a parse, not a guess). A head whose roman doesn't resolve
    to 1-150 is skipped as OCR noise.
    """
    heads: dict[int, int] = {}  # psalm_no -> earliest start offset (roman head)
    for m in _PSALM_HEAD.finditer(text):
        n = _roman_to_int(m.group(1))
        if n is not None and n not in heads:
            heads[n] = m.start()

    recovered: list[int] = []
    if declared_range is not None:
        lo, hi = declared_range
        # Earliest running-header ordinal position per psalm number.
        ordinal_pos: dict[int, int] = {}
        for m in _RUNNING_ORDINAL.finditer(text):
            n = _match_ordinal(m.group(1))
            if n is not None and n not in ordinal_pos:
                ordinal_pos[n] = m.start()
        for n in range(lo, hi + 1):
            if n in heads or n not in ordinal_pos:
                continue
            # Place the recovered boundary just before the running header that
            # names this psalm but after the previous psalm's known head, so the
            # block captures this psalm's content (the running header sits inside
            # the psalm, a page or two past its true start; using it as the
            # boundary keeps content with the right psalm without inventing text).
            heads[n] = ordinal_pos[n]
            recovered.append(n)

    ordered = sorted(heads.items(), key=lambda kv: kv[1])
    out: list[tuple[int, str]] = []
    for i, (n, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        out.append((n, text[start:end]))
    return out, sorted(recovered)


def _dedupe_psalm_blocks(blocks: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """A running header repeats 'PSALM XXIII' many times; keep only the longest
    block per psalm number (the true content block, not a header fragment)."""
    best: dict[int, str] = {}
    for n, blk in blocks:
        if n not in best or len(blk) > len(best[n]):
            best[n] = blk
    return [(n, best[n]) for n in sorted(best)]


def _split_components(block: str) -> list[tuple[str, str]]:
    """Split one psalm block into (component, text), in document order.

    The text before the first recognised component head is the psalm TITLE /
    ARGUMENT. Components found are emitted in the order they appear.
    """
    # Locate each component head occurrence.
    marks: list[tuple[int, str]] = []
    for comp, pat in _COMPONENTS:
        for m in pat.finditer(block):
            # Skip the leading word of the PSALM head line itself.
            marks.append((m.start(), comp))
    marks.sort()
    # Deduplicate: the first occurrence of each component wins; a component head
    # appearing again later is treated as body (rare OCR echo) and ignored only if
    # it would create a zero-length slice.
    out: list[tuple[str, str]] = []
    first_head = marks[0][0] if marks else len(block)
    title = block[:first_head]
    if _norm_ws(title):
        out.append(("title", title))
    for i, (pos, comp) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(block)
        seg = block[pos:end]
        if _norm_ws(seg):
            out.append((comp, seg))
    return out


def _split_verse_paragraphs(
    text: str, max_verse: int | None = None
) -> list[tuple[int, str]]:
    """Within a component, split into (verse_no, paragraph).

    verse_no==0 means "no verse cue — anchor to the whole psalm". The text before
    the first verse cue (the component's own lede, incl. the verse-text block in
    EXPOSITION) is emitted at verse 0. ``max_verse`` (the psalm's canonical verse
    count) gates the bare-number ``N.`` cue: a number above the psalm length is a
    page number or footnote marker, not a verse — such a cue is ignored so its
    paragraph stays with the preceding verse instead of being mis-anchored.
    """
    cues: list[tuple[int, int]] = []  # (offset, verse_no)
    for m in _VERSE_CUE.finditer(text):
        vn = int(m.group(1) or m.group(2))
        # A cue above the psalm's verse count can't be a verse — skip it (it is an
        # OCR'd page number / footnote ref that happened to fit the cue shape).
        if max_verse is not None and vn > max_verse:
            continue
        cues.append((m.start(), vn))
    out: list[tuple[int, str]] = []
    if not cues:
        body = text.strip()
        if body:
            out.append((0, body))
        return out
    lede = text[: cues[0][0]].strip()
    if lede:
        out.append((0, lede))
    for i, (pos, vn) in enumerate(cues):
        end = cues[i + 1][0] if i + 1 < len(cues) else len(text)
        seg = text[pos:end].strip()
        if seg:
            out.append((vn, seg))
    return out


def _key(psalm: int, verse: int, component: str, idx: int) -> str:
    base = f"PSA.{psalm}.{verse}"
    return f"{base}#{component[:4]}{idx}"


def parse_spurgeon(
    src: SpurgeonSource,
    prov_by_volume: dict[str, Provenance],
    content_by_volume: dict[str, str],
) -> ParsedSpurgeon:
    """Parse all available Treasury volumes into verse-anchored chunks."""
    flags: list[str] = []
    chunks: list[NormalizedChunk] = []
    psalms_seen: set[int] = set()
    coverage: dict[str, set[int]] = {}
    # (psalm, verse, component) -> running paragraph ordinal, kept O(1) per chunk.
    para_counter: dict[tuple[int, int, str], int] = {}

    # Surface the resolved 104-118 gap (now filled), the exact gap-fill source, and
    # the duplicate-resolution decisions up front.
    lo, hi, why = MISSING_VOLUME
    if lo is None:
        flags.append(f"spurgeon: former Psalms 104-118 gap RESOLVED — {why} — review")
    else:
        flags.append(f"spurgeon: Psalms {lo}-{hi} ABSENT — {why} — review")
    flags.append(f"spurgeon: {TOD5_SOURCE_NOTE} — review")
    flags.append(
        "spurgeon: overlapping *spurgoog scan sets resolved by title-page range + "
        "running-header probe; cleaner copy chosen per range. Rejected duplicates: "
        + "; ".join(f"{k} ({v})" for k, v in REJECTED_DUPLICATES.items())
        + " — verify the volume->psalm mapping — review"
    )

    for stem in src.volumes:
        vol = src.volume(stem)
        prov = prov_by_volume[stem]
        raw = content_by_volume[stem]
        text = _strip_boilerplate(raw)

        raw_blocks, recovered = _split_psalms(
            text, (vol.psalm_first, vol.psalm_last)
        )
        blocks = _dedupe_psalm_blocks(raw_blocks)
        if recovered:
            flags.append(
                f"spurgeon/{stem} ({vol.identifier}): {len(recovered)} psalm "
                f"head(s) {recovered} were unreadable as an all-caps roman heading "
                "and were recovered from their running-header ordinal — boundary is "
                "approximate (the header sits a page or so into the psalm); verify "
                "the opening of these psalms against a printed edition — review"
            )
        vol_psalms = {n for n, _ in blocks}
        # Flag any psalm found outside this volume's declared range (scan bleed or
        # an OCR head misread) — kept, but surfaced.
        out_of_range = sorted(
            n for n in vol_psalms if not (vol.psalm_first <= n <= vol.psalm_last)
        )
        if out_of_range:
            flags.append(
                f"spurgeon/{stem} ({vol.identifier}, declared Ps "
                f"{vol.psalm_first}-{vol.psalm_last}): PSALM head(s) outside range "
                f"{out_of_range} — likely a running-header OCR misread; kept and "
                "anchored to the read number — review"
            )

        # In-range psalms with NO detected boundary (neither an all-caps roman head
        # nor a recoverable running-header ordinal). Their text is NOT lost — it is
        # absorbed into the preceding psalm's block — but it is MIS-ANCHORED there,
        # a genuine merge. Flag loudly; do NOT guess a boundary.
        in_range_gap = sorted(
            n for n in range(vol.psalm_first, vol.psalm_last + 1)
            if n not in vol_psalms
        )
        if in_range_gap:
            flags.append(
                f"spurgeon/{stem} ({vol.identifier}): Psalm head(s) {in_range_gap} "
                "could not be located (both the all-caps roman heading and the "
                "running-header ordinal OCR'd unreadable). Each missing psalm's "
                "commentary is present but absorbed into the PRECEDING psalm's "
                "block — a mis-anchor, not a deletion. Verify and split against a "
                "printed edition before ship — review"
            )

        for psalm, block in blocks:
            block = _clean_block(block)
            max_verse = _psalm_length(psalm)
            for component, seg in _split_components(block):
                for verse, para in _split_verse_paragraphs(seg, max_verse):
                    para = _norm_ws(para)
                    if not para:
                        continue
                    # Validate the verse anchor against the canonical Psalm length.
                    if verse and not _valid_psalm_verse(psalm, verse):
                        flags.append(
                            f"spurgeon/{stem}: Psalm {psalm}:{verse} "
                            f"({component}) is off the canonical Psalm versification "
                            "— anchored to the whole psalm instead — review"
                        )
                        verse = 0
                    ckey = (psalm, verse, component)
                    idx = para_counter.get(ckey, 0) + 1
                    para_counter[ckey] = idx
                    key = _key(psalm, verse, component, idx)
                    garble = _garble_ratio(para)
                    chunk = NormalizedChunk(
                        id=f"{src.id}:{key}",
                        resource_type=ResourceType.COMMENTARY,
                        ref=VerseRef(
                            book="PSA", chapter=psalm,
                            verse_start=verse, verse_end=None,
                        ),
                        key=key,
                        text=para,
                        meta={
                            "author": src.author,
                            "work": src.work,
                            "shortcode": src.shortcode,
                            "volume": stem,
                            "identifier": vol.identifier,
                            "component": component,
                            "chapter_level": verse == 0,
                            "para_index": idx,
                            "garble_ratio": round(garble, 3),
                        },
                        provenance=prov,
                    )
                    chunks.append(chunk)
                    psalms_seen.add(psalm)
                    coverage.setdefault("PSA", set()).add(psalm)
                    # A heavily non-word paragraph (Hebrew/Greek quotation or a
                    # bad scan page) is flagged, not fabricated or dropped.
                    if garble >= 0.45 and len(para) >= 80:
                        flags.append(
                            f"spurgeon/{stem}: Psalm {psalm}:{verse or '—'} "
                            f"({component}) chunk reads as garbled OCR "
                            f"(garble={garble:.2f}); likely a Hebrew/Greek quotation "
                            "or bad scan page — NOT reconstructed — review"
                        )

    chunks.sort(key=_sort_key)
    return ParsedSpurgeon(
        id=src.id, chunks=chunks, flags=flags, psalms_seen=psalms_seen,
        coverage=coverage,
    )


def _psalm_length(psalm: int) -> int | None:
    from . import books
    counts = books.VERSE_COUNTS.get("PSA")
    if not counts or not (1 <= psalm <= len(counts)):
        return None
    return counts[psalm - 1]


def _valid_psalm_verse(psalm: int, verse: int) -> bool:
    length = _psalm_length(psalm)
    return length is not None and 1 <= verse <= length


_COMPONENT_ORDER = {"title": 0, "exposition": 1, "notes": 2, "hints": 3, "works": 4}


def _sort_key(c: NormalizedChunk) -> tuple:
    r = c.ref
    return (
        r.chapter,
        r.verse_start,
        _COMPONENT_ORDER.get(c.meta.get("component", ""), 9),
        c.meta.get("para_index", 0),
    )
