"""Focused USFM parser for the markers the Bible pipeline needs.

Deliberately dependency-light and narrow: it handles exactly the markers present
in our four canonical snapshots (BSB bereanbible.com; KJV/ASV/WEB eBible.org),
not the full USFM 3 spec. Anything unexpected is surfaced (via :class:`ParseWarning`)
rather than silently dropped.

What it extracts per file:
  * ``\\id``      — book id (validated against the canonical 66-book table)
  * ``\\c``       — chapter number
  * ``\\v``       — verse number, including bridges like ``\\v 15-16``
  * verse text   — with character/footnote markup stripped to clean prose
  * ``\\wj``      — words-of-Christ spans, preserved as character offsets so the
                   app can render red letters without re-parsing

Markup handled (stripped from display text, recorded where meaningful):
  * footnotes ``\\f ... \\f*`` and cross-ref notes ``\\x ... \\x*`` — removed
  * inline cross-references ``\\ref ...\\ref*`` — keep the display text
  * word-level ``\\w word|strong="G1234"\\w*`` and nested ``\\+w ...\\+w*`` — keep word
  * added words ``\\add ...\\add*`` / ``\\+add`` — keep word (no italics in plain text)
  * ``\\nd`` (name of God), ``\\qs`` (Selah), ``\\wj`` and other char markers — keep text
  * pilcrow ``¶`` paragraph marks — removed
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .books import CANON, OMITTED_VARIANT_SET


@dataclass
class WjSpan:
    """A words-of-Christ span as [start, end) character offsets into verse text."""

    start: int
    end: int


@dataclass
class ParsedVerse:
    book: str
    chapter: int
    verse_start: int
    verse_end: int          # == verse_start for single verses
    text: str
    wj_spans: list[WjSpan] = field(default_factory=list)
    # Plain text of a source textual footnote carried *on this verse*, if any.
    # Populated only for verses whose body text is empty (critical-text
    # omissions where the source hangs an "ancient authorities insert…" note on
    # the empty verse — ASV and WEB do this). Best-effort; NULL otherwise.
    source_note: str | None = None
    # Hebrew Psalm superscription (``\d``) text carried on this verse, if any. In
    # the BSB the superscription is numbered as verse 1 (``\d \v 1 …``); we keep
    # the superscription prose here (so the app can render it distinctly) while the
    # verse's ``text`` holds the body line(s). NULL when there is no superscription.
    superscription: str | None = None
    # True when this verse OPENS a new prose paragraph (a ``\p``-family marker
    # immediately precedes its ``\v``). Mid-verse paragraph breaks are not
    # captured (see the parser note on ``pending_para``).
    para_start: bool = False

    @property
    def is_bridge(self) -> bool:
        return self.verse_end != self.verse_start

    @property
    def has_red_letter(self) -> bool:
        return bool(self.wj_spans)


@dataclass
class ParsedBook:
    book: str
    verses: list[ParsedVerse]
    warnings: list[str] = field(default_factory=list)
    # BSB only: omitted-verse attributions recovered from preceding-verse
    # footnotes via embedded ``\fv N\fv*`` markers, keyed by (chapter, verse).
    # Empty for translations that render the omitted row directly. See
    # :func:`extract_bsb_omission_notes`.
    omission_notes: dict[tuple[int, int], str] = field(default_factory=dict)
    # Section headings (``\s``/``\s1``..``\s3``) attached to the verse they
    # precede: ``(chapter, verse_start, text)`` in document order. Only the BSB
    # carries a real heading apparatus (3,096 headings); KJV/ASV/WEB have none
    # or only stray artifacts, which are kept verbatim (never silently judged).
    headings: list[tuple[int, int, str]] = field(default_factory=list)


# --- regexes -----------------------------------------------------------------

# A footnote / cross-ref note: \f ... \f*  or  \x ... \x*  (non-greedy, no nesting
# of f/x within f/x in our sources). Removed entirely from display text.
_NOTE_RE = re.compile(r"\\(?:f|x)\b.*?\\(?:f|x)\*", re.DOTALL)

# A *footnote only* (\f ... \f*), used to recover the note text on an omitted
# verse. Captures the inner body so we can render it as the source_note.
_FOOTNOTE_RE = re.compile(r"\\f\b\s*(.*?)\\f\*", re.DOTALL)

# Footnote sub-markers we strip to recover plain note prose. \fr is the
# verse-reference echo (e.g. "17:21"); the leading "+" / "-" is the caller. The
# \ft (text) and \fqa (alternate reading) runs carry the human-readable body.
_FOOTNOTE_CALLER_RE = re.compile(r"^\s*[-+?]\s*")
_FOOTNOTE_FR_RE = re.compile(r"\\fr\b\s*\S+\s*")
_FOOTNOTE_FV_RE = re.compile(r"\\fv\b\s*\d+\s*\\fv\*")  # embedded verse marker
_FOOTNOTE_TAG_RE = re.compile(r"\\\+?[a-z]+\d?\*?")     # \ft \fqa \fk etc.

# \ref display|LINK\ref*  -> keep the display portion before the pipe.
_REF_RE = re.compile(r"\\ref\s+([^|\\]*?)(?:\|[^\\]*?)?\\ref\*", re.DOTALL)

# \w word|strong="..."\w*  and \+w ...\+w*  -> keep the surface word before '|'.
_WORD_RE = re.compile(r"\\\+?w\s+([^|\\]*?)(?:\|[^\\]*?)?\\\+?w\*", re.DOTALL)

# Any remaining *closing* char marker like \add* \nd* \+add* \qs* \wj* -> drop tag.
_CLOSE_TAG_RE = re.compile(r"\\\+?[a-z]+\d?\*")

# Any remaining *opening* inline char marker like \add \nd \qs \+add (with the
# trailing space it introduces) -> drop tag, keep following text.
_OPEN_TAG_RE = re.compile(r"\\\+?[a-z]+\d?\b ?")

# Verse number token: "16", "15-16", "15,16", "2a".
_VERSE_NUM_RE = re.compile(r"^(\d+)(?:[-,](\d+))?[a-z]?$")

# Paragraph / structural markers whose *content* we keep as verse text but whose
# tag we drop (poetry lines, paragraphs). They appear at line starts.
_PARA_TAGS = {"p", "m", "pi", "pi1", "pi2", "q", "q1", "q2", "q3", "q4", "qr",
              "qc", "qm", "qm1", "qm2", "li", "li1", "li2", "pc", "pmo", "pm",
              "pmc", "pmr", "nb", "b", "tr", "th1", "th2", "tc1", "tc2"}

# Section headings (``\s``-family): captured as structural data and attached to
# the verse they precede (Rank 8a). Handled BEFORE the drop-line set below.
_HEADING_TAGS = {"s", "s1", "s2", "s3"}

# Prose-paragraph openers: a verse whose ``\v`` immediately follows one of these
# OPENS a new paragraph (``ParsedVerse.para_start``). Poetry (``\q*``) and table
# markers are content layout, not prose paragraphs, and are excluded.
_PARA_START_TAGS = {"p", "m", "pi", "pi1", "pi2", "pc", "pmo", "pm", "pmc", "nb"}

# Markers whose *whole line* is discarded (titles, references, intro).
# NOTE: ``\d`` (descriptive title / Hebrew Psalm superscription) is handled
# SEPARATELY below — not here — because the BSB numbers the superscription as
# ``\d \v 1 …`` (the superscription carries the verse-1 marker). Dropping the whole
# ``\d`` line would discard BSB's verse 1 entirely (both the superscription text and
# the first body line that follows). We instead drop only the superscription *text*
# and keep processing any ``\v`` on the line, so BSB's verse 1 survives.
# ``\s``-family headings are no longer dropped (see _HEADING_TAGS).
_DROP_LINE_TAGS = {"id", "usfm", "ide", "h", "toc1", "toc2", "toc3", "toca1",
                   "toca2", "toca3", "mt", "mt1", "mt2", "mt3", "mte", "ms",
                   "ms1", "ms2", "mr", "sr", "r", "rq",
                   "sp", "cl", "cp", "cd", "rem", "sts", "imt", "is",
                   "ip", "im", "io", "io1", "io2", "iot", "ie", "iex", "qa",
                   "periph"}


def _clean_inline(text: str) -> str:
    """Strip inline markup from a run of verse text, returning clean prose."""
    text = _NOTE_RE.sub("", text)
    text = _REF_RE.sub(lambda m: m.group(1), text)
    text = _WORD_RE.sub(lambda m: m.group(1), text)
    text = _CLOSE_TAG_RE.sub("", text)
    text = _OPEN_TAG_RE.sub("", text)
    text = text.replace("¶", " ")        # pilcrow
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([,.;:!?’”])", r"\1", text)  # tidy space before punctuation
    return text.strip()


def _extract_footnote_text(raw: str) -> str | None:
    """Recover the plain-text body of the *first* footnote in ``raw``.

    Used only for omitted/empty verses, where the source hangs an
    "ancient authorities insert…" note on the empty verse. Strips the caller,
    the ``\\fr`` reference echo, embedded ``\\fv`` verse markers, ``\\ref``
    links, and footnote sub-tags (``\\ft``/``\\fqa``), leaving readable prose.
    Returns ``None`` when there is no footnote.
    """
    m = _FOOTNOTE_RE.search(raw)
    if not m:
        return None
    body = m.group(1)
    body = _FOOTNOTE_CALLER_RE.sub("", body)
    body = _FOOTNOTE_FR_RE.sub("", body)
    body = _FOOTNOTE_FV_RE.sub("", body)
    body = _REF_RE.sub(lambda mm: mm.group(1), body)
    body = _WORD_RE.sub(lambda mm: mm.group(1), body)
    body = _FOOTNOTE_TAG_RE.sub("", body)
    body = body.replace("¶", " ")
    body = unicodedata.normalize("NFC", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" ([,.;:!?’”])", r"\1", body)
    body = body.strip()
    return body or None


# --- BSB omitted-verse footnote attribution ---------------------------------
#
# BSB drops the omitted verse's row entirely and carries its explanatory note on
# the PRECEDING verse's footnote, with an embedded ``\fv N\fv*`` marker naming the
# omitted verse, e.g. Matt 18:11's note lives inside 18:10's footnote:
#
#   \f + \fr 18:10 \ft BYZ and TR include \fqa \fv 11\fv*For the Son of Man came
#   to save the lost\ft ; see \ref Luke 19:10|LUK 19:10\ref*.\f*
#
# We recover, per footnote, the text that immediately follows each ``\fv N\fv*``
# marker (the omitted verse's own wording) up to the next footnote sub-marker, and
# key it by (chapter, N). Chapter comes from the footnote's ``\fr`` reference echo.
# This never fabricates: a verse with no recoverable ``\fv`` segment is simply
# absent from the returned map (callers leave source_note NULL and flag it).

# A whole footnote, captured so we can scan its body for \fv markers.
_BSB_FOOTNOTE_RE = re.compile(r"\\f\b(.*?)\\f\*", re.DOTALL)
# The \fr reference echo, e.g. "18:10" or "9:43" — chapter is the part before ":".
_BSB_FR_REF_RE = re.compile(r"\\fr\b\s*(\d+):\d+")
# An embedded omitted-verse marker, capturing the verse number it names.
_BSB_FV_MARKER_RE = re.compile(r"\\fv\b\s*(\d+)\s*\\fv\*")
# Boundary that ends one \fv segment: the next footnote sub-marker or the close.
_BSB_FV_STOP_RE = re.compile(r"\\(?:fv|ft|fqa|fk|fr|fp|fl)\b|\\f\*")


def _clean_footnote_segment(seg: str) -> str | None:
    """Clean a recovered ``\\fv`` segment into readable verse prose, or None."""
    seg = _REF_RE.sub(lambda mm: mm.group(1), seg)
    seg = _WORD_RE.sub(lambda mm: mm.group(1), seg)
    seg = _FOOTNOTE_TAG_RE.sub("", seg)
    seg = seg.replace("¶", " ")
    seg = unicodedata.normalize("NFC", seg)
    seg = re.sub(r"[ \t]+", " ", seg)
    seg = re.sub(r" ([,.;:!?’”])", r"\1", seg)
    # Leading ellipsis/connector noise ("...", "…") sometimes precedes the text
    # when the source elides into a longer variant; trim it but keep real prose.
    seg = seg.strip().lstrip(".… ").strip()
    return seg or None


def extract_bsb_omission_notes(content: str) -> dict[tuple[int, int], str]:
    """Harvest BSB omitted-verse attributions from a raw USFM book.

    Returns ``{(chapter, verse): note_text}`` for every omitted verse whose text
    the BSB carries via an embedded ``\\fv N\\fv*`` marker on a preceding verse's
    footnote. Only verses with a genuinely recoverable segment appear; nothing is
    fabricated. A footnote may name several omitted verses (e.g. Acts 24:7-8);
    each ``\\fv`` marker is mapped independently.
    """
    out: dict[tuple[int, int], str] = {}
    for fm in _BSB_FOOTNOTE_RE.finditer(content):
        body = fm.group(1)
        if "\\fv" not in body:
            continue
        fr = _BSB_FR_REF_RE.search(body)
        if not fr:
            continue
        chapter = int(fr.group(1))
        for vm in _BSB_FV_MARKER_RE.finditer(body):
            vnum = int(vm.group(1))
            rest = body[vm.end():]
            stop = _BSB_FV_STOP_RE.search(rest)
            seg = rest[:stop.start()] if stop else rest
            note = _clean_footnote_segment(seg)
            if note:
                out[(chapter, vnum)] = note
    return out


def _extract_wj(raw: str) -> tuple[str, list[WjSpan]]:
    """Resolve \\wj ... \\wj* spans within a *raw* verse string.

    Returns clean display text plus the char offsets of words-of-Christ within it.
    We clean wj-segments and non-wj-segments identically so offsets line up with
    the final cleaned text.
    """
    if "\\wj" not in raw:
        return _clean_inline(raw), []

    out: list[str] = []
    spans: list[WjSpan] = []
    pos = 0
    cursor = 0  # offset into the assembled clean text
    pattern = re.compile(r"\\wj\b ?(.*?)\\wj\*", re.DOTALL)
    for m in pattern.finditer(raw):
        before = _clean_inline(raw[pos:m.start()])
        if before:
            if out and not out[-1].endswith(" ") and not before.startswith(" "):
                out.append(" ")
                cursor += 1
            out.append(before)
            cursor += len(before)
        inner = _clean_inline(m.group(1))
        if inner:
            if out and not out[-1].endswith(" "):
                out.append(" ")
                cursor += 1
            start = cursor
            out.append(inner)
            cursor += len(inner)
            spans.append(WjSpan(start=start, end=cursor))
        pos = m.end()
    tail = _clean_inline(raw[pos:])
    if tail:
        if out and not out[-1].endswith(" ") and not tail.startswith(" "):
            out.append(" ")
            cursor += 1
        out.append(tail)

    full = "".join(out)
    # Re-tidy whitespace without disturbing the span math: collapse only is safe
    # because _clean_inline already collapsed each segment and we inserted single
    # spaces between them.
    return full.strip(), [s for s in spans if s.end > s.start]


def parse_usfm(content: str) -> ParsedBook:
    """Parse one USFM book file into a :class:`ParsedBook`.

    Raises ``ValueError`` only on a missing/invalid ``\\id`` book code. Other
    anomalies are accumulated as warnings on the returned book.
    """
    book_id: str | None = None
    warnings: list[str] = []

    # First pass: find the \id book code.
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("\\id "):
            book_id = line[4:].strip().split()[0].upper() if len(line) > 4 else None
            break
    if not book_id:
        raise ValueError("USFM file has no \\id marker")
    if book_id not in CANON:
        raise ValueError(f"\\id {book_id!r} is not in the 66-book canon")

    verses: list[ParsedVerse] = []
    headings: list[tuple[int, int, str]] = []
    chapter = 0
    cur_v_start: int | None = None
    cur_v_end: int | None = None
    cur_buf: list[str] = []
    cur_para = False
    # Superscription (\d) text waiting to attach to the next verse it numbers. In
    # the BSB the superscription IS verse 1 (``\d \v 1 …``); we hold its prose here
    # and attach it to that verse while keeping the verse's body text separate.
    pending_superscription: list[str] = []
    # Section-heading text (\s-family) waiting to attach to the NEXT verse.
    pending_heading: list[str] = []
    # A \p-family marker was seen with no verse text after it — the NEXT verse
    # opens a prose paragraph. A \p carrying mid-verse continuation text is a
    # mid-verse break and deliberately does NOT flag the next verse.
    pending_para = False

    def flush() -> None:
        nonlocal cur_v_start, cur_v_end, cur_buf, pending_superscription, cur_para
        if cur_v_start is None:
            cur_buf = []
            return
        raw = " ".join(p for p in cur_buf if p)
        text, spans = _extract_wj(raw)
        # An omitted verse is one whose body cleaned to nothing. Where the source
        # hangs a textual footnote on that empty verse (ASV/WEB), recover it.
        note = _extract_footnote_text(raw) if not text else None
        sup = _clean_inline(" ".join(p for p in pending_superscription if p)) \
            if pending_superscription else None
        verses.append(ParsedVerse(
            book=book_id, chapter=chapter,
            verse_start=cur_v_start, verse_end=cur_v_end or cur_v_start,
            text=text, wj_spans=spans, source_note=note,
            superscription=sup or None,
            para_start=cur_para,
        ))
        cur_v_start = cur_v_end = None
        cur_buf = []
        cur_para = False
        pending_superscription = []

    # Set when a \d superscription opened a verse whose inline text is the
    # superscription itself (BSB ``\d \v 1 …``); the NEXT \v's inline text routes to
    # pending_superscription, and the body that follows attaches to that verse.
    expecting_superscription = False

    # Tokenise by marker. USFM verse text can span multiple physical lines, and
    # multiple \v can sit on one \p line, so we work marker-by-marker.
    # Split keeping the markers: each piece starts with a backslash marker.
    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith("\\"):
            # Continuation of current verse text (rare in our sources).
            if cur_v_start is not None:
                cur_buf.append(s)
            continue

        # A line may contain several markers; split on the marker boundary while
        # preserving order. We scan for \c, \v and para/drop tags.
        # Find all marker positions.
        idx = 0
        while idx < len(s):
            if s[idx] != "\\":
                # text belonging to the marker we're inside — handled below
                idx += 1
                continue
            # read marker name
            m = re.match(r"\\(\+?[a-z]+\d?)\*?", s[idx:])
            if not m:
                idx += 1
                continue
            tag = m.group(1).lstrip("+")
            after = idx + m.end()

            if tag == "c":
                flush()
                num_m = re.match(r"\s*(\d+)", s[after:])
                if num_m:
                    chapter = int(num_m.group(1))
                # rest of line after the number is usually nothing
                break  # \c lines carry no verse text we need
            elif tag == "v":
                flush()
                # verse token + the verse text up to the next \v or \c on this line
                rest = s[after:].lstrip()
                tok_m = re.match(r"(\S+)\s?(.*)$", rest, re.DOTALL)
                if not tok_m:
                    break
                token, text_after = tok_m.group(1), tok_m.group(2)
                vm = _VERSE_NUM_RE.match(token)
                if not vm:
                    warnings.append(f"{book_id} {chapter}: unparseable verse token {token!r}")
                    break
                cur_v_start = int(vm.group(1))
                cur_v_end = int(vm.group(2)) if vm.group(2) else cur_v_start
                # Consume any pending structural markers for this verse: a
                # section heading attaches here; a \p opener flags para_start.
                if pending_heading:
                    htext = _clean_inline(" ".join(pending_heading))
                    if htext:
                        headings.append((chapter, cur_v_start, htext))
                    pending_heading = []
                cur_para = pending_para
                pending_para = False
                # If a \d superscription opened this verse, its inline text is the
                # superscription, not the body — route it to the superscription
                # buffer. The body line(s) follow on later \q markers with no \v and
                # land in cur_buf via the para handler.
                target = pending_superscription if expecting_superscription \
                    else cur_buf
                expecting_superscription = False
                # The remaining text on the line up to the next \v / \c belongs here.
                # Find next \v or \c.
                nxt = re.search(r"\\[cv]\b", text_after)
                if nxt:
                    target.append(text_after[:nxt.start()])
                    # continue scanning from that marker
                    s = text_after[nxt.start():]
                    idx = 0
                    continue
                else:
                    target.append(text_after)
                    break
            elif tag == "d":
                # Descriptive title / Hebrew Psalm superscription. We KEEP the
                # superscription prose (as verse metadata) but never as body text.
                # Two source shapes:
                #   * BSB NUMBERS it:  ``\d \v 1 A Psalm of David…`` then the verse-1
                #     body line follows on a later ``\q`` (no ``\v``). The ``\v 1`` is
                #     ON THIS LINE — route its inline text to the superscription and
                #     keep scanning so verse 1 (its body) still exists.
                #   * KJV/ASV/WEB leave it UNNUMBERED: ``\d Michtam of David.`` on its
                #     own line, with ``\v 1`` on the NEXT line carrying the real body.
                #     Here we must NOT mark the next verse as superscription — only
                #     capture the \d text and attach it to whatever verse follows.
                # The discriminator is whether a ``\v`` sits on THIS \d line.
                rest = s[after:]
                nxt = re.search(r"\\[cv]\b", rest)
                if nxt and re.match(r"\\v\b", rest[nxt.start():]):
                    # Numbered superscription: the \v on this line carries it.
                    expecting_superscription = True
                    s = rest[nxt.start():]
                    idx = 0
                    continue
                # Unnumbered superscription (its body verse follows on a later line):
                # capture the \d prose; do NOT consume the next verse's text.
                pending_superscription.append(rest if not nxt else rest[:nxt.start()])
                if nxt:
                    s = rest[nxt.start():]
                    idx = 0
                    continue
                break
            elif tag in _HEADING_TAGS:
                # Section heading: capture its text (up to any \c/\v on the same
                # line) and attach it to the NEXT verse that opens. A heading also
                # implies the next verse starts a new unit of prose.
                rest = s[after:]
                nxt = re.search(r"\\[cv]\b", rest)
                pending_heading.append(rest[:nxt.start()] if nxt else rest)
                pending_para = True
                if nxt:
                    s = rest[nxt.start():]
                    idx = 0
                    continue
                break
            elif tag in _DROP_LINE_TAGS:
                break  # discard whole line
            elif tag in _PARA_TAGS:
                # paragraph/poetry marker: text after it belongs to current verse
                rest = s[after:]
                nxt = re.search(r"\\[cv]\b", rest)
                seg = rest[:nxt.start()] if nxt else rest
                if tag in _PARA_START_TAGS and not seg.strip():
                    # Empty prose-paragraph opener: the NEXT verse starts a
                    # paragraph. (A \p with trailing text is a mid-verse break.)
                    pending_para = True
                if cur_v_start is not None:
                    cur_buf.append(seg)
                if nxt:
                    s = rest[nxt.start():]
                    idx = 0
                    continue
                break
            else:
                # An inline char marker at line start with no preceding \p (rare):
                # treat remainder as verse text.
                rest = s[idx:]
                if cur_v_start is not None:
                    cur_buf.append(rest)
                break

    flush()

    # Sanity warnings: empty verses. Known critical-text omissions are expected
    # to be empty (they become omitted=1 rows downstream), so don't warn on them —
    # only an *unexpected* empty verse is a parser anomaly worth surfacing.
    for v in verses:
        if not v.text and (book_id, v.chapter, v.verse_start) not in OMITTED_VARIANT_SET:
            warnings.append(f"{book_id} {v.chapter}:{v.verse_start} parsed empty")

    # Harvest BSB-style omitted-verse attributions from preceding-verse footnotes.
    # Translation-agnostic: translations that render the verse directly simply
    # produce no \fv-marked omission notes, so the map is empty and unused.
    omission_notes = extract_bsb_omission_notes(content)

    return ParsedBook(
        book=book_id, verses=verses, warnings=warnings,
        omission_notes=omission_notes, headings=headings,
    )
