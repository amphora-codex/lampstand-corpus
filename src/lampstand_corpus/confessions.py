"""P2 — Confessions & catechisms ingestion (CCEL ThML → NormalizedChunk).

Canonical sources only (CLAUDE.md §Sources of record). The documents converge on
the normalized schema at **section / Q&A granularity**: one chunk per confession
section (``chapter.section``) or per catechism question (the question number).
Scripture proof-texts the source provides are captured as :class:`VerseRef`s in
each chunk's ``meta``.

Sourcing decisions (every CCEL structural ambiguity is FLAGGED for human review,
never silently resolved — CLAUDE.md):

* **WSC** (CCEL ``westminster1.xml``) and **WLC** (``westminster2.xml``) — clean
  per-question ThML; the original Q/A text. No embedded proof-texts in this CCEL
  edition (left empty, not fabricated).
* **WCF** (CCEL ``westminster3.xml``) — this CCEL edition is the **modern American
  PCUS/UPCUSA text**, whose chapters are RENUMBERED and which adds two modern
  chapters (Of the Holy Spirit, Of the Gospel) plus denomination-specific marriage
  chapters and inline ``[PCUS …] [UPCUSA …]`` variant brackets. We recover the
  **33 original chapters** using CCEL's own ``Chapter N (orig)`` mapping, skip the
  two added modern chapters and the duplicate denominational marriage chapter, and
  preserve any variant brackets verbatim in the text (never guessing which reading
  to keep). The edition mismatch is flagged: the architect must decide whether to
  re-source the original 1646/1647 WCF (e.g. Schaff) before ship.
* **Heidelberg** (CCEL ``heidelberg.xml``) — 129 Q across 52 Lord's Days, with
  machine-readable ``<scripRef osisRef=…>`` proof-texts we capture as VerseRefs.
* **Canons of Dort** (CCEL ``canonsofdort.xml``) — a clean, dedicated ENGLISH ThML
  edition (not the Schaff bilingual table). ``div1 type="section"`` per Head of
  Doctrine; ``div2 type="subsection"`` splitting each head's positive Articles from
  its "Rejection of Errors"; bodies marked ``ARTICLE N`` / ``PARAGRAPH N`` (both
  restart per head). We key positive articles ``headN.aM`` and rejection paragraphs
  ``headN.rM`` so every chunk maps to (head, kind, number) without collision.

Flagged-and-SKIPPED (genuinely not cleanly available from a canonical source —
surfaced for human sourcing, never substituted with a non-canonical text):

* **1689 London Baptist Confession** — CCEL/Schaff (*Creeds of Christendom* III,
  "Baptist Confession of 1688 / Philadelphia Confession") prints only the *editorial
  differences* from the Westminster/Savoy text, not the full 32 chapters (its body
  is prose paragraphs that read "It is a slight modification of the Westminster
  Confession… In Chapter XX… In the Chapter Of the Church…"). No standalone clean
  full-text 1689 ThML exists on CCEL. Reconstructing the 32 chapters would mean
  splicing Westminster text with Schaff's noted deltas — that is guesswork, which
  CLAUDE.md forbids. FLAGGED for the architect to approve a canonical full-text
  source (e.g. the official 1689 text via the Westminster-Standards-style repo)
  before inclusion.
* **Belgic Confession** — present in Schaff Creeds III only as a 2-column
  French/Latin–English TABLE (54 tables, English in the right cell) whose 37
  articles are fragmented mid-sentence across rows/tables with OCR artifacts
  ("A rt. II.", "G uy de B rès"). Positionally the English column is isolable but
  clean per-article reconstruction needs heuristic stitching + OCR repair — a guess,
  not a parse. FLAGGED; re-source a clean canonical English Belgic before inclusion.
* **Apostles'/Nicene/Athanasian creeds** — tentative per spec §6.2. CCEL lists them
  under the Creeds subject but exposes only rendered HTML pages (no standalone
  ThML download; the ``/ccel/anonymous/{apostles,nicene,athanasian}.xml`` endpoints
  return the HTML viewer, not ThML), and in Schaff they sit amid many historical
  variant creeds with Greek/Latin parallels. Not cleanly available → skipped per
  the "include only if clean" rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from .schema import NormalizedChunk, Provenance, ResourceType, VerseRef
from .sources import SOURCES_DIR

CONFESSIONS_DIR = SOURCES_DIR / "confessions"


# --- known totals (validation spine) -----------------------------------------
# Per-document expected counts (CLAUDE.md Part B scope). Deviations are FLAGGED.
EXPECTED_COUNTS: dict[str, int] = {
    "wcf": 33,    # chapters
    "wlc": 196,   # questions
    "wsc": 107,   # questions
    "heidelberg": 129,  # questions
    "dort": 5,    # heads of doctrine (the validation spine for Dort)
}
HEIDELBERG_LORDS_DAYS = 52

# Canons of Dort: the canonical text presents FIVE heads of doctrine, but the
# third and fourth are published as a single combined head ("Third and Fourth
# Heads of Doctrine"), giving four numbered head-sections plus a Conclusion. Each
# head carries positive Articles and a "Rejection of Errors" paragraph series.
# We key heads by a short slug derived from the CCEL section title rather than a
# bare integer, so the combined head is represented faithfully (not split/guessed).
# (head_slug -> (expected positive articles, expected rejection paragraphs))
DORT_HEAD_COUNTS: dict[str, tuple[int, int]] = {
    "1": (18, 9),     # First Head of Doctrine
    "2": (9, 7),      # Second Head of Doctrine
    "3-4": (17, 9),   # Third and Fourth Heads of Doctrine (combined)
    "5": (15, 9),     # Fifth Head of Doctrine
}
DORT_HEADS = 5  # canonical count of heads of doctrine


# --- OSIS book code -> USFM book id ------------------------------------------
# Maps the OSIS ids in CCEL <scripRef osisRef="Bible:Rom.3.20"> to our canonical
# USFM spine (books.ORDER). Covers the full 66-book canon for robustness.
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


@dataclass(frozen=True)
class ConfessionSource:
    id: str
    name: str
    shortcode: str        # citation shortcode prefix, e.g. "WCF", "HC"
    url: str
    filename: str
    version: str
    license: str

    @property
    def dest(self) -> Path:
        return CONFESSIONS_DIR / self.id / self.filename


# CCEL ThML, public-domain. Committed under sources/confessions/<id>/ via git-lfs.
CONFESSION_SOURCES: dict[str, ConfessionSource] = {
    "wcf": ConfessionSource(
        id="wcf", name="Westminster Confession of Faith", shortcode="WCF",
        url="https://ccel.org/ccel/anonymous/westminster3.xml",
        filename="westminster3.xml",
        version="CCEL ThML (PCUS/UPCUSA American edition; original 33 chapters recovered)",
        license="Public domain (CCEL)",
    ),
    "wlc": ConfessionSource(
        id="wlc", name="Westminster Larger Catechism", shortcode="WLC",
        url="https://ccel.org/ccel/anonymous/westminster2.xml",
        filename="westminster2.xml",
        version="CCEL ThML",
        license="Public domain (CCEL)",
    ),
    "wsc": ConfessionSource(
        id="wsc", name="Westminster Shorter Catechism", shortcode="WSC",
        url="https://ccel.org/ccel/anonymous/westminster1.xml",
        filename="westminster1.xml",
        version="CCEL ThML (1674)",
        license="Public domain (CCEL)",
    ),
    "heidelberg": ConfessionSource(
        id="heidelberg", name="Heidelberg Catechism", shortcode="HC",
        url="https://ccel.org/ccel/anonymous/heidelberg.xml",
        filename="heidelberg.xml",
        version="CCEL ThML",
        license="Public domain (CCEL)",
    ),
    "dort": ConfessionSource(
        id="dort", name="Canons of Dort", shortcode="Dort",
        url="https://ccel.org/ccel/anonymous/canonsofdort.xml",
        filename="canonsofdort.xml",
        version="CCEL ThML (English; Synod of Dordrecht 1618-1619)",
        license="Public domain (CCEL)",
    ),
}


@dataclass
class ParsedConfession:
    """One confession's normalized chunks plus any ambiguities to flag."""

    id: str
    chunks: list[NormalizedChunk]
    flags: list[str]


def _norm_ws(text: str) -> str:
    return re.sub(r"[ \t\n\r]+", " ", text).strip()


def _osis_to_verseref(osis: str) -> VerseRef | None:
    """Convert an OSIS id like ``Bible:Rom.3.20`` to a VerseRef, or None."""
    osis = osis.removeprefix("Bible:")
    parts = osis.split(".")
    if len(parts) < 3:
        return None
    book_osis, chap, verse = parts[0], parts[1], parts[2]
    usfm = OSIS_TO_USFM.get(book_osis)
    if usfm is None or not chap.isdigit() or not verse.isdigit():
        return None
    return VerseRef(book=usfm, chapter=int(chap), verse_start=int(verse))


def _make_chunk(
    src: ConfessionSource, prov: Provenance, key: str, text: str,
    meta: dict,
) -> NormalizedChunk:
    return NormalizedChunk(
        id=f"{src.id}:{key}",
        resource_type=ResourceType.CONFESSION,
        ref=None,                # confessions key by section/question, not a verse
        key=key,
        text=text,
        meta={"document": src.id, "shortcode": src.shortcode, **meta},
        provenance=prov,
    )


# --- Westminster catechisms (WSC / WLC) --------------------------------------
# Q/A markup: <p><b><i>Q1: …?</i></b></p> then <p>A1: …</p>  (WSC uses both
# "Q1:" and "Q1." separators; WLC uses "Question 1:" / "Answer:").
_WSC_Q_RE = re.compile(r"^Q(\d{1,3})[:.]\s*(.*)$", re.DOTALL)
_WSC_A_RE = re.compile(r"^A(\d{1,3})[:.]\s*(.*)$", re.DOTALL)
_WLC_Q_RE = re.compile(r"^Question\s+(\d{1,3})[:.]\s*(.*)$", re.DOTALL)
_WLC_A_RE = re.compile(r"^Answer[:.]?\s*(.*)$", re.DOTALL)


def parse_westminster_catechism(
    src: ConfessionSource, prov: Provenance, content: str, *, larger: bool
) -> ParsedConfession:
    soup = BeautifulSoup(content, "lxml-xml")
    flags: list[str] = []
    # Gather all paragraph texts in document order.
    paras = [_norm_ws(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    paras = [p for p in paras if p]

    q_re = _WLC_Q_RE if larger else _WSC_Q_RE
    a_re = _WLC_A_RE if larger else _WSC_A_RE

    chunks: list[NormalizedChunk] = []
    seen: set[int] = set()
    pending_q: tuple[int, str] | None = None
    for p in paras:
        qm = q_re.match(p)
        if qm:
            pending_q = (int(qm.group(1)), qm.group(2).strip())
            continue
        am = a_re.match(p)
        if am and pending_q is not None:
            qnum, qtext = pending_q
            # WLC "Answer:" has no number (1 group); WSC "A<n>:" carries one (2
            # groups) we can cross-check against the question number.
            if larger:
                atext = am.group(1).strip()
            else:
                anum = int(am.group(1))
                if anum != qnum:
                    flags.append(
                        f"{src.id}: Q{qnum} answer numbered A{anum} (mismatch) — review"
                    )
                atext = am.group(2).strip()
            if qnum in seen:
                flags.append(f"{src.id}: duplicate Q{qnum} — review")
            seen.add(qnum)
            text = f"Q{qnum}. {qtext}\nA{qnum}. {atext}"
            chunks.append(_make_chunk(
                src, prov, key=str(qnum),
                text=text,
                meta={"question": qnum, "question_text": qtext,
                      "answer_text": atext, "proof_texts": []},
            ))
            pending_q = None
    chunks.sort(key=lambda c: int(c.key))
    return ParsedConfession(id=src.id, chunks=chunks, flags=flags)


# --- Westminster Confession of Faith (WCF) -----------------------------------
# CCEL westminster3.xml: div2 per chapter, title="Chapter N" or "Chapter N (orig)"
# where orig is the ORIGINAL 1646 chapter number. <h1> = chapter heading; section
# bodies are <p> starting "N. ". We recover only the 33 originals via the mapping.
_WCF_TITLE_RE = re.compile(r"^Chapter\s+(\d+)(?:\s+\((\d+)\))?$")
_WCF_SECTION_RE = re.compile(r"^(\d+)\.\s+(.*)$", re.DOTALL)


def parse_wcf(
    src: ConfessionSource, prov: Provenance, content: str
) -> ParsedConfession:
    soup = BeautifulSoup(content, "lxml-xml")
    flags: list[str] = []
    flags.append(
        "wcf: CCEL source is the modern American PCUS/UPCUSA edition (renumbered, "
        "adds modern chapters Of the Holy Spirit / Of the Gospel and denominational "
        "marriage chapters, carries [PCUS…]/[UPCUSA…] variant brackets). Recovered "
        "the 33 ORIGINAL chapters via CCEL's own (orig) mapping; brackets preserved "
        "verbatim. Architect: confirm whether to re-source the original 1646/1647 "
        "WCF (e.g. Schaff Creeds III) before ship."
    )
    chunks: list[NormalizedChunk] = []
    seen_orig: set[int] = set()

    for d2 in soup.find_all("div2"):
        title = (d2.get("title") or "").strip()
        m = _WCF_TITLE_RE.match(title)
        if not m:
            # Skip non-chapter divs (Note, Declaratory Statement, denominational
            # marriage chapters titled e.g. "Chapter 24: UPCUSA").
            continue
        modern = int(m.group(1))
        orig = int(m.group(2)) if m.group(2) else modern
        if not (1 <= orig <= 33):
            # Modern-only additions (orig 34/35: Holy Spirit, Gospel) — skip + flag.
            flags.append(
                f"wcf: skipping modern chapter {title!r} (maps to original {orig}, "
                "outside the 33-chapter original) — review"
            )
            continue
        if orig in seen_orig:
            flags.append(
                f"wcf: original chapter {orig} appears twice ({title!r}); kept first, "
                "skipped duplicate (likely a denominational variant) — review"
            )
            continue
        seen_orig.add(orig)

        h1 = d2.find("h1")
        heading = _norm_ws(h1.get_text(" ", strip=True)) if h1 else ""

        # Sections: <p> bodies starting "N. ". Skip the parallel-column chapter
        # headers (PCUS/UPCUSA label tables) and the BOK paragraph-number <h5>s.
        seen_sec: set[int] = set()
        for p in d2.find_all("p"):
            ptext = _norm_ws(p.get_text(" ", strip=True))
            sm = _WCF_SECTION_RE.match(ptext)
            if not sm:
                continue
            sec_num = int(sm.group(1))
            sec_text = sm.group(2).strip()
            key = f"{orig}.{sec_num}"
            if sec_num in seen_sec:
                # The CCEL source mislabels WCF 19's 7th section as "6." (a source
                # numbering error). Don't silently renumber to 7 — keep the source's
                # number, give the row a deterministic disambiguated key, and FLAG.
                suffix = sum(1 for c in chunks
                             if c.meta["chapter"] == orig
                             and c.meta["section"] == sec_num) + 1
                key = f"{orig}.{sec_num}#{suffix}"
                flags.append(
                    f"wcf: chapter {orig} has a repeated section number {sec_num} in "
                    f"the CCEL source (stored as key {key!r}); likely a source "
                    "mislabel of the following section — review"
                )
            seen_sec.add(sec_num)
            chunks.append(_make_chunk(
                src, prov, key=key,
                text=sec_text,
                meta={"chapter": orig, "section": sec_num,
                      "chapter_title": heading, "proof_texts": []},
            ))

    chunks.sort(key=lambda c: (c.meta["chapter"], c.meta["section"]))
    present = {c.meta["chapter"] for c in chunks}
    missing = [n for n in range(1, EXPECTED_COUNTS["wcf"] + 1) if n not in present]
    if missing:
        flags.append(
            f"wcf: original chapter(s) {missing} not recoverable from this CCEL "
            "edition — the modern PCUS/UPCUSA text replaced WCF 24 (Of Marriage and "
            "Divorce) with denomination-specific chapters titled 'Chapter 24: UPCUSA' "
            "/ 'Chapter 26: PCUS', which lack the (orig) mapping. NOT guessed; "
            "re-source the original 1646/1647 chapter for ship — review"
        )
    if len(present) != EXPECTED_COUNTS["wcf"]:
        flags.append(
            f"wcf: recovered {len(present)}/{EXPECTED_COUNTS['wcf']} original chapters "
            "— review"
        )
    return ParsedConfession(id=src.id, chunks=chunks, flags=flags)


# --- Heidelberg Catechism ----------------------------------------------------
# Markup: "<n>. Lord's Day" markers, "<b>Question N.</b> … <b>Answer.</b> …" with
# (a),(b) footnote letters and <scripRef osisRef=…> proof-texts. We walk the body
# paragraphs, tracking the current Lord's Day, splitting on "Question N.".
_HC_Q_SPLIT_RE = re.compile(r"\bQuestion\s+(\d{1,3})\.\s*")
# Lord's-Day marker. The CCEL source is inconsistent: most read "N. Lord's Day"
# but some drop the period ("28 Lord's Day") or lowercase "day" ("6. Lord's day").
# We accept those unambiguous variants. (One marker is a source typo — "2." where
# 29 belongs — which we detect by sequence and FLAG rather than silently fix.)
_HC_LD_RE = re.compile(r"^(\d{1,3})\.?\s*Lord's [Dd]ay\b")


def parse_heidelberg(
    src: ConfessionSource, prov: Provenance, content: str
) -> ParsedConfession:
    soup = BeautifulSoup(content, "lxml-xml")
    flags: list[str] = []

    # Collect, in document order, (kind, payload) events: Lord's-Day markers,
    # question starts, and the scripRefs that follow (proof-texts).
    # We accumulate per-question proof-texts by scanning each question's own
    # paragraph subtree for <scripRef>.
    body = soup.find("body") or soup
    paragraphs = body.find_all("p")

    cur_ld = 0
    ld_seq: list[int] = []  # Lord's Day numbers in document order (for anomaly check)
    # First pass: map each question number -> its Lord's Day, and gather the
    # question/answer text + proof-text VerseRefs from the paragraphs that carry it.
    questions: dict[int, dict] = {}
    cur_q: int | None = None
    for p in paragraphs:
        ptext = _norm_ws(p.get_text(" ", strip=True))
        ld_m = _HC_LD_RE.match(ptext)
        if ld_m:
            marked = int(ld_m.group(1))
            # The FIRST marker seen anchors the sequence (its own value, whatever
            # it is). Each subsequent marker is checked against prev+1. A marker
            # out of sequence (the known CCEL typo: "2." where 29 belongs) is NOT
            # silently renumbered — we advance the counter to the expected value
            # and flag. Anchoring on the first marker (rather than forcing it to 1)
            # keeps the parser correct whether it sees the whole document or a
            # mid-document slice.
            if not ld_seq:
                cur_ld = marked
            else:
                expected = ld_seq[-1] + 1
                if marked != expected:
                    flags.append(
                        f"heidelberg: Lord's Day marker reads {marked!r} where "
                        f"{expected} is expected in sequence (likely a CCEL source "
                        "typo) — review; advanced the Lord's Day counter without "
                        "renumbering the source"
                    )
                    cur_ld = expected
                else:
                    cur_ld = marked
            ld_seq.append(cur_ld)
            # A paragraph can carry "N. Lord's Day" then a Question on the same <p>.
        # A <p> may contain a Question start (bold "Question N.").
        q_starts = list(_HC_Q_SPLIT_RE.finditer(ptext))
        if q_starts:
            for qi, qm in enumerate(q_starts):
                qnum = int(qm.group(1))
                start = qm.end()
                end = q_starts[qi + 1].start() if qi + 1 < len(q_starts) else len(ptext)
                seg = ptext[start:end].strip()
                cur_q = qnum
                entry = questions.setdefault(
                    qnum, {"lords_day": cur_ld, "text_parts": [], "proofs": []}
                )
                entry["lords_day"] = cur_ld
                if seg:
                    entry["text_parts"].append(seg)
            # proof-texts on this paragraph attach to the last question started
            for sr in p.find_all("scripRef"):
                vr = _osis_to_verseref(sr.get("osisRef", ""))
                if vr is not None and cur_q is not None:
                    questions[cur_q]["proofs"].append(vr)
        elif cur_q is not None:
            # Continuation paragraph for the current question (answer body / proofs).
            if ptext and not _HC_LD_RE.match(ptext):
                questions[cur_q]["text_parts"].append(ptext)
            for sr in p.find_all("scripRef"):
                vr = _osis_to_verseref(sr.get("osisRef", ""))
                if vr is not None:
                    questions[cur_q]["proofs"].append(vr)

    chunks: list[NormalizedChunk] = []
    for qnum in sorted(questions):
        e = questions[qnum]
        text = _norm_ws(" ".join(e["text_parts"]))
        # Dedupe proof VerseRefs deterministically, preserving first-seen order.
        seen: set[tuple] = set()
        proofs: list[dict] = []
        for vr in e["proofs"]:
            k = (vr.book, vr.chapter, vr.verse_start)
            if k not in seen:
                seen.add(k)
                proofs.append(vr.model_dump(exclude_none=True))
        chunks.append(_make_chunk(
            src, prov, key=str(qnum),
            text=f"Question {qnum}. {text}",
            meta={"question": qnum, "lords_day": e["lords_day"],
                  "proof_texts": proofs},
        ))

    lds = {c.meta["lords_day"] for c in chunks if c.meta["lords_day"]}
    if len(lds) != HEIDELBERG_LORDS_DAYS:
        flags.append(
            f"heidelberg: found {len(lds)} Lord's Days, expected "
            f"{HEIDELBERG_LORDS_DAYS} — review"
        )
    return ParsedConfession(id=src.id, chunks=chunks, flags=flags)


# --- Canons of Dort ----------------------------------------------------------
# CCEL canonsofdort.xml: <div1 type="section" title="First Head of Doctrine."> per
# head; inside, <div2 type="subsection"> separates the positive Articles from the
# "Rejection of Errors". Positive bodies start "ARTICLE N", rejections start
# "PARAGRAPH N"; both restart their numbering each head. We emit one chunk per
# article/paragraph, keyed "h<slug>.a<N>" / "h<slug>.r<N>".
_DORT_HEAD_TITLE_RE = re.compile(r"^(First|Second|Third and Fourth|Fifth)\s+Heads?\s+of\s+Doctrine")
_DORT_HEAD_SLUG: dict[str, str] = {
    "First": "1", "Second": "2", "Third and Fourth": "3-4", "Fifth": "5",
}
_DORT_ART_RE = re.compile(r"^ARTICLE\s+(\d+)\s*\.?\s*(.*)$", re.DOTALL)
_DORT_PARA_RE = re.compile(r"^PARAGRAPH\s+(\d+)\s*\.?\s*(.*)$", re.DOTALL)


def parse_dort(
    src: ConfessionSource, prov: Provenance, content: str
) -> ParsedConfession:
    soup = BeautifulSoup(content, "lxml-xml")
    flags: list[str] = []
    chunks: list[NormalizedChunk] = []
    seen_heads: list[str] = []

    for d1 in soup.find_all("div1", attrs={"type": "section"}):
        title = _norm_ws(d1.get("title") or "")
        hm = _DORT_HEAD_TITLE_RE.match(title)
        if not hm:
            # Conclusion (and Title Page) are not numbered heads of doctrine.
            if title.lower().startswith("conclusion"):
                concl = _norm_ws(d1.get_text(" ", strip=True))
                # Drop the leading "Conclusion" heading word from the body text.
                concl = re.sub(r"^Conclusion\s*", "", concl).strip()
                if concl:
                    chunks.append(_make_chunk(
                        src, prov, key="conclusion", text=concl,
                        meta={"head": "conclusion", "head_title": "Conclusion",
                              "kind": "conclusion", "proof_texts": []},
                    ))
            continue
        slug = _DORT_HEAD_SLUG[hm.group(1)]
        seen_heads.append(slug)
        head_title = title.rstrip(".")

        for d2 in d1.find_all("div2", attrs={"type": "subsection"}):
            sub_title = _norm_ws(d2.get("title") or "")
            is_rejection = "rejection" in sub_title.lower()
            kind = "rejection" if is_rejection else "article"
            marker_re = _DORT_PARA_RE if is_rejection else _DORT_ART_RE
            letter = "r" if is_rejection else "a"

            seen_nums: set[int] = set()
            for p in d2.find_all("p"):
                ptext = _norm_ws(p.get_text(" ", strip=True))
                m = marker_re.match(ptext)
                if not m:
                    continue
                num = int(m.group(1))
                body = m.group(2).strip()
                if num in seen_nums:
                    flags.append(
                        f"dort: head {slug} {kind} {num} appears twice — review"
                    )
                    continue
                seen_nums.add(num)
                key = f"h{slug}.{letter}{num}"
                chunks.append(_make_chunk(
                    src, prov, key=key, text=body,
                    meta={"head": slug, "head_title": head_title,
                          "kind": kind, "number": num,
                          "subsection": sub_title, "proof_texts": []},
                ))

    # Validate per-head article / rejection counts against the known spine.
    for slug, (exp_a, exp_r) in DORT_HEAD_COUNTS.items():
        got_a = sum(1 for c in chunks
                    if c.meta.get("head") == slug and c.meta.get("kind") == "article")
        got_r = sum(1 for c in chunks
                    if c.meta.get("head") == slug and c.meta.get("kind") == "rejection")
        if got_a != exp_a:
            flags.append(
                f"dort: head {slug} has {got_a} articles, expected {exp_a} — review"
            )
        if got_r != exp_r:
            flags.append(
                f"dort: head {slug} has {got_r} rejection paragraphs, expected "
                f"{exp_r} — review"
            )
    n_heads = len(seen_heads)
    if n_heads != len(DORT_HEAD_COUNTS):
        flags.append(
            f"dort: parsed {n_heads} head-sections, expected {len(DORT_HEAD_COUNTS)} "
            f"(canonically {DORT_HEADS} heads, the 3rd & 4th combined) — review"
        )
    return ParsedConfession(id=src.id, chunks=chunks, flags=flags)


# --- dispatch ----------------------------------------------------------------
def parse_confession(
    src: ConfessionSource, prov: Provenance, content: str
) -> ParsedConfession:
    if src.id == "wsc":
        return parse_westminster_catechism(src, prov, content, larger=False)
    if src.id == "wlc":
        return parse_westminster_catechism(src, prov, content, larger=True)
    if src.id == "wcf":
        return parse_wcf(src, prov, content)
    if src.id == "heidelberg":
        return parse_heidelberg(src, prov, content)
    if src.id == "dort":
        return parse_dort(src, prov, content)
    raise ValueError(f"no parser for confession {src.id!r}")
