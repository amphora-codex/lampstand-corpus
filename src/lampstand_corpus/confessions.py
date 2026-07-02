"""M1 — Confessions & catechisms ingestion → NormalizedChunk.

Canonical, public-domain sources only (CLAUDE.md §Sources of record). The
documents converge on the normalized schema at **section / Q&A / article
granularity**: one chunk per confession section (``chapter.section``), catechism
question (the question number), or confession article (the article number).
Scripture proof-texts the source provides are captured as :class:`VerseRef`s in
each chunk's ``meta`` and re-validated against the verse spine downstream.

Sourcing decisions (every structural ambiguity is FLAGGED for human review,
never silently resolved — CLAUDE.md):

* **WCF — RE-SOURCED to the original 1646/47 text** (``andrewhwaller/
  westminster-json`` ``wcf.json``; MIT repo, public-domain text). This REPLACES
  the earlier CCEL ``westminster3.xml``, which was the modern American
  PCUS/UPCUSA edition (renumbered, modern chapters added, ch. 24 unrecoverable).
  The new source carries all **33 original chapters** including the pre-American
  ch. 23 (*Of the Civil Magistrate*) and ch. 24 (*Of Marriage and Divorce*).
  Proof-texts are inline parentheticals we parse to VerseRefs. The six loci the
  **1788 American revision** amended (20.4, 22.3, 23.3, 24.4, 25.6, 31.1-2) are
  marked with an ``amendment_1788`` note on those sections so the app can show
  "original, with the American revision marked"; a diff that finds amended loci
  beyond those six is FLAGGED, not absorbed.
* **WSC** (CCEL ``westminster1.xml``) and **WLC** (``westminster2.xml``) — KEPT
  from P2: clean per-question ThML, the original Q/A text. Unchanged here.
* **1689 London Baptist Confession — ADDED** (``ParticularBaptists/lbcf-1689``;
  CC0). The 1677/89 text, all **32 chapters** incl. ch. 26 (*Of the Church*).
  ``lbcf.json`` is the authoritative chapter/paragraph text; per-paragraph proof
  refs come from the same repo's ``lbcf_with_scripture_refs.md`` (cross-checked
  paragraph-for-paragraph against the JSON; any divergence is FLAGGED).
* **Belgic Confession — ADDED** (Wikisource, the 1840 RPDC public-domain English
  translation). All **37 articles** (I-XXXVII), Roman numerals normalized to
  integers. The 1840 edition carries only occasional inline scripture mentions,
  not a structured proof-text apparatus, so article proof_texts are empty (not
  fabricated).
* **Heidelberg** (CCEL ``heidelberg.xml``) — KEPT from P2: 129 Q across 52 Lord's
  Days, machine-readable ``<scripRef osisRef=…>`` proof-texts. Unchanged.
* **Canons of Dort** (CCEL ``canonsofdort.xml``) — KEPT from P2: a clean English
  ThML edition, 5 heads (3rd & 4th combined), positive Articles + Rejection of
  Errors per head. Unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from .schema import NormalizedChunk, Provenance, ResourceType, VerseRef
from .scripref import parse_proof_block
from .sources import SOURCES_DIR

CONFESSIONS_DIR = SOURCES_DIR / "confessions"


# --- known totals (validation spine) -----------------------------------------
# Per-document expected counts (CLAUDE.md Part B scope). Deviations are FLAGGED.
EXPECTED_COUNTS: dict[str, int] = {
    "wcf": 33,    # chapters
    "wlc": 196,   # questions
    "wsc": 107,   # questions
    "lbcf": 32,   # chapters (1689 London Baptist Confession)
    "belgic": 37,  # articles
    "heidelberg": 129,  # questions
    "dort": 5,    # heads of doctrine (the validation spine for Dort)
}
HEIDELBERG_LORDS_DAYS = 52

# The six loci amended by the 1788 American revision of the WCF (CLAUDE.md / task
# spec). Keyed (chapter, section) -> a short, factual note of the change. The note
# records THAT the locus was revised and the nature of the change (matters of
# public record); the verbatim 1788 replacement text is NOT invented here — a
# canonical public-domain American-revision text must be sourced to populate it
# (FLAGGED). Storing the loci now lets the app mark "original with the American
# revision marked" and lets the validator confirm exactly these six and no others.
WCF_1788_AMENDMENTS: dict[tuple[int, int], str] = {
    (20, 4): "1788 American revision: removed the clause empowering the civil "
             "magistrate to proceed against those who publish opinions or "
             "practices contrary to the light of nature or the church's peace.",
    (22, 3): "1788 American revision: softened the obligation to take a lawful "
             "oath when imposed by lawful authority.",
    (23, 3): "1788 American revision: replaced the magistrate's authority over "
             "the church (calling synods, suppressing heresy) with a duty to "
             "protect the church of our common Lord without preference to any "
             "denomination.",
    (24, 4): "1788 American revision: dropped the clause forbidding marriage "
             "within the degrees of consanguinity/affinity forbidden in the Word "
             "and the reference to the wife's kindred.",
    (25, 6): "1788 American revision: removed the identification of the Pope as "
             "the Antichrist / 'that man of sin'.",
    (31, 1): "1788 American revision: synods and councils are convened by the "
             "ministers and officers of the church, not at the magistrate's call.",
    (31, 2): "1788 American revision (cont.): removed the magistrate's role in "
             "convening synods, consistent with 31.1.",
}

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
    repo_license: str = ""   # license of the upstream REPO (text itself is PD)
    aux_url: str = ""        # secondary source file (e.g. the 1689 proof-text md)
    aux_filename: str = ""
    aux_note: str = ""       # provenance/license note for the aux file (manifest)
    xref_url: str = ""       # validation-only cross-check snapshot (not a source)
    xref_filename: str = ""
    amend_url: str = ""      # amendment source (e.g. the 1788 American-revision WCF)
    amend_filename: str = ""
    amend_license: str = ""  # license of the amendment source text (PD)

    @property
    def dest(self) -> Path:
        return CONFESSIONS_DIR / self.id / self.filename

    @property
    def aux_dest(self) -> Path | None:
        return CONFESSIONS_DIR / self.id / self.aux_filename if self.aux_filename \
            else None

    @property
    def xref_dest(self) -> Path | None:
        return CONFESSIONS_DIR / self.id / self.xref_filename if self.xref_filename \
            else None

    @property
    def amend_dest(self) -> Path | None:
        return CONFESSIONS_DIR / self.id / self.amend_filename if self.amend_filename \
            else None


# Public-domain confession texts. Committed under sources/confessions/<id>/ via
# git-lfs (repo licenses recorded per source; the underlying confessions are PD).
CONFESSION_SOURCES: dict[str, ConfessionSource] = {
    # RE-SOURCED: original 1646/47 WCF (replaces the bad CCEL American edition).
    "wcf": ConfessionSource(
        id="wcf", name="Westminster Confession of Faith", shortcode="WCF",
        url="https://raw.githubusercontent.com/andrewhwaller/westminster-json/"
            "main/wcf.json",
        filename="wcf-original-1646.json",
        version="Original 1646/47 text (andrewhwaller/westminster-json wcf.json); "
                "33 chapters incl. original ch. 23 & 24; 1788 loci marked",
        license="Public domain (original 1646/47 text)",
        repo_license="MIT (andrewhwaller/westminster-json)",
        xref_url="https://en.wikisource.org/w/api.php?action=parse&page="
                 "The_Confession_of_Faith_of_the_Assembly_of_Divines_at_Westminster"
                 "&prop=text&formatversion=2&format=json",
        xref_filename="wcf-burges-1646-wikisource.parse.json",
        # 1788 AMERICAN-REVISION verbatim text source (PD). CCEL westminster3.xml is
        # the American revision: it carries the revised wording of the six amended
        # loci. NOT the OPC's copyrighted edition; NOT a modern annotated typeset.
        # The edition renumbers chapters for the 1903 additions (Of the Holy Spirit
        # / Of the Gospel inserted at 9-10), but records each chapter's ORIGINAL
        # number in a parenthetical title ("Chapter 25 (23)" = original ch. 23), so
        # the six original-numbered loci are recoverable deterministically. The
        # edition also embeds later DENOMINATIONAL variant brackets ([PCUS ...] /
        # [UPCUSA ...]); the verbatim 1788 base reading is the text with those later
        # insertions removed — done explicitly and FLAGGED, never silently merged.
        amend_url="https://ccel.org/ccel/anonymous/westminster3.xml",
        amend_filename="westminster3-american-revision.xml",
        amend_license="Public domain (1788 American revision text; CCEL ThML)",
    ),
    "wlc": ConfessionSource(
        id="wlc", name="Westminster Larger Catechism", shortcode="WLC",
        url="https://ccel.org/ccel/anonymous/westminster2.xml",
        filename="westminster2.xml",
        version="CCEL ThML",
        license="Public domain (CCEL)",
        # ARCHITECT-APPROVED proof-text supplement (Rank 14 follow-up): the
        # Westminster Standards JSON repo carries the Assembly's lettered
        # proof groups per answer clause. Text still comes from CCEL (chunk
        # ids unchanged); ONLY the proof apparatus is merged in.
        aux_url="https://raw.githubusercontent.com/reformed-christian/"
                "westminster-standards-json/main/catechisms/larger/"
                "westminster_larger_catechism_with_references.json",
        aux_filename="wlc_with_references.json",
        aux_note="reformed-christian/westminster-standards-json — repo "
                 "declares NO license; underlying 1647/48 catechism text + "
                 "proof apparatus are public domain (repo's own sources/ are "
                 "the PD PRTS edition PDFs). FLAGGED for architect review.",
    ),
    "wsc": ConfessionSource(
        id="wsc", name="Westminster Shorter Catechism", shortcode="WSC",
        url="https://ccel.org/ccel/anonymous/westminster1.xml",
        filename="westminster1.xml",
        version="CCEL ThML (1674)",
        license="Public domain (CCEL)",
        aux_url="https://raw.githubusercontent.com/reformed-christian/"
                "westminster-standards-json/main/catechisms/shorter/"
                "westminster_shorter_catechism.json",
        aux_filename="wsc_with_references.json",
        aux_note="reformed-christian/westminster-standards-json — repo "
                 "declares NO license; underlying 1647/48 catechism text + "
                 "proof apparatus are public domain (repo's own sources/ are "
                 "the PD PRTS edition PDFs). FLAGGED for architect review.",
    ),
    # ADDED: 1689 London Baptist Confession (original 1677/89, 32 chapters).
    "lbcf": ConfessionSource(
        id="lbcf", name="Second London Baptist Confession of Faith (1689)",
        shortcode="1689",
        url="https://raw.githubusercontent.com/ParticularBaptists/lbcf-1689/"
            "master/lbcf.json",
        filename="lbcf.json",
        version="1677/89 text (ParticularBaptists/lbcf-1689 lbcf.json); 32 chapters",
        license="Public domain (1677/89 text)",
        repo_license="CC0-1.0 (ParticularBaptists/lbcf-1689)",
        aux_url="https://raw.githubusercontent.com/ParticularBaptists/lbcf-1689/"
                "master/lbcf_with_scripture_refs.md",
        aux_filename="lbcf_with_scripture_refs.md",
    ),
    # ADDED: Belgic Confession (1840 RPDC public-domain English translation).
    "belgic": ConfessionSource(
        id="belgic", name="Belgic Confession", shortcode="BC",
        url="https://en.wikisource.org/w/api.php?action=parse&page="
            "The_Constitution_of_the_Reformed_Dutch_Church_of_North_America/"
            "The_Confession_of_Faith&prop=text&formatversion=2&format=json",
        filename="belgic-1840-rpdc.parse.json",
        version="1840 RPDC English translation (Wikisource parse snapshot); "
                "37 articles I-XXXVII",
        license="Public domain (PD-old; 1840 translation)",
        repo_license="Wikisource (CC BY-SA wrapper; underlying text PD-old)",
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


def _load_westminster_proofs(
    src: ConfessionSource, flags: list[str]
) -> dict[int, list[dict]]:
    """Proof-texts per question from the Westminster-Standards JSON aux file.

    The source follows the Assembly's convention of LETTERED PROOF GROUPS per
    answer clause: each question carries ``clauses[]``, each clause a footnote
    number and its ``references[]`` (citation string + KJV text). We flatten to
    the per-question union in clause order (the shape every other document's
    ``proof_texts`` uses), deduped first-seen. Every reference string is parsed
    through the same :func:`scripref.parse_proof_block` spine parser used by
    the WCF/1689 — an unresolvable citation is FLAGGED, never guessed.

    Bookkeeping identity (source-count sanity, checked here): for each
    question, parsed refs + flagged-unparsed refs == reference entries in the
    source JSON.
    """
    import json as _json

    aux = src.aux_dest
    if aux is None or not aux.exists():
        flags.append(
            f"{src.id}: Westminster proof-text JSON not present "
            f"({src.aux_filename}); questions ingested without proof-texts — "
            "run snapshot-confessions")
        return {}
    data = _json.loads(aux.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        flags.append(f"{src.id}: unexpected proof JSON shape (not a list) — review")
        return {}

    out: dict[int, list[dict]] = {}
    for q in data:
        qnum = q.get("number")
        if not isinstance(qnum, int):
            flags.append(f"{src.id}: proof JSON entry without a question number — review")
            continue
        n_source_refs = 0
        n_parsed = 0
        n_unparsed = 0
        seen: set[tuple] = set()
        proofs: list[dict] = []
        for clause in q.get("clauses", []):
            for ref in clause.get("references", []):
                citation = _norm_ws(str(ref.get("reference", "")))
                if not citation:
                    continue
                n_source_refs += 1
                res = parse_proof_block(citation)
                for vr in res.refs:
                    k = (vr.book, vr.chapter, vr.verse_start)
                    if k not in seen:
                        seen.add(k)
                        proofs.append(vr.model_dump(exclude_none=True))
                n_parsed += len(res.refs)
                for tok in res.unparsed:
                    n_unparsed += 1
                    flags.append(
                        f"{src.id}: Q{qnum} proof citation not resolved -> "
                        f"{tok!r} (kept for human review, not guessed)")
        if n_parsed == 0 and n_source_refs > 0:
            flags.append(
                f"{src.id}: Q{qnum} has {n_source_refs} source citations but "
                "NONE parsed — review")
        out[qnum] = proofs
    return out


def parse_westminster_catechism(
    src: ConfessionSource, prov: Provenance, content: str, *, larger: bool
) -> ParsedConfession:
    soup = BeautifulSoup(content, "lxml-xml")
    flags: list[str] = []
    proofs_by_q = _load_westminster_proofs(src, flags)
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
                      "answer_text": atext,
                      "proof_texts": proofs_by_q.get(qnum, [])},
            ))
            pending_q = None
    chunks.sort(key=lambda c: int(c.key))
    # Proof coverage bookkeeping: when the aux apparatus is present, every
    # question should carry proofs; question numbers in the JSON must match the
    # CCEL question set exactly.
    if proofs_by_q:
        ccel_q = {int(c.key) for c in chunks}
        json_q = set(proofs_by_q)
        for qnum in sorted(ccel_q - json_q):
            flags.append(f"{src.id}: Q{qnum} missing from the proof JSON — review")
        for qnum in sorted(json_q - ccel_q):
            flags.append(f"{src.id}: proof JSON has unknown Q{qnum} — review")
        empties = [int(c.key) for c in chunks if not c.meta["proof_texts"]]
        if empties:
            flags.append(
                f"{src.id}: {len(empties)} question(s) with zero parsed "
                f"proof-texts: {empties[:12]} — review")
    return ParsedConfession(id=src.id, chunks=chunks, flags=flags)


# --- proof-text helper -------------------------------------------------------
def _proofs_from_text(text: str, flags: list[str], where: str) -> list[dict]:
    """Extract VerseRef dicts from the parenthetical scripture citations in
    ``text`` (the WCF inlines proof-texts in parentheses). Unparsable citation
    tokens are FLAGGED (with ``where`` context), never guessed."""
    proofs: list[dict] = []
    seen: set[tuple] = set()
    for grp in re.findall(r"\(([^()]+)\)", text):
        if not re.search(r"\d+\s*[:.]\s*\d+", grp):
            continue  # not a scripture citation group (no chapter:verse)
        res = parse_proof_block(grp)
        for vr in res.refs:
            k = (vr.book, vr.chapter, vr.verse_start, vr.verse_end)
            if k not in seen:
                seen.add(k)
                proofs.append(vr.model_dump(exclude_none=True))
        for tok in res.unparsed:
            flags.append(
                f"{where}: proof-text token not resolved -> {tok} "
                "(left out; review — chapter-only / source-typo citations are "
                "expected here)"
            )
    return proofs


# --- WCF 1788 American-revision verbatim text -------------------------------
# Source: CCEL westminster3.xml (the American revision). Chapters render as
# <div2 title="Chapter N (M)"> where M is the ORIGINAL chapter number; sections
# are <p> bodies opening "N. ". We pull the verbatim revised wording for EXACTLY
# the six amended loci (keyed by ORIGINAL chapter.section) and nothing else.
#
# The edition embeds later denominational variant brackets, e.g.
#   "[PCUS Yet it is a sin to refuse an oath ...]"
#   "[PCUS without warrant in fact ...] [UPCUSA unscriptural ...] a usurpation"
# These mark 19th/20th-century PRESBYTERIAN-denomination changes, not the 1788
# text. The verbatim 1788 base reading is the text with those bracketed insertions
# removed. We do this deterministically and FLAG every locus where a bracket was
# present so the human verifies the base reading against a printed 1788 reference.
_WCF_AMEND_CHAP_RE = re.compile(r"Chapter\s+\d+\s*\((\d+)\)")
_WCF_AMEND_SEC_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$", re.DOTALL)
# A later-denomination variant bracket: "[PCUS ...]" / "[UPCUSA ...]" (the tag is
# an ALL-CAPS denomination code right after the "["). Non-greedy to the matching
# "]"; our source has no nested brackets in these loci.
_WCF_DENOM_BRACKET_RE = re.compile(r"\[(PCUS|UPCUSA|PCA|OPC|ARP)\b[^\]]*\]")


def _original_chapter(title: str) -> int | None:
    """Recover a CCEL American-edition chapter's ORIGINAL number from its title.

    "Chapter 25 (23)" -> 23 (the parenthetical original). Falls back to the bare
    leading number when there is no parenthetical (chapters 1-8, unrenumbered).
    """
    m = _WCF_AMEND_CHAP_RE.search(title)
    if m:
        return int(m.group(1))
    m2 = re.search(r"Chapter\s+(\d+)", title)
    return int(m2.group(1)) if m2 else None


def parse_wcf_1788_amendments(
    content: str, flags: list[str]
) -> dict[tuple[int, int], str]:
    """Extract verbatim 1788 American-revision text for EXACTLY the six loci.

    Returns ``{(orig_chapter, section): verbatim_text}`` for the six amended loci.
    A locus carrying a later denominational bracket is recorded with the bracket
    removed (the 1788 base reading) and FLAGGED for human verification. A locus we
    cannot locate is FLAGGED and omitted (never fabricated). The caller confirms
    the result covers exactly the six loci and touches nothing beyond them.
    """
    soup = BeautifulSoup(content, "lxml-xml")
    wanted = set(WCF_1788_AMENDMENTS)
    by_chap: dict[int, list] = {}
    for d2 in soup.find_all("div2"):
        cn = _original_chapter(d2.get("title") or "")
        if cn is not None:
            by_chap.setdefault(cn, []).append(d2)

    out: dict[tuple[int, int], str] = {}
    for (cn, sn) in sorted(wanted):
        divs = by_chap.get(cn)
        if not divs:
            flags.append(
                f"wcf 1788: original chapter {cn} not found in the American-revision "
                "source (westminster3.xml) — verbatim {cn}.{sn} not populated; review"
            )
            continue
        found = False
        for d2 in divs:
            for p in d2.find_all("p"):
                txt = _norm_ws(p.get_text(" ", strip=True))
                m = _WCF_AMEND_SEC_RE.match(txt)
                if not m or int(m.group(1)) != sn:
                    continue
                body = m.group(2).strip()
                had_bracket = bool(_WCF_DENOM_BRACKET_RE.search(body))
                if had_bracket:
                    body = _norm_ws(_WCF_DENOM_BRACKET_RE.sub(" ", body))
                    flags.append(
                        f"wcf 1788: locus {cn}.{sn} carries later denominational "
                        "variant bracket(s) ([PCUS]/[UPCUSA]) in the source; the "
                        "verbatim text stored is the 1788 BASE reading with those "
                        "later insertions removed — VERIFY against a printed 1788 "
                        "American-revision reference; review"
                    )
                out[(cn, sn)] = body
                found = True
                break
            if found:
                break
        if not found:
            # Diagnose WHY (so the human knows it isn't a parser miss). The most
            # common real cause: this CCEL edition carries a LATER denominational
            # rewrite of the whole chapter (notably ch. 24 "Of Marriage" is the
            # UPCUSA 1953 wholesale rewrite, only two sections, not the 1788 text),
            # so the 1788 section simply isn't present to quote verbatim.
            present_secs: list[int] = []
            for d2 in divs:
                for p in d2.find_all("p"):
                    mm = _WCF_AMEND_SEC_RE.match(_norm_ws(p.get_text(" ", strip=True)))
                    if mm:
                        present_secs.append(int(mm.group(1)))
            title = _norm_ws(divs[0].get("title") or "")
            denom = re.search(r"\b(PCUS|UPCUSA|PCA|OPC|ARP)\b", title)
            if denom or (cn == 24 and max(present_secs or [0]) < sn):
                flags.append(
                    f"wcf 1788: locus {cn}.{sn} could NOT be quoted verbatim — this "
                    "CCEL American edition carries a later DENOMINATIONAL rewrite of "
                    f"the chapter (title={title!r}, sections present={sorted(set(present_secs))}), "
                    "not the 1788 revision text. The 1788 revision of this locus only "
                    "DROPPED a clause from the original (see the descriptive "
                    "amendment_1788 note); a clean 1788 verbatim must be sourced "
                    "separately or reconstructed by the human from the original minus "
                    "the dropped clause — NOT fabricated here; review"
                )
            else:
                flags.append(
                    f"wcf 1788: section {cn}.{sn} not found in the American-revision "
                    f"source (chapter present, sections={sorted(set(present_secs))}) "
                    "— verbatim text not populated; review"
                )

    # Defensive: confirm we touched only the six loci (the regex/section walk above
    # only ever queries `wanted`, but assert it so a future edit can't widen scope).
    extra = set(out) - wanted
    if extra:
        flags.append(
            f"wcf 1788: verbatim extraction produced loci beyond the six "
            f"({sorted(extra)}) — this should be impossible; review"
        )
    return out


# --- Westminster Confession of Faith (WCF) — original 1646/47 ----------------
# Source: andrewhwaller/westminster-json wcf.json. JSON shape:
#   {languages: {eng: {chapters: [{id, title, sections: [{id, text}]}]}}}
# Chapter 12 (Of Adoption) is a single-text chapter: {id, title, text} with no
# `sections` list. Proof-texts are inline parentheticals in each section's text.
def parse_wcf(
    src: ConfessionSource, prov: Provenance, content: str
) -> ParsedConfession:
    data = json.loads(content)
    flags: list[str] = []
    try:
        chapters = data["languages"]["eng"]["chapters"]
    except (KeyError, TypeError):
        flags.append("wcf: unexpected JSON shape (no languages.eng.chapters) — review")
        return ParsedConfession(id=src.id, chunks=[], flags=flags)

    # Verbatim 1788 American-revision text for the six amended loci, sourced from
    # the committed CCEL American-revision snapshot (if present). When the snapshot
    # is absent we keep only the descriptive notes and FLAG (no fabrication).
    amend_text: dict[tuple[int, int], str] = {}
    amend = src.amend_dest
    if amend is not None and amend.exists():
        amend_text = parse_wcf_1788_amendments(
            amend.read_text(encoding="utf-8"), flags
        )
    else:
        flags.append(
            "wcf 1788: American-revision verbatim source "
            "(westminster3-american-revision.xml) not present; the six loci carry "
            "only the descriptive amendment notes, not verbatim revised text — "
            "run snapshot-confessions to fetch it; review"
        )

    chunks: list[NormalizedChunk] = []
    amended_seen: set[tuple[int, int]] = set()
    amend_text_seen: set[tuple[int, int]] = set()

    for c in chapters:
        cnum = int(c["id"])
        title = _norm_ws(str(c.get("title", "")))
        # Normalize the two chapter shapes to a list of (section_id, text).
        if "sections" in c:
            sections = [(int(s["id"]), s["text"]) for s in c["sections"]]
        elif "text" in c:
            sections = [(1, c["text"])]  # single-paragraph chapter (ch. 12)
        else:
            flags.append(f"wcf: chapter {cnum} has neither sections nor text — review")
            continue

        for snum, raw in sections:
            text = _norm_ws(raw)
            proofs = _proofs_from_text(text, flags, f"wcf {cnum}.{snum}")
            meta: dict = {
                "chapter": cnum, "section": snum,
                "chapter_title": title, "proof_texts": proofs,
            }
            note = WCF_1788_AMENDMENTS.get((cnum, snum))
            if note is not None:
                meta["amendment_1788"] = note
                amended_seen.add((cnum, snum))
                verbatim = amend_text.get((cnum, snum))
                if verbatim is not None:
                    meta["amendment_1788_text"] = verbatim
                    amend_text_seen.add((cnum, snum))
            chunks.append(_make_chunk(
                src, prov, key=f"{cnum}.{snum}", text=text, meta=meta,
            ))

    chunks.sort(key=lambda c: (c.meta["chapter"], c.meta["section"]))

    # Structural confirmations the task asks for, surfaced as informational flags.
    present = {c.meta["chapter"] for c in chunks}
    n_chapters = len(present)
    if n_chapters != EXPECTED_COUNTS["wcf"]:
        flags.append(
            f"wcf: parsed {n_chapters}/{EXPECTED_COUNTS['wcf']} chapters — review"
        )
    ch23 = next((c for c in chunks if c.meta["chapter"] == 23), None)
    ch24 = next((c for c in chunks if c.meta["chapter"] == 24), None)
    if not (ch23 and "Civil Magistrate" in ch23.meta["chapter_title"]):
        flags.append(
            "wcf: original chapter 23 'Of the Civil Magistrate' not found as "
            "expected — review"
        )
    if not (ch24 and "Marriage" in ch24.meta["chapter_title"]):
        flags.append(
            "wcf: original chapter 24 'Of Marriage and Divorce' not found as "
            "expected — review"
        )
    # Confirm exactly the six known 1788 loci were marked (seven sections, since
    # 31 splits into .1 and .2). Anything missing is a real problem to flag.
    expected_loci = set(WCF_1788_AMENDMENTS)
    if amended_seen != expected_loci:
        missing = sorted(expected_loci - amended_seen)
        extra = sorted(amended_seen - expected_loci)
        flags.append(
            f"wcf: 1788 amendment loci marked={sorted(amended_seen)} differ from "
            f"the expected six (missing={missing}, extra={extra}) — review"
        )
    # Confirm verbatim 1788 revised text landed on the amended loci AND on NOTHING
    # beyond them (the task's acceptance check). 24.4 is a KNOWN, justified gap:
    # this PD American edition's ch. 24 is a later denominational rewrite, not the
    # 1788 text, so 24.4 cannot be quoted verbatim from it (flagged above).
    if amend_text:
        beyond = amend_text_seen - expected_loci
        missing = expected_loci - amend_text_seen
        if beyond:
            flags.append(
                f"wcf 1788: verbatim revised text touched loci BEYOND the amended "
                f"set ({sorted(beyond)}) — must not happen; review"
            )
        sourceable = expected_loci - {(24, 4)}  # 24.4 not verbatim-sourceable here
        if amend_text_seen >= sourceable and missing <= {(24, 4)}:
            flags.append(
                "wcf 1788: verbatim American-revision text populated on the amended "
                "loci (20.4, 22.3, 23.3, 25.6, 31.1, 31.2) alongside the retained "
                "original text; 24.4 verbatim is the one justified gap (source ch.24 "
                "is a later denominational rewrite) — human-verify the two bracket "
                "loci 22.3 & 25.6, and source/reconstruct 24.4 separately"
            )
        else:
            flags.append(
                f"wcf 1788: verbatim text present on {sorted(amend_text_seen)} — "
                f"unexpected coverage (missing={sorted(missing)}); review"
            )
    return ParsedConfession(id=src.id, chunks=chunks, flags=flags)


# --- 1689 London Baptist Confession ------------------------------------------
# Source: ParticularBaptists/lbcf-1689 lbcf.json — {title, chapters: {"N":
# {title, paragraphs: {"M": text}}}}. Per-paragraph proof refs come from the same
# repo's lbcf_with_scripture_refs.md (chapter headers "## CHAPTER N: TITLE", each
# paragraph "M. text" then a "( refs )" line). We cross-check the two file's
# paragraph maps and FLAG any divergence rather than trusting one blindly.
_LBCF_CHAP_RE = re.compile(r"^##\s*CHAPTER\s+(\d+):\s*(.+)$")
_LBCF_PARA_RE = re.compile(r"^\s*(\d+)\.\s+(.+)$")
_LBCF_REFS_RE = re.compile(r"^\s*\((.+)\)\s*$")


def _parse_lbcf_proof_md(md: str) -> dict[tuple[int, int], str]:
    """Map (chapter, paragraph) -> raw proof-ref block string from the 1689 md.

    The md's chapter 12 (Of Adoption) is a single UNNUMBERED paragraph; it is
    attributed to paragraph 1 deterministically (the JSON keys it the same way).
    """
    out: dict[tuple[int, int], str] = {}
    cur_chap: int | None = None
    cur_para: int | None = None
    saw_numbered_para = False
    for line in md.splitlines():
        cm = _LBCF_CHAP_RE.match(line)
        if cm:
            cur_chap = int(cm.group(1))
            cur_para = None
            saw_numbered_para = False
            continue
        if cur_chap is None:
            continue
        pm = _LBCF_PARA_RE.match(line)
        if pm and not _LBCF_REFS_RE.match(line):
            cur_para = int(pm.group(1))
            saw_numbered_para = True
            continue
        rm = _LBCF_REFS_RE.match(line)
        if rm:
            # A single-paragraph chapter has no "M." marker before its refs.
            para = cur_para if cur_para is not None else (
                1 if not saw_numbered_para else None
            )
            if para is not None and cur_chap is not None:
                out[(cur_chap, para)] = rm.group(1)
    return out


def parse_lbcf(
    src: ConfessionSource, prov: Provenance, content: str
) -> ParsedConfession:
    data = json.loads(content)
    flags: list[str] = []
    chapters = data.get("chapters")
    if not isinstance(chapters, dict):
        flags.append("lbcf: unexpected JSON shape (no chapters dict) — review")
        return ParsedConfession(id=src.id, chunks=[], flags=flags)

    # Proof refs from the auxiliary md (same CC0 repo), if snapshotted.
    proof_md: dict[tuple[int, int], str] = {}
    aux = src.aux_dest
    if aux is not None and aux.exists():
        proof_md = _parse_lbcf_proof_md(aux.read_text(encoding="utf-8"))
    else:
        flags.append(
            "lbcf: proof-text md (lbcf_with_scripture_refs.md) not present; "
            "chapters ingested without proof-texts — review"
        )

    chunks: list[NormalizedChunk] = []
    for cnum_s in sorted(chapters, key=int):
        cnum = int(cnum_s)
        chap = chapters[cnum_s]
        title = _norm_ws(str(chap.get("title", "")))
        paras = chap.get("paragraphs", {})
        for pnum_s in sorted(paras, key=int):
            pnum = int(pnum_s)
            text = _norm_ws(str(paras[pnum_s]))
            proofs: list[dict] = []
            block = proof_md.get((cnum, pnum))
            if block is not None:
                proofs = _proofs_from_block(block, flags, f"lbcf {cnum}.{pnum}")
            elif proof_md:
                flags.append(
                    f"lbcf: no proof-text block found in the md for {cnum}.{pnum} "
                    "(JSON has the paragraph but the md cross-reference is missing) "
                    "— review"
                )
            chunks.append(_make_chunk(
                src, prov, key=f"{cnum}.{pnum}", text=text,
                meta={"chapter": cnum, "section": pnum,
                      "chapter_title": title, "proof_texts": proofs},
            ))

    # Cross-check: every md (chapter, para) should have a matching JSON paragraph.
    json_keys = {(int(c), int(p)) for c in chapters for p in chapters[c].get(
        "paragraphs", {})}
    for key in sorted(proof_md):
        if key not in json_keys:
            flags.append(
                f"lbcf: proof-md has {key} but the JSON text has no such paragraph "
                "— md/JSON divergence, review"
            )

    chunks.sort(key=lambda c: (c.meta["chapter"], c.meta["section"]))
    present = {c.meta["chapter"] for c in chunks}
    if len(present) != EXPECTED_COUNTS["lbcf"]:
        flags.append(
            f"lbcf: parsed {len(present)}/{EXPECTED_COUNTS['lbcf']} chapters — review"
        )
    ch26 = next((c for c in chunks if c.meta["chapter"] == 26), None)
    if not (ch26 and "Church" in ch26.meta["chapter_title"]):
        flags.append(
            "lbcf: chapter 26 'Of the Church' not found as expected — review"
        )
    return ParsedConfession(id=src.id, chunks=chunks, flags=flags)


def _proofs_from_block(block: str, flags: list[str], where: str) -> list[dict]:
    """Resolve a ``;``-separated proof block (1689 md) to VerseRef dicts; flag the
    unresolved tokens (chapter-only / source typos) for review."""
    res = parse_proof_block(block)
    proofs: list[dict] = []
    seen: set[tuple] = set()
    for vr in res.refs:
        k = (vr.book, vr.chapter, vr.verse_start, vr.verse_end)
        if k not in seen:
            seen.add(k)
            proofs.append(vr.model_dump(exclude_none=True))
    for tok in res.unparsed:
        flags.append(
            f"{where}: proof-text token not resolved -> {tok!r} (left out; review)"
        )
    return proofs


# --- Belgic Confession — 1840 RPDC English translation -----------------------
# Source: Wikisource parse-API JSON ({parse: {text: "<rendered html>"}}). Article
# boundaries render two ways in this edition: Article I appears as a bare header
# "Article I." with its title on the next paragraph; Articles II-XXXVII render as
# one paragraph that begins with the Roman numeral and carries the title inline
# ("II. By what means God is made known unto us."). The body follows in later
# paragraph(s) until the next header. Roman numerals I-XXXVII -> integers.
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
# A header paragraph: optional "Article", a Roman numeral, a period, then either
# nothing (Article I form) or the inline article title (II-XXXVII form).
_BELGIC_HEAD_RE = re.compile(
    r"^(?:Article\s+)?([IVXLC]+)\.\s*(.*)$"
)


def _roman_to_int(s: str) -> int | None:
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        v = _ROMAN_VALUES.get(ch)
        if v is None:
            return None
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def _belgic_paragraphs(html: str) -> list[str]:
    """Flatten the rendered Belgic HTML into ordered paragraph strings, keeping
    italic boundaries collapsed into plain text (titles read inline)."""
    soup = BeautifulSoup(html, "lxml")
    paras: list[str] = []
    for p in soup.find_all("p"):
        txt = _norm_ws(p.get_text(" ", strip=True))
        if txt:
            paras.append(txt)
    return paras


def parse_belgic(
    src: ConfessionSource, prov: Provenance, content: str
) -> ParsedConfession:
    flags: list[str] = []
    try:
        html = json.loads(content)["parse"]["text"]
    except (KeyError, TypeError, ValueError):
        flags.append("belgic: unexpected parse-API JSON shape — review")
        return ParsedConfession(id=src.id, chunks=[], flags=flags)

    paras = _belgic_paragraphs(html)

    # Walk paragraphs: a header paragraph ("II." / "Article I.") opens an article;
    # its title is the next short italic-style paragraph; the rest is body text
    # until the next header.
    chunks: list[NormalizedChunk] = []
    cur_num: int | None = None
    cur_title: str | None = None
    cur_body: list[str] = []
    prev_num = 0

    def flush() -> None:
        nonlocal cur_num, cur_title, cur_body
        if cur_num is None:
            return
        body = _norm_ws(" ".join(cur_body))
        title = cur_title or ""
        chunks.append(_make_chunk(
            src, prov, key=str(cur_num), text=body,
            meta={"article": cur_num, "article_title": title, "proof_texts": []},
        ))
        cur_num, cur_title, cur_body = None, None, []

    for para in paras:
        hm = _BELGIC_HEAD_RE.match(para)
        # A header only if the leading token is a valid Roman numeral that is the
        # NEXT article in sequence — this avoids misreading a body paragraph that
        # happens to begin "I." and keeps us from guessing across a real gap.
        num = _roman_to_int(hm.group(1)) if hm else None
        if num is not None and num == prev_num + 1:
            flush()
            prev_num = num
            cur_num = num
            inline_title = _norm_ws(hm.group(2))
            cur_title = inline_title or None  # None -> title is the next paragraph
            cur_body = []
            continue
        if cur_num is None:
            continue  # front-matter before Article I
        if cur_title is None:
            # Article I form: the first paragraph after the bare header is the title.
            cur_title = para
        else:
            cur_body.append(para)
    flush()

    chunks.sort(key=lambda c: c.meta["article"])
    n = len(chunks)
    if n != EXPECTED_COUNTS["belgic"]:
        flags.append(
            f"belgic: parsed {n}/{EXPECTED_COUNTS['belgic']} articles — review"
        )
    nums = [c.meta["article"] for c in chunks]
    missing = [i for i in range(1, EXPECTED_COUNTS["belgic"] + 1) if i not in nums]
    if missing:
        flags.append(f"belgic: missing article(s) {missing} — review")
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
            # Inline <scripRef> citations attach as proof-texts to the article
            # (or rejection paragraph) whose <p> carries them; refs in a
            # continuation paragraph attach to the most recent article. The
            # citation TEXT stays in the body verbatim (get_text includes it),
            # so recovering the refs changes no chunk text.
            cur_proofs: list | None = None
            for p in d2.find_all("p"):
                ptext = _norm_ws(p.get_text(" ", strip=True))
                m = marker_re.match(ptext)
                if not m:
                    if cur_proofs is not None:
                        for sr in p.find_all("scripRef"):
                            vr = _osis_to_verseref(sr.get("osisRef", ""))
                            if vr is not None:
                                cur_proofs.append(
                                    vr.model_dump(exclude_none=True))
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
                cur_proofs = [
                    vr.model_dump(exclude_none=True)
                    for sr in p.find_all("scripRef")
                    if (vr := _osis_to_verseref(sr.get("osisRef", ""))) is not None
                ]
                chunks.append(_make_chunk(
                    src, prov, key=key, text=body,
                    meta={"head": slug, "head_title": head_title,
                          "kind": kind, "number": num,
                          "subsection": sub_title,
                          "proof_texts": cur_proofs},
                ))

    # Dedupe each section's proof refs deterministically (first-seen order) —
    # a ref cited twice in one article must not double-count downstream.
    for c in chunks:
        seen: set[tuple] = set()
        unique: list[dict] = []
        for d in c.meta.get("proof_texts", []):
            k = (d.get("book"), d.get("chapter"), d.get("verse_start"))
            if k not in seen:
                seen.add(k)
                unique.append(d)
        c.meta["proof_texts"] = unique

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
    if src.id == "lbcf":
        return parse_lbcf(src, prov, content)
    if src.id == "belgic":
        return parse_belgic(src, prov, content)
    if src.id == "heidelberg":
        return parse_heidelberg(src, prov, content)
    if src.id == "dort":
        return parse_dort(src, prov, content)
    raise ValueError(f"no parser for confession {src.id!r}")
