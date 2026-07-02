"""Confession/catechism parser unit tests + built-DB spot checks.

Unit tests run on small inline ThML fixtures (no network, no snapshots). The
spot-check class queries the built confessions.sqlite and skips automatically
when it isn't present (fresh code-only checkout). Run
``python -m lampstand_corpus.cli build-confessions`` first to exercise them.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus.confessions import (
    CONFESSION_SOURCES,
    _belgic_osis_verserefs,
    _extract_belgic_proofs,
    _osis_to_verseref,
    parse_confession,
)
from lampstand_corpus.schema import Provenance, ResourceType

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "output" / "confessions.sqlite"

_PROV = Provenance(
    source="ccel:test", version="t", license="pd",
    retrieved="2026-06-10", url="u", checksum="c",
)


# --- OSIS mapping ------------------------------------------------------------
def test_osis_to_verseref_basic():
    vr = _osis_to_verseref("Bible:Rom.3.20")
    assert vr is not None
    assert (vr.book, vr.chapter, vr.verse_start) == ("ROM", 3, 20)


def test_osis_to_verseref_numbered_book():
    vr = _osis_to_verseref("Bible:1Cor.6.19")
    assert vr is not None and vr.book == "1CO"


def test_osis_to_verseref_rejects_unknown():
    assert _osis_to_verseref("Bible:Tob.1.1") is None
    assert _osis_to_verseref("garbage") is None


# --- WSC parser --------------------------------------------------------------
def test_parse_wsc_question_answer():
    content = (
        "<?xml version='1.0'?><ThML><ThML.body>"
        "<div1><div2 title='Questions 1-25'>"
        "<p><b><i>Q1: What is the chief end of man? </i></b></p>"
        "<p>A1: Man's chief end is to glorify God.</p>"
        "<p><b><i>Q2. What rule hath God given? </i></b></p>"
        "<p>A2. The Word of God.</p>"
        "</div2></div1></ThML.body></ThML>"
    )
    pc = parse_confession(CONFESSION_SOURCES["wsc"], _PROV, content)
    assert [c.key for c in pc.chunks] == ["1", "2"]
    assert pc.chunks[0].resource_type == ResourceType.CONFESSION
    assert "glorify God" in pc.chunks[0].text
    assert pc.chunks[0].meta["shortcode"] == "WSC"
    assert pc.chunks[0].meta["question"] == 1


# --- WLC parser --------------------------------------------------------------
def test_parse_wlc_question_answer():
    content = (
        "<?xml version='1.0'?><ThML><ThML.body><div1><div2 title='Questions 1-25'>"
        "<p><b><i>Question 1: What is the chief end of man?</i></b></p>"
        "<p>Answer: To glorify God, and fully to enjoy him forever.</p>"
        "</div2></div1></ThML.body></ThML>"
    )
    pc = parse_confession(CONFESSION_SOURCES["wlc"], _PROV, content)
    assert len(pc.chunks) == 1
    assert pc.chunks[0].key == "1"
    assert "fully to enjoy him" in pc.chunks[0].text


# --- WCF parser (original 1646/47 JSON) --------------------------------------
def _wcf_json(chapters: list[dict]) -> str:
    return json.dumps({"languages": {"eng": {"chapters": chapters}}})


def test_parse_wcf_sections_and_inline_proofs():
    content = _wcf_json([
        {"id": 1, "title": "Of the Holy Scripture", "sections": [
            {"id": 1, "text": "Although the light of nature (Rom. 2:14; Ps. 19:1)."},
            {"id": 2, "text": "Under the name of Holy Scripture."},
        ]},
        # Single-text chapter (like the real ch. 12, Of Adoption).
        {"id": 12, "title": "Of Adoption",
         "text": "All those that are justified (Eph. 1:5)."},
    ])
    pc = parse_confession(CONFESSION_SOURCES["wcf"], _PROV, content)
    keys = [c.key for c in pc.chunks]
    assert keys == ["1.1", "1.2", "12.1"]
    assert pc.chunks[0].meta["chapter_title"] == "Of the Holy Scripture"
    # inline parenthetical proof-texts become VerseRef dicts
    proofs = pc.chunks[0].meta["proof_texts"]
    assert {"book": "ROM", "chapter": 2, "verse_start": 14} in proofs
    # chapter 12's single paragraph becomes section 1
    assert pc.chunks[2].meta["chapter"] == 12 and pc.chunks[2].meta["section"] == 1


def test_parse_wcf_marks_1788_amendment_loci():
    content = _wcf_json([
        {"id": 23, "title": "Of the Civil Magistrate", "sections": [
            {"id": 3, "text": "The civil magistrate may not assume to himself."},
        ]},
        {"id": 24, "title": "Of Marriage and Divorce", "sections": [
            {"id": 4, "text": "Marriage ought not to be within the degrees."},
        ]},
    ])
    pc = parse_confession(CONFESSION_SOURCES["wcf"], _PROV, content)
    by_key = {c.key: c for c in pc.chunks}
    assert "amendment_1788" in by_key["23.3"].meta
    assert "amendment_1788" in by_key["24.4"].meta
    # 23 'Of the Civil Magistrate' / 24 'Of Marriage and Divorce' confirmed present
    assert by_key["23.3"].meta["chapter_title"] == "Of the Civil Magistrate"
    assert by_key["24.4"].meta["chapter_title"] == "Of Marriage and Divorce"


# --- 1689 LBCF parser --------------------------------------------------------
def test_parse_lbcf_chapters_and_paragraphs():
    content = json.dumps({
        "title": "Baptist Confession of Faith of 1689",
        "chapters": {
            "1": {"title": "Of the Holy Scriptures",
                  "paragraphs": {"1": "The Holy Scripture is the only rule.",
                                 "2": "Under the name of Holy Scripture."}},
            "26": {"title": "Of the Church",
                   "paragraphs": {"1": "The catholic or universal church."}},
        },
    })
    pc = parse_confession(CONFESSION_SOURCES["lbcf"], _PROV, content)
    keys = [c.key for c in pc.chunks]
    assert keys == ["1.1", "1.2", "26.1"]
    assert pc.chunks[0].meta["shortcode"] == "1689"
    # chapter 26 'Of the Church' present
    ch26 = [c for c in pc.chunks if c.meta["chapter"] == 26][0]
    assert ch26.meta["chapter_title"] == "Of the Church"


# --- Belgic parser -----------------------------------------------------------
def _belgic_parse_json(paras_html: str) -> str:
    return json.dumps({"parse": {"text": f"<div>{paras_html}</div>"}})


def test_parse_belgic_articles_roman_to_int():
    # Article I renders as a bare header + title paragraph; II carries its title
    # inline (the two real forms in the 1840 RPDC edition).
    html = (
        "<p>Article I.</p><p>That there is one only God.</p>"
        "<p>We all believe with the heart that there is one only God.</p>"
        "<p>II. By what means God is made known unto us.</p>"
        "<p>We know him by two means.</p>"
    )
    pc = parse_confession(CONFESSION_SOURCES["belgic"], _PROV,
                          _belgic_parse_json(html))
    assert [c.key for c in pc.chunks] == ["1", "2"]
    assert pc.chunks[0].meta["article"] == 1
    assert pc.chunks[0].meta["article_title"] == "That there is one only God."
    assert "one only God" in pc.chunks[0].text
    assert pc.chunks[1].meta["article_title"].startswith("By what means")
    assert pc.chunks[0].meta["shortcode"] == "BC"


# --- Belgic proof-texts (CCEL schaff/creeds3) --------------------------------
def test_belgic_osis_verserefs_single_range_and_multi():
    # Single ref.
    refs, un = _belgic_osis_verserefs("Bible:Eph.4.5")
    assert un == [] and len(refs) == 1
    assert (refs[0].book, refs[0].chapter, refs[0].verse_start) == ("EPH", 4, 5)
    # Same-chapter range collapses into verse_end.
    refs, un = _belgic_osis_verserefs("Bible:2Tim.3.15-2Tim.3.17")
    assert un == [] and refs[0].verse_end == 17
    # Multiple refs packed in ONE osisRef (space separated).
    refs, un = _belgic_osis_verserefs("Bible:Rom.7.8 Bible:Rom.7.10")
    assert un == [] and [r.verse_start for r in refs] == [8, 10]


def test_belgic_osis_verserefs_flags_unresolvable_never_guesses():
    # Chapter-only (no verse), unknown book, and OCR garbage -> unparsed, no ref.
    refs, un = _belgic_osis_verserefs("Bible:Ps.3")
    assert refs == [] and un == ["Ps.3"]
    refs, un = _belgic_osis_verserefs("Bible:Tob.1.1")
    assert refs == [] and un == ["Tob.1.1"]


_BELGIC_CREEDS3_FIXTURE = """<?xml version="1.0"?>
<ThML><ThML.body><div1>
<div2 title="The Belgic Confession. A.D. 1561. Revised 1619.">
 <table>
  <tr>
   <td><span lang="fr">A<span class="sc">rt.</span> I</span></td>
   <td>A<span class="sc">rt.</span> I.</td>
  </tr>
  <tr>
   <td><span lang="fr">Nous croyons<note n="1">
       <scripRef osisRef="Bible:Eph.4.5">Eph. iv. 5</scripRef>;
       <scripRef osisRef="Bible:Deut.6.4">Deut. vi. 4</scripRef>.</note></span></td>
   <td>We all believe<note n="1">
       <scripRef osisRef="Bible:Eph.4.5">Eph. iv. 5</scripRef></note></td>
  </tr>
  <tr>
   <td><span lang="fr">A<span class="sc">rt.</span> II</span></td>
   <td>A<span class="sc">rt.</span> II.</td>
  </tr>
  <tr>
   <td><span lang="fr">Par deux moyens<note n="2">
       <scripRef osisRef="Bible:Ps.19.2 Bible:Rom.1.20">two refs</scripRef>;
       <scripRef osisRef="Bible:Ps.3">bad</scripRef>.</note></span></td>
   <td>By two means</td>
  </tr>
 </table>
</div2>
</div1></ThML.body></ThML>"""


def test_extract_belgic_proofs_anchors_to_french_column_articles():
    flags: list[str] = []
    proofs = _extract_belgic_proofs(_BELGIC_CREEDS3_FIXTURE, flags)
    # Article 1: two French-column refs (English-column duplicate NOT double-counted
    # into a different article; French cell is the source of record).
    a1 = proofs[1]
    assert [(p["book"], p["chapter"], p["verse_start"]) for p in a1] == [
        ("EPH", 4, 5), ("DEU", 6, 4)]
    # Article 2: the multi-ref osisRef expands to two refs; the bad "Ps.3"
    # chapter-only token is FLAGGED and NOT attached.
    a2 = proofs[2]
    assert [(p["book"], p["chapter"], p["verse_start"]) for p in a2] == [
        ("PSA", 19, 2), ("ROM", 1, 20)]
    assert any("Ps.3" in f for f in flags)
    assert any("attached" in f for f in flags)


def test_extract_belgic_proofs_flags_ref_before_first_marker():
    # A note that appears before any "Art." marker is article-ambiguous ->
    # left UNATTACHED and flagged, never guessed onto article 1.
    xml = (
        '<?xml version="1.0"?><ThML><ThML.body><div1>'
        '<div2 title="The Belgic Confession. A.D. 1561.">'
        '<table><tr><td><span lang="fr">preface'
        '<note><scripRef osisRef="Bible:Gen.1.1">Gen 1:1</scripRef></note>'
        '</span></td></tr></table></div2></div1></ThML.body></ThML>'
    )
    flags: list[str] = []
    proofs = _extract_belgic_proofs(xml, flags)
    assert all(not v for v in proofs.values())  # nothing attached
    assert any("before any Art. marker" in f for f in flags)


# --- Heidelberg parser -------------------------------------------------------
def test_parse_heidelberg_question_lords_day_and_proofs():
    content = (
        "<?xml version='1.0'?><ThML><ThML.body><div1><div2 title='Part One'>"
        "<p class='left'>1. Lord's Day</p>"
        "<p class='left'><b>Question 1. </b> What is thy only comfort? "
        "<b>Answer.</b> That I belong to Christ. (a)</p>"
        "<p class='left'>(a) <scripRef osisRef='Bible:Rom.14.8'>Rom. 14:8</scripRef> "
        "text of the proof.</p>"
        "</div2></div1></ThML.body></ThML>"
    )
    pc = parse_confession(CONFESSION_SOURCES["heidelberg"], _PROV, content)
    assert len(pc.chunks) == 1
    ch = pc.chunks[0]
    assert ch.key == "1"
    assert ch.meta["lords_day"] == 1
    assert "belong to Christ" in ch.text
    assert ch.meta["proof_texts"] == [
        {"book": "ROM", "chapter": 14, "verse_start": 8}
    ]


def test_parse_heidelberg_out_of_sequence_lords_day_flagged():
    # The "2." typo where 29 belongs: counter advances, source not renumbered, flag.
    content = (
        "<?xml version='1.0'?><ThML><ThML.body><div1><div2 title='Part'>"
        "<p>28 Lord's Day</p>"
        "<p><b>Question 75. </b> A? <b>Answer.</b> B.</p>"
        "<p>2. Lord's Day</p>"
        "<p><b>Question 76. </b> C? <b>Answer.</b> D.</p>"
        "</div2></div1></ThML.body></ThML>"
    )
    pc = parse_confession(CONFESSION_SOURCES["heidelberg"], _PROV, content)
    lds = {c.meta["question"]: c.meta["lords_day"] for c in pc.chunks}
    assert lds[75] == 28 and lds[76] == 29   # advanced, not left at the source's "2"
    assert any("source typo" in f for f in pc.flags)


# --- Canons of Dort parser ---------------------------------------------------
def test_parse_dort_articles_rejections_and_keys():
    content = (
        "<?xml version='1.0'?><ThML><ThML.body>"
        "<div1 type='section' title='First Head of Doctrine.'>"
        "<h2>FIRST HEAD OF DOCTRINE.</h2>"
        "<div2 type='subsection' title='Divine Election and Reprobation'>"
        "<p><b>ARTICLE 1.</b> As all men have sinned in Adam.</p>"
        "<p><b>ARTICLE 2.</b> But in this the love of God was manifested.</p>"
        "</div2>"
        "<div2 type='subsection' title='Rejection of Errors'>"
        "<p>The true doctrine concerning election having been explained.</p>"
        "<p><b>PARAGRAPH 1.</b> Who teach that the will of God to save.</p>"
        "</div2></div1>"
        "<div1 type='section' title='Conclusion'>"
        "<h2>Conclusion</h2><p>This is the perspicuous, simple, and ingenuous "
        "declaration of the orthodox doctrine.</p></div1>"
        "</ThML.body></ThML>"
    )
    pc = parse_confession(CONFESSION_SOURCES["dort"], _PROV, content)
    keys = [c.key for c in pc.chunks]
    assert keys == ["h1.a1", "h1.a2", "h1.r1", "conclusion"]
    art = pc.chunks[0]
    assert art.meta["head"] == "1" and art.meta["kind"] == "article"
    assert art.meta["number"] == 1
    assert "sinned in Adam" in art.text
    rej = pc.chunks[2]
    assert rej.meta["kind"] == "rejection" and rej.meta["number"] == 1
    concl = pc.chunks[3]
    assert concl.meta["kind"] == "conclusion"
    assert concl.text.startswith("This is the perspicuous")
    assert pc.chunks[0].meta["shortcode"] == "Dort"


def test_parse_dort_combined_third_fourth_head_slug():
    content = (
        "<?xml version='1.0'?><ThML><ThML.body>"
        "<div1 type='section' title='Third and Fourth Heads of Doctrine.'>"
        "<div2 type='subsection' title='The Corruption of Man'>"
        "<p><b>ARTICLE 1.</b> Man was originally formed after the image of God.</p>"
        "</div2></div1></ThML.body></ThML>"
    )
    pc = parse_confession(CONFESSION_SOURCES["dort"], _PROV, content)
    assert pc.chunks[0].key == "h3-4.a1"
    assert pc.chunks[0].meta["head"] == "3-4"


# --- built-DB spot checks ----------------------------------------------------
@pytest.mark.skipif(not DB.exists(),
                    reason="confessions.sqlite not built; run build-confessions")
class TestConfessionsDB:
    @pytest.fixture(scope="class")
    def conn(self):
        c = sqlite3.connect(DB)
        yield c
        c.close()

    def test_seven_documents_present(self, conn):
        ids = {r[0] for r in conn.execute("SELECT id FROM document")}
        assert ids == {"wcf", "wlc", "wsc", "lbcf", "belgic",
                       "heidelberg", "dort"}

    def test_dort_article_and_rejection_counts(self, conn):
        # 59 positive articles + 34 rejection paragraphs + 1 Conclusion = 94 chunks.
        n = conn.execute(
            "SELECT COUNT(*) FROM section WHERE document='dort'"
        ).fetchone()[0]
        assert n == 94
        heads = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT key FROM section WHERE document='dort' "
                "AND key LIKE 'h%'"
            )
        }
        # the combined Third-and-Fourth head is represented by the 'h3-4' slug
        assert any(k.startswith("h3-4.") for k in heads)

    def test_catechism_counts(self, conn):
        for doc, expected in (("wlc", 196), ("wsc", 107), ("heidelberg", 129)):
            n = conn.execute(
                "SELECT COUNT(*) FROM section WHERE document=?", (doc,)
            ).fetchone()[0]
            assert n == expected, doc

    def test_wcf_has_all_33_original_chapters(self, conn):
        # Re-sourced to the original 1646/47: all 33 chapters incl. ch. 23 & 24.
        n = conn.execute(
            "SELECT COUNT(DISTINCT chapter) FROM section WHERE document='wcf'"
        ).fetchone()[0]
        assert n == 33

    def test_wcf_original_ch23_and_ch24(self, conn):
        t23 = conn.execute(
            "SELECT DISTINCT title FROM section WHERE document='wcf' AND chapter=23"
        ).fetchone()[0]
        t24 = conn.execute(
            "SELECT DISTINCT title FROM section WHERE document='wcf' AND chapter=24"
        ).fetchone()[0]
        assert t23 == "Of the Civil Magistrate"
        assert t24 == "Of Marriage and Divorce"

    def test_wcf_1788_amendment_loci_marked(self, conn):
        keys = {
            r[0] for r in conn.execute(
                "SELECT key FROM section WHERE document='wcf' "
                "AND amendment_1788 IS NOT NULL")
        }
        assert keys == {"20.4", "22.3", "23.3", "24.4", "25.6", "31.1", "31.2"}

    def test_wcf_1788_verbatim_text_only_on_amended_loci(self, conn):
        # Verbatim 1788 revised wording is populated ONLY on amended loci, and on
        # nothing beyond them. 24.4 is the one justified gap (this PD American
        # edition's ch. 24 is a later denominational rewrite, not 1788 text).
        keys = {
            r[0] for r in conn.execute(
                "SELECT key FROM section WHERE document='wcf' "
                "AND amendment_1788_text IS NOT NULL")
        }
        assert keys == {"20.4", "22.3", "23.3", "25.6", "31.1", "31.2"}
        # No other document carries verbatim 1788 text.
        other = conn.execute(
            "SELECT COUNT(*) FROM section WHERE document!='wcf' "
            "AND amendment_1788_text IS NOT NULL").fetchone()[0]
        assert other == 0

    def test_wcf_1788_verbatim_differs_from_original_and_drops_antichrist(self, conn):
        # 25.6: the revised text must NOT call the Pope 'Antichrist' / 'man of sin'
        # (the 1788 revision removed that), and must differ from the retained
        # original prose stored in `text`.
        orig, revised = conn.execute(
            "SELECT text, amendment_1788_text FROM section "
            "WHERE document='wcf' AND key='25.6'").fetchone()
        assert revised is not None and revised != orig
        assert "Antichrist" in orig and "man of sin" in orig
        assert "Antichrist" not in revised and "man of sin" not in revised
        assert "vicar of Christ" in revised  # the revised wording's hallmark
        # The denominational bracket markers must be stripped from the 1788 base.
        assert "[PCUS" not in revised and "[UPCUSA" not in revised

    def test_lbcf_has_32_chapters_incl_church(self, conn):
        n = conn.execute(
            "SELECT COUNT(DISTINCT chapter) FROM section WHERE document='lbcf'"
        ).fetchone()[0]
        assert n == 32
        t26 = conn.execute(
            "SELECT DISTINCT title FROM section WHERE document='lbcf' AND chapter=26"
        ).fetchone()[0]
        assert t26 == "Of the Church"

    def test_belgic_has_37_articles(self, conn):
        nums = sorted(
            r[0] for r in conn.execute(
                "SELECT article FROM section WHERE document='belgic'")
        )
        assert nums == list(range(1, 38))

    def test_belgic_proof_texts_supplemented_from_creeds3(self, conn):
        # The Belgic proof apparatus is merged from CCEL schaff/creeds3.
        n_with, n_refs = conn.execute(
            "SELECT COUNT(*) FILTER (WHERE proof_texts IS NOT NULL), "
            "COALESCE(SUM(json_array_length(proof_texts)),0) "
            "FROM section WHERE document='belgic'").fetchone()
        assert n_with == 34  # 34 of 37 articles carry proofs
        assert n_refs > 700  # ~799 stored refs
        # Articles 4,5,6 (the canon list) legitimately carry NO proofs.
        zero = sorted(r[0] for r in conn.execute(
            "SELECT article FROM section WHERE document='belgic' "
            "AND proof_texts IS NULL"))
        assert zero == [4, 5, 6]
        # Article 1 carries the DE NATURA DEI proofs (Eph 4:5 leads).
        a1 = json.loads(conn.execute(
            "SELECT proof_texts FROM section WHERE document='belgic' AND key='1'"
        ).fetchone()[0])
        assert (a1[0]["book"], a1[0]["chapter"], a1[0]["verse_start"]) == (
            "EPH", 4, 5)

    def test_wsc_q1_text(self, conn):
        t = conn.execute(
            "SELECT text FROM section WHERE document='wsc' AND key='1'"
        ).fetchone()[0]
        assert "chief end of man" in t and "glorify God" in t

    def test_proof_texts_resolve_against_bibles(self, conn):
        # Every proof-text ref must resolve in at least one bundled translation
        # of bibles.sqlite (the app renders verse text from our own Bibles).
        bibles = REPO_ROOT / "output" / "bibles.sqlite"
        if not bibles.exists():
            pytest.skip("bibles.sqlite not built")
        b = sqlite3.connect(bibles)
        translations = [r[0] for r in b.execute("SELECT id FROM translation")]
        rows = conn.execute(
            "SELECT document, key, proof_texts FROM section "
            "WHERE proof_texts IS NOT NULL").fetchall()
        unresolved = []
        for did, key, pj in rows:
            for pt in json.loads(pj):
                ok = any(
                    b.execute(
                        "SELECT 1 FROM verse WHERE translation=? AND book=? "
                        "AND chapter=? AND verse_start<=? AND verse_end>=? LIMIT 1",
                        (t, pt["book"], pt["chapter"],
                         pt["verse_start"], pt["verse_start"])).fetchone()
                    for t in translations
                )
                if not ok:
                    unresolved.append(f"{did}:{key} {pt['book']} "
                                      f"{pt['chapter']}:{pt['verse_start']}")
        b.close()
        # Exactly the KNOWN source-typo refs are allowed to resolve nowhere;
        # each is FLAGGED in the report for the architect, never dropped:
        #   * lbcf 5.4 -> PSA 1:21 (1689 source typo, known since P2)
        #   * wsc 1 -> 1CO 6:31 (Westminster-Standards JSON collapsed
        #     "1 Cor. 6:20; 10:31" into "6:20, 31", losing the chapter change)
        #   * belgic -> 8 CCEL schaff/creeds3 osisRef OCR defects (mis-numbered
        #     Ps./Gal. chapters/verses in the source's own osisRef attributes;
        #     carried faithfully and FLAGGED, never silently corrected).
        known = {
            "lbcf:5.4 PSA 1:21", "wsc:1 1CO 6:31",
            "belgic:3 PSA 2:19", "belgic:7 GAL 30:15", "belgic:12 PSA 4:10",
            "belgic:12 PSA 3:20", "belgic:13 PSA 4:9", "belgic:13 PSA 5:25",
            "belgic:14 PSA 49:21", "belgic:37 PSA 62:13",
        }
        assert set(unresolved) <= known, unresolved
