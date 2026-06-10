"""Confession/catechism parser unit tests + built-DB spot checks.

Unit tests run on small inline ThML fixtures (no network, no snapshots). The
spot-check class queries the built confessions.sqlite and skips automatically
when it isn't present (fresh code-only checkout). Run
``python -m lampstand_corpus.cli build-confessions`` first to exercise them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus.confessions import (
    CONFESSION_SOURCES,
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


# --- WCF parser --------------------------------------------------------------
def test_parse_wcf_original_chapter_mapping_and_skip():
    content = (
        "<?xml version='1.0'?><ThML><ThML.body><div1>"
        "<div2 title='Chapter 1'><h1>Of the Holy Scripture</h1>"
        "<p>1. Although the light of nature.</p>"
        "<p>2. Under the name of Holy Scripture.</p></div2>"
        # A modern-only added chapter (orig 34) must be skipped + flagged.
        "<div2 title='Chapter 9 (34)'><h1>Of the Holy Spirit</h1>"
        "<p>1. The Holy Spirit.</p></div2>"
        "</div1></ThML.body></ThML>"
    )
    pc = parse_confession(CONFESSION_SOURCES["wcf"], _PROV, content)
    keys = [c.key for c in pc.chunks]
    assert keys == ["1.1", "1.2"]            # only the original chapter kept
    assert pc.chunks[0].meta["chapter"] == 1
    assert pc.chunks[0].meta["chapter_title"] == "Of the Holy Scripture"
    assert any("Chapter 9 (34)" in f for f in pc.flags)


def test_parse_wcf_duplicate_section_disambiguated_and_flagged():
    # "Chapter 21 (19)": the (orig) number 19 is the original WCF chapter — the CCEL
    # source mislabels its 7th section as a second "6.". Keep both, disambiguate, flag.
    content = (
        "<?xml version='1.0'?><ThML><ThML.body><div1>"
        "<div2 title='Chapter 21 (19)'><h1>Of the Law of God</h1>"
        "<p>6. Although true believers be not under the law.</p>"
        "<p>6. Neither are the forementioned uses of the law.</p></div2>"
        "</div1></ThML.body></ThML>"
    )
    pc = parse_confession(CONFESSION_SOURCES["wcf"], _PROV, content)
    keys = [c.key for c in pc.chunks]
    assert "19.6" in keys and "19.6#2" in keys  # neither dropped, no key collision
    assert any("repeated section number 6" in f for f in pc.flags)


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

    def test_five_documents_present(self, conn):
        ids = {r[0] for r in conn.execute("SELECT id FROM document")}
        assert ids == {"wcf", "wlc", "wsc", "heidelberg", "dort"}

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

    def test_wcf_has_32_recovered_chapters(self, conn):
        # Original ch 24 is unrecoverable from the CCEL modern edition (flagged),
        # so 32 of 33 are present — documents the known, flagged gap.
        n = conn.execute(
            "SELECT COUNT(DISTINCT chapter) FROM section WHERE document='wcf'"
        ).fetchone()[0]
        assert n == 32

    def test_wsc_q1_text(self, conn):
        t = conn.execute(
            "SELECT text FROM section WHERE document='wsc' AND key='1'"
        ).fetchone()[0]
        assert "chief end of man" in t and "glorify God" in t

    def test_heidelberg_proof_texts_resolve_to_canon(self, conn):
        import json

        from lampstand_corpus import books
        rows = conn.execute(
            "SELECT proof_texts FROM section WHERE document='heidelberg' "
            "AND proof_texts IS NOT NULL"
        ).fetchall()
        assert rows, "expected Heidelberg proof-texts in the DB"
        for (pt_json,) in rows:
            for pt in json.loads(pt_json):
                assert pt["book"] in books.CANON
