"""Lexicon parser unit tests + built-DB spot checks (M1-P4).

Unit tests run on small inline fixtures (no network, no snapshots). The spot-check
class queries the built lexicons.sqlite and skips automatically when it isn't
present (fresh code-only checkout). Run
``python -m lampstand_corpus.cli build-lexicons`` first to exercise them.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus.lexicons import (
    LEXICON_SOURCES,
    THAYERS_FLAG,
    _extract_js_object,
    _normalize_strongs_key,
    _oshb_lemma_strongs,
    _oshb_osis_to_ref,
    _parse_lexical_index,
    parse_bdb,
    parse_strongs,
)
from lampstand_corpus.schema import Provenance
from lampstand_corpus.validate_lexicons import (
    validate_lexicon,
    validate_orphans,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "output" / "lexicons.sqlite"

_PROV = Provenance(
    source="openscriptures:test", version="t", license="CC-BY-SA",
    retrieved="2026-06-11", url="u", checksum="c",
)

_GREEK_JS = """
/* header comment */
var strongsGreekDictionary = {"G25":{"strongs_def":" to love","derivation":"perhaps from agan;","translit":"agapáō","lemma":"ἀγαπάω","kjv_def":"(be-)love(-ed)"},
"G26":{"lemma":"ἀγάπη","translit":"agápē","kjv_def":"(feast of) charity, dear, love","strongs_def":" love, i.e. affection or benevolence","derivation":"from G25 (ἀγαπάω);"}};
"""  # noqa: E501

_HEBREW_JS = """
var strongsHebrewDictionary = {"H1":{"lemma":"אָב","xlit":"ʼâb","pron":"awb","derivation":"a primitive word;","strongs_def":"father","kjv_def":"chief, father."},
"H08012":{"lemma":"x","xlit":"y","pron":"z","strongs_def":"Salmon","kjv_def":"Salmon."}};
"""  # noqa: E501

_BDB_XML = """<?xml version="1.0" encoding="UTF-8"?>
<lexicon xmlns="http://openscriptures.github.com/morphhb/namespace">
  <part id="a">
    <section id="a.aa">
      <entry id="a.ae.ab"><w>אָב</w> <pos>N</pos>
        <def>father</def>, <def>head of household</def></entry>
      <entry id="a.ab.ab"><w>אֵב</w> <def>freshness</def></entry>
      <entry id="a.zz.zz"><w>א</w> a cross-ref only</entry>
    </section>
  </part>
</lexicon>
"""

_BDB_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<index xmlns="http://openscriptures.github.com/morphhb/namespace">
  <part>
    <entry id="aac"><w>אָב</w><xref bdb="a.ae.ab" strong="1"/></entry>
    <entry id="aab"><w>אֵב</w><xref bdb="a.ab.ab" strong="3"/></entry>
  </part>
</index>
"""


# --- JS object extraction ----------------------------------------------------
def test_extract_js_object_strips_assignment_and_semicolon():
    obj = _extract_js_object(_GREEK_JS)
    data = json.loads(obj)
    assert "G25" in data and data["G25"]["translit"] == "agapáō"


# --- Strong's key normalization ----------------------------------------------
def test_normalize_strongs_key_padding():
    assert _normalize_strongs_key("H08012", "H") == "H8012"
    assert _normalize_strongs_key("G25", "G") == "G25"
    assert _normalize_strongs_key("garbage", "G") is None


# --- Strong's parser ---------------------------------------------------------
def test_parse_strongs_greek():
    src = LEXICON_SOURCES["strongs-greek"]
    pl = parse_strongs(src, _PROV, _GREEK_JS)
    keys = {e.strongs for e in pl.entries}
    assert keys == {"G25", "G26"}
    g25 = next(e for e in pl.entries if e.strongs == "G25")
    assert g25.lemma and g25.definition == "to love"
    assert g25.pronunciation is None  # Greek set has no pron field


def test_parse_strongs_hebrew_has_pronunciation():
    src = LEXICON_SOURCES["strongs-hebrew"]
    pl = parse_strongs(src, _PROV, _HEBREW_JS)
    h1 = next(e for e in pl.entries if e.strongs == "H1")
    assert h1.pronunciation == "awb"
    assert h1.translit == "ʼâb"
    # leading-zero key normalized
    assert any(e.strongs == "H8012" for e in pl.entries)


# --- BDB + lexical index -----------------------------------------------------
def test_parse_lexical_index_maps_bdb_to_strongs():
    mapping = _parse_lexical_index(_BDB_INDEX)
    assert mapping == {"a.ae.ab": "H1", "a.ab.ab": "H3"}


def test_parse_bdb_links_strongs_and_flags_unlinked():
    src = LEXICON_SOURCES["bdb"]
    pl = parse_bdb(src, _PROV, _BDB_XML, _BDB_INDEX)
    by_id = {e.raw_key: e for e in pl.entries}
    assert by_id["a.ae.ab"].strongs == "H1"
    assert by_id["a.ae.ab"].definition == "father; head of household"
    # The cross-ref-only entry has no Strong's link and is kept (not guessed).
    assert by_id["a.zz.zz"].strongs == ""
    assert any("no Strong's number" in f for f in pl.flags)


# --- OSHB lemma -> Strong's --------------------------------------------------
def test_oshb_lemma_strongs():
    assert _oshb_lemma_strongs("c/1961") == ["H1961"]
    assert _oshb_lemma_strongs("6965 b") == ["H6965"]
    assert _oshb_lemma_strongs("l") == []  # bare particle prefix, no Strong's


def test_oshb_osis_to_ref():
    vr = _oshb_osis_to_ref("Ruth.1.1")
    assert vr is not None and (vr.book, vr.chapter, vr.verse_start) == ("RUT", 1, 1)
    assert _oshb_osis_to_ref("Tob.1.1") is None


# --- validation --------------------------------------------------------------
def test_validate_lexicon_flags_coverage_gaps():
    src = LEXICON_SOURCES["strongs-greek"]
    pl = parse_strongs(src, _PROV, _GREEK_JS)  # only G25, G26
    rep = validate_lexicon(pl)
    assert rep.n_entries == 2
    assert rep.coverage_gaps  # G1..G24, G27.. are missing -> flagged


def test_validate_orphans_detects_missing_dict_entry():
    src = LEXICON_SOURCES["strongs-greek"]
    pl = parse_strongs(src, _PROV, _GREEK_JS)  # has G25, G26 dict entries
    # A fake BDB entry referencing H1 (no Hebrew dict loaded) is an orphan.
    bdb_src = LEXICON_SOURCES["bdb"]
    bdb = parse_bdb(bdb_src, _PROV, _BDB_XML, _BDB_INDEX)  # links H1, H3
    orphans = validate_orphans({"strongs-greek": pl, "bdb": bdb}, {})
    assert "H1" in orphans.from_bdb and "H3" in orphans.from_bdb


def test_thayers_is_flagged_not_fabricated():
    assert "Thayer" in THAYERS_FLAG and "OMITTED" in THAYERS_FLAG


# --- built-DB spot checks ----------------------------------------------------
@pytest.mark.skipif(not DB.exists(), reason="lexicons.sqlite not built")
class TestBuiltLexicons:
    @pytest.fixture(scope="class")
    def conn(self):
        c = sqlite3.connect(DB)
        yield c
        c.close()

    def test_greek_and_hebrew_present(self, conn):
        langs = {r[0] for r in conn.execute(
            "SELECT DISTINCT language FROM lexicon")}
        assert {"greek", "hebrew"} <= langs

    def test_strongs_greek_count_in_range(self, conn):
        n = conn.execute(
            "SELECT COUNT(*) FROM entry WHERE lexicon='strongs-greek'"
        ).fetchone()[0]
        assert 5000 < n <= 5624

    def test_strongs_hebrew_count_in_range(self, conn):
        n = conn.execute(
            "SELECT COUNT(*) FROM entry WHERE lexicon='strongs-hebrew'"
        ).fetchone()[0]
        assert 8000 < n <= 8700

    def test_known_lemma_g26_agape(self, conn):
        row = conn.execute(
            "SELECT lemma, definition FROM entry "
            "WHERE lexicon='strongs-greek' AND strongs='G26'"
        ).fetchone()
        assert row and "love" in (row[1] or "").lower()

    def test_known_lemma_h1_father(self, conn):
        row = conn.execute(
            "SELECT definition FROM entry "
            "WHERE lexicon='strongs-hebrew' AND strongs='H1'"
        ).fetchone()
        assert row and "father" in (row[0] or "").lower()

    def test_text_license_recorded(self, conn):
        rows = conn.execute(
            "SELECT id, text_license FROM lexicon").fetchall()
        assert all(tl for _, tl in rows)
        assert any("public domain" in tl.lower() for _, tl in rows)
