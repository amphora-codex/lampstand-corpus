"""End-to-end spot checks against the built bibles.sqlite.

These verify the *whole* pipeline (snapshot -> parse -> build) produced correct,
recognizable Scripture. They are skipped automatically when the snapshots or the
built database aren't present (e.g. a fresh clone before `cli build`), so CI on
code-only checkouts still passes. Run `python -m lampstand_corpus.cli build`
first to exercise them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "output" / "bibles.sqlite"

pytestmark = pytest.mark.skipif(
    not DB.exists(), reason="bibles.sqlite not built; run `cli build` first"
)


def _verse(conn, translation, book, chapter, verse) -> str | None:
    row = conn.execute(
        "SELECT text FROM verse WHERE translation=? AND book=? AND chapter=? "
        "AND verse_start=?",
        (translation, book, chapter, verse),
    ).fetchone()
    return row[0] if row else None


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(DB)
    yield c
    c.close()


def test_all_four_translations_present(conn):
    ids = {r[0] for r in conn.execute("SELECT id FROM translation")}
    assert ids == {"bsb", "kjv", "asv", "web"}


def test_66_books_each_translation(conn):
    for tid in ("bsb", "kjv", "asv", "web"):
        n = conn.execute(
            "SELECT COUNT(DISTINCT book) FROM verse WHERE translation=?", (tid,)
        ).fetchone()[0]
        assert n == 66, tid


def test_john_3_16_known_text(conn):
    # Distinctive phrasing per translation; confirms correct verse addressing.
    assert "God so loved the world" in _verse(conn, "kjv", "JHN", 3, 16)
    assert "only begotten Son" in _verse(conn, "kjv", "JHN", 3, 16)
    assert "God so loved the world" in _verse(conn, "bsb", "JHN", 3, 16)
    assert "one and only Son" in _verse(conn, "bsb", "JHN", 3, 16)
    assert "God so loved the world" in _verse(conn, "asv", "JHN", 3, 16)


def test_psalm_119_105_known_text(conn):
    for tid in ("kjv", "bsb", "asv", "web"):
        t = _verse(conn, tid, "PSA", 119, 105)
        assert t is not None
        assert "lamp" in t.lower() and "feet" in t.lower(), tid


def test_genesis_1_1(conn):
    assert "beginning" in _verse(conn, "kjv", "GEN", 1, 1).lower()
    assert "God created" in _verse(conn, "kjv", "GEN", 1, 1)


def test_kjv_red_letter_present(conn):
    # KJV marks words of Christ; a known red-letter verse should carry spans.
    row = conn.execute(
        "SELECT red_letter, wj_spans FROM verse WHERE translation='kjv' "
        "AND book='JHN' AND chapter=11 AND verse_start=25"
    ).fetchone()
    assert row[0] == 1 and row[1] is not None  # "I am the resurrection..."


def test_asv_has_no_red_letter(conn):
    # The ASV eBible edition carries no \wj markup — expected, not a defect.
    n = conn.execute(
        "SELECT COUNT(*) FROM verse WHERE translation='asv' AND red_letter=1"
    ).fetchone()[0]
    assert n == 0


def test_no_empty_text_except_known_variants(conn):
    # Every stored verse has text, except documented textual-variant omissions.
    rows = conn.execute(
        "SELECT translation, book, chapter, verse_start FROM verse WHERE text=''"
    ).fetchall()
    # All empties must be among the known omitted/relocated variant verses.
    known = {
        ("asv", "MAT", 17, 21), ("asv", "MAT", 18, 11), ("asv", "MAT", 23, 14),
        ("asv", "MRK", 7, 16), ("asv", "MRK", 9, 44), ("asv", "MRK", 9, 46),
        ("asv", "MRK", 11, 26), ("asv", "MRK", 15, 28), ("asv", "LUK", 17, 36),
        ("asv", "LUK", 23, 17), ("asv", "JHN", 5, 4), ("asv", "ACT", 8, 37),
        ("asv", "ACT", 15, 34), ("asv", "ACT", 24, 7), ("asv", "ACT", 28, 29),
        ("asv", "ROM", 16, 24),
        ("web", "LUK", 17, 36), ("web", "ACT", 8, 37), ("web", "ACT", 15, 34),
        ("web", "ACT", 24, 7), ("web", "ROM", 16, 25),
    }
    unexpected = [tuple(r) for r in rows if tuple(r) not in known]
    assert not unexpected, f"unexpected empty verses: {unexpected}"
