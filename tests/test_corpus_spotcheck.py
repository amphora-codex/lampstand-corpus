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


def _row(conn, translation, book, chapter, verse):
    """Full omitted-aware row (text, omitted, source_note) or None."""
    return conn.execute(
        "SELECT text, omitted, source_note FROM verse WHERE translation=? "
        "AND book=? AND chapter=? AND verse_start=?",
        (translation, book, chapter, verse),
    ).fetchone()


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


def test_empty_text_rows_are_flagged_or_known(conn):
    # Every empty-text row is either an omitted=1 critical-text variant, or the
    # one documented versification artifact (WEB's relocated Romans doxology).
    rows = conn.execute(
        "SELECT translation, book, chapter, verse_start, omitted FROM verse "
        "WHERE text=''"
    ).fetchall()
    known_non_omitted = {("web", "ROM", 16, 25)}  # doxology renumbering, not omission
    for tid, b, ch, v, omitted in rows:
        if (tid, b, ch, v) in known_non_omitted:
            assert omitted == 0
        else:
            assert omitted == 1, f"empty but not flagged omitted: {tid} {b} {ch}:{v}"


def test_matt_18_11_omitted_in_critical_text(conn):
    # The classic "missing verse." Critical-text translations (BSB, ASV) carry it
    # as an omitted=1 empty row; TR/majority translations (KJV, WEB) keep the text.
    for tid in ("bsb", "asv"):
        text, omitted, _note = _row(conn, tid, "MAT", 18, 11)
        assert text == "" and omitted == 1, tid
    for tid in ("kjv", "web"):
        text, omitted, _note = _row(conn, tid, "MAT", 18, 11)
        assert text and omitted == 0, tid
        assert "save" in text.lower(), tid


def test_john_3_16_not_omitted_anywhere(conn):
    for tid in ("bsb", "kjv", "asv", "web"):
        text, omitted, _note = _row(conn, tid, "JHN", 3, 16)
        assert text and omitted == 0, tid


def test_every_disputed_reference_resolves_in_all_translations(conn):
    # Uniform addressability: each disputed reference must return a row in every
    # translation — the union resolves everywhere, no NULL rows.
    from lampstand_corpus import books

    for b, ch, v in books.OMITTED_VARIANTS:
        for tid in ("bsb", "kjv", "asv", "web"):
            row = _row(conn, tid, b, ch, v)
            assert row is not None, f"{tid} missing row for {b} {ch}:{v}"


def test_asv_omitted_verses_carry_source_note(conn):
    # ASV hangs an "ancient authorities insert…" footnote on each omitted verse;
    # we recover it as source_note.
    _t, omitted, note = _row(conn, "asv", "MAT", 17, 21)
    assert omitted == 1
    assert note and "authorities" in note.lower()
    assert "\\f" not in note and "\\ft" not in note  # markup stripped


def test_bsb_matt_18_11_has_recovered_source_note(conn):
    # BSB drops the omitted row entirely and carries the verse's wording on the
    # PRECEDING verse's footnote via an embedded \fv 11\fv* marker. P1.2 recovers
    # it and attaches it as source_note — so it must now be non-null and clean.
    text, omitted, note = _row(conn, "bsb", "MAT", 18, 11)
    assert text == "" and omitted == 1
    assert note is not None
    assert "Son of Man came to save" in note
    assert "\\f" not in note and "\\fv" not in note  # markup stripped


def test_bsb_all_omitted_verses_attributed(conn):
    # All 16 BSB critical-text omissions are recoverable via \fv markers; target
    # is 16/16. If the source ever drops a \fv segment, this surfaces it.
    rows = conn.execute(
        "SELECT source_note FROM verse WHERE translation='bsb' AND omitted=1"
    ).fetchall()
    assert len(rows) == 16
    assert all(r[0] for r in rows), "a BSB omitted verse lost its source_note"
