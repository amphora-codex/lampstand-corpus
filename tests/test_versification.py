"""Canonical reference spine + BSB Psalm-superscription map tests.

Unit tests run without snapshots. The DB-backed checks query the built
bibles.sqlite and skip automatically when it isn't present.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus.schema import VerseRef
from lampstand_corpus.versification import (
    BSB_PSALM_SUPERSCRIPTION_UNNUMBERED,
    BSB_PSALM_SUPERSCRIPTION_V1,
    is_superscription_verse,
    resolve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "output" / "bibles.sqlite"


# --- the map (unit) ----------------------------------------------------------
def test_superscription_set_size_and_membership():
    # 116 psalms carry a BSB \d \v 1 numbered superscription (derived from source).
    assert len(BSB_PSALM_SUPERSCRIPTION_V1) == 116
    for p in (3, 23, 51, 110, 145):
        assert p in BSB_PSALM_SUPERSCRIPTION_V1
    # Psalms with no superscription, and the two unnumbered-\d psalms, are excluded.
    for p in (1, 2, 33, 107):
        assert p not in BSB_PSALM_SUPERSCRIPTION_V1
    assert BSB_PSALM_SUPERSCRIPTION_UNNUMBERED == frozenset({1, 107})
    # The two sets are disjoint.
    assert not (BSB_PSALM_SUPERSCRIPTION_V1 & BSB_PSALM_SUPERSCRIPTION_UNNUMBERED)


def test_resolve_bsb_psalm_body_is_offset_zero():
    # Canonical (KJV) Ps 3:2 resolves to BSB 3:2 — body offset 0, NOT +1.
    rr = resolve("bsb", VerseRef(book="PSA", chapter=3, verse_start=2))
    assert (rr.book, rr.chapter, rr.verse) == ("PSA", 3, 2)
    rr8 = resolve("bsb", VerseRef(book="PSA", chapter=3, verse_start=8))
    assert (rr8.book, rr8.chapter, rr8.verse) == ("PSA", 3, 8)


def test_resolve_bsb_psalm_verse_one_is_superscription_folded():
    rr = resolve("bsb", VerseRef(book="PSA", chapter=51, verse_start=1))
    assert (rr.book, rr.chapter, rr.verse) == ("PSA", 51, 1)
    assert rr.note and "superscription" in rr.note.lower()
    assert is_superscription_verse("bsb", "PSA", 51, 1)
    assert not is_superscription_verse("bsb", "PSA", 51, 2)
    # A psalm with no superscription is not a superscription verse.
    assert not is_superscription_verse("bsb", "PSA", 1, 1)


def test_resolve_is_identity_for_other_translations_and_books():
    for t in ("kjv", "asv", "web", "bsb"):
        rr = resolve(t, VerseRef(book="ROM", chapter=9, verse_start=15))
        assert (rr.book, rr.chapter, rr.verse) == ("ROM", 9, 15)
    # Non-superscribed psalm in BSB is identity at verse 1 too.
    rr = resolve("bsb", VerseRef(book="PSA", chapter=1, verse_start=1))
    assert (rr.book, rr.chapter, rr.verse) == ("PSA", 1, 1)
    assert rr.note is None


# --- against the built DB ----------------------------------------------------
@pytest.mark.skipif(not DB.exists(), reason="bibles.sqlite not built")
class TestVersificationAgainstDB:
    @pytest.fixture(scope="class")
    def conn(self):
        c = sqlite3.connect(DB)
        yield c
        c.close()

    def test_bsb_superscribed_psalms_now_have_verse_one(self, conn):
        # The ingestion fix means every superscribed psalm carries a BSB verse 1.
        missing = []
        for p in sorted(BSB_PSALM_SUPERSCRIPTION_V1):
            row = conn.execute(
                "SELECT text FROM verse WHERE translation='bsb' AND book='PSA' "
                "AND chapter=? AND verse_start=1", (p,)).fetchone()
            if row is None or not row[0].strip():
                missing.append(p)
        assert missing == [], f"BSB Psalms missing a verse-1 body: {missing}"

    def test_bsb_verse_one_body_matches_kjv_verse_one(self, conn):
        # Canonical Ps N:1 (KJV body) resolves to BSB verse 1's BODY text (the
        # superscription is held separately), so they share salient words.
        import re

        def words(s):
            return set(re.findall(r"[a-z]{4,}", (s or "").lower()))

        for p in (3, 18, 23, 51):
            kjv = conn.execute(
                "SELECT text FROM verse WHERE translation='kjv' AND book='PSA' "
                "AND chapter=? AND verse_start=1", (p,)).fetchone()[0]
            bsb = conn.execute(
                "SELECT text FROM verse WHERE translation='bsb' AND book='PSA' "
                "AND chapter=? AND verse_start=1", (p,)).fetchone()[0]
            # Some shared content word beyond stop-words (e.g. LORD, mercy, shepherd).
            assert words(kjv) & words(bsb), f"Ps {p}: KJV v1 and BSB v1 share no word"

    def test_bsb_superscription_stored_separately_not_in_body(self, conn):
        # The superscription prose is in the `superscription` column, never the body.
        text, sup = conn.execute(
            "SELECT text, superscription FROM verse WHERE translation='bsb' "
            "AND book='PSA' AND chapter=51 AND verse_start=1").fetchone()
        assert sup is not None and "choirmaster" in sup.lower()
        assert "choirmaster" not in text.lower()
        assert "Have mercy" in text

    def test_canonical_psalm_refs_resolve_in_bsb_under_map(self, conn):
        # The previously-failing canonical Ps N:1 refs now resolve in BSB.
        from lampstand_corpus.versification import resolve as _resolve
        for (p, v) in [(19, 1), (77, 1), (110, 1), (51, 1), (127, 1)]:
            rr = _resolve("bsb", VerseRef(book="PSA", chapter=p, verse_start=v))
            row = conn.execute(
                "SELECT 1 FROM verse WHERE translation='bsb' AND book='PSA' "
                "AND chapter=? AND verse_start<=? AND verse_end>=?",
                (rr.chapter, rr.verse, rr.verse)).fetchone()
            assert row is not None, f"canonical PSA {p}:{v} did not resolve in BSB"
