"""Spurgeon (Treasury of David) OCR parser unit tests + built-DB spot checks.

Unit tests run on small inline OCR-shaped fixtures (no network, no snapshots).
The spot-check class queries the built commentaries.sqlite and skips when it isn't
present. Run ``python -m lampstand_corpus.cli snapshot-spurgeon`` then
``build-commentaries`` to exercise the DB checks.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus.schema import Provenance, ResourceType
from lampstand_corpus.spurgeon import (
    MISSING_VOLUME,
    REJECTED_DUPLICATES,
    SPURGEON_VOLUMES,
    SpurgeonSource,
    SpurgeonVolume,
    _clean_block,
    _match_ordinal,
    _ordinal_word,
    _roman_to_int,
    _split_components,
    _split_psalms,
    _split_verse_paragraphs,
    _strip_boilerplate,
    parse_spurgeon,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "output" / "commentaries.sqlite"

_PROV = Provenance(
    source="ia:spurgeon:test", version="t", license="pd",
    retrieved="2026-06-10", url="u", checksum="cafe",
)


def _parse_fixture(stem: str, lo: int, hi: int, body: str):
    """Parse a single fixture volume by temporarily swapping the volume table."""
    import lampstand_corpus.spurgeon as sp

    saved = sp.SPURGEON_VOLUMES
    vol = SpurgeonVolume(stem, "ia-test", lo, hi)
    sp.SPURGEON_VOLUMES = (vol,)
    try:
        src = SpurgeonSource()
        return parse_spurgeon(src, {stem: _PROV}, {stem: body})
    finally:
        sp.SPURGEON_VOLUMES = saved


# --- roman + ordinal mapping -------------------------------------------------
def test_roman_to_int():
    assert _roman_to_int("XXIII") == 23
    assert _roman_to_int("CXIX") == 119
    assert _roman_to_int("CL") == 150
    assert _roman_to_int("MM") is None  # 2000, out of psalm range


def test_ordinal_word_forms():
    assert _ordinal_word(7) == "SEVENTH"
    assert _ordinal_word(23) == "TWENTYTHIRD"
    assert _ordinal_word(131) == "HUNDREDANDTHIRTYFIRST"


def test_match_ordinal_exact_and_fuzzy():
    assert _match_ordinal("TWENTY-THIRD") == 23
    assert _match_ordinal("HUNDRED  AND  THIRTY-FIRST") == 131
    # OCR garble within tolerance still resolves.
    assert _match_ordinal("TWEXTY-THIRD") == 23
    # Pure noise resolves to nothing (no guess).
    assert _match_ordinal("QZXQW") is None


# --- OCR cleaning ------------------------------------------------------------
def test_strip_boilerplate_removes_google_front_matter():
    raw = (
        "Google This is a digital copy of a book that survived. Original from "
        "Harvard.\n\nPSALM I.\nBlessed is the man."
    )
    out = _strip_boilerplate(raw)
    assert "digital copy" not in out
    assert "Google" not in out
    assert out.lstrip().startswith("PSALM I.")


def test_clean_block_rejoins_hyphenation_and_strips_running_header():
    block = (
        "the paths of righteous- \n ness for his name. \n"
        "PSALM THE TWENTY-THIRD. 399\n more comment here."
    )
    out = _clean_block(block)
    assert "righteousness" in out
    assert "TWENTY-THIRD" not in out  # running header stripped


# --- segmentation ------------------------------------------------------------
def test_split_psalms_uses_line_isolated_caps_head_not_inline_ref():
    text = (
        "PSALM VI.\nSome comment. As we read in Psalm xliv. 17 we have not "
        "forgotten thee.\nPSALM VII.\nFresh comment."
    )
    blocks, recovered = _split_psalms(text, (6, 7))
    nums = sorted(n for n, _ in blocks)
    assert nums == [6, 7]  # the inline 'Psalm xliv. 17' is NOT a head
    assert recovered == []


def test_split_psalms_recovers_unreadable_head_from_running_ordinal():
    # Psalm 7's all-caps roman head is OCR'd unreadable ('vir'), but its running
    # header names it — recovery should find it and report it as recovered.
    text = (
        "PSALM VI.\nComment on six.\n"
        "PSALM vir.\nLede.\nPSALM THE SEVENTH. 81\nComment on seven."
    )
    blocks, recovered = _split_psalms(text, (6, 7))
    nums = sorted(n for n, _ in blocks)
    assert 7 in nums
    assert 7 in recovered


def test_split_components_separates_the_four_sections():
    block = (
        "PSALM XXIII.\nTitle argument here.\n"
        "EXPOSITION.\nThe Lord is my shepherd.\n"
        "EXPLANATORY NOTES AND QUAINT SAYINGS.\nVerse 1. A note.\n"
        "HINTS TO THE VILLAGE PREACHER.\nVerse 1. A hint.\n"
        "WORKS UPON THE TWENTY-THIRD PSALM.\nA book list."
    )
    comps = dict(_split_components(block))
    assert set(comps) == {"title", "exposition", "notes", "hints", "works"}


def test_verse_cue_gated_by_psalm_length():
    # '80.' is a page number, not a verse; Psalm with 6 verses must reject it.
    text = 'Lede paragraph.\n1. " A real verse-one comment.\n80. Not a verse.'
    paras = _split_verse_paragraphs(text, max_verse=6)
    verses = [v for v, _ in paras]
    assert 1 in verses
    assert 80 not in verses  # rejected; stays with verse 1's paragraph


# --- full parse on a fixture volume ------------------------------------------
def test_parse_fixture_anchors_to_psalm_verses():
    body = (
        "Google This is a digital copy of a book.\n\n"
        "PSALM I.\nBlessed is the man argument.\n"
        "EXPOSITION.\n1 Blessed is the man.\n"
        '1. " Blessed is the man," this is the comment on verse one.\n'
        "PSALM II.\nWhy do the heathen rage.\n"
        "EXPOSITION.\n"
        '1. " Why do the heathen," comment on psalm two verse one.'
    )
    pc = _parse_fixture("tod1", 1, 2, body)
    refs = {(c.ref.chapter, c.ref.verse_start) for c in pc.chunks}
    assert (1, 1) in refs
    assert (2, 1) in refs
    assert all(c.ref.book == "PSA" for c in pc.chunks)
    assert all(c.resource_type == ResourceType.COMMENTARY for c in pc.chunks)
    assert pc.psalms_seen == {1, 2}
    # The first chunk's shortcode is Spurgeon's citation prefix.
    assert pc.chunks[0].meta["shortcode"] == "CHS"


# --- source-registration + gap guards ----------------------------------------
def test_six_volumes_and_missing_104_118():
    stems = [v.stem for v in SPURGEON_VOLUMES]
    assert stems == ["tod1", "tod2", "tod3", "tod4", "tod6", "tod7"]  # tod5 absent
    lo, hi, _why = MISSING_VOLUME
    assert (lo, hi) == (104, 118)
    # No available volume covers any of 104-118.
    for v in SPURGEON_VOLUMES:
        assert not (v.psalm_first <= 104 <= v.psalm_last)
        assert not (v.psalm_first <= 118 <= v.psalm_last)


def test_volumes_tile_1_103_and_119_150_without_overlap():
    covered: set[int] = set()
    for v in SPURGEON_VOLUMES:
        rng = set(range(v.psalm_first, v.psalm_last + 1))
        assert not (covered & rng), "volume ranges must not overlap"
        covered |= rng
    assert covered == set(range(1, 104)) | set(range(119, 151))


def test_rejected_duplicates_are_not_in_chosen_volumes():
    chosen = {v.identifier for v in SPURGEON_VOLUMES}
    assert not (chosen & set(REJECTED_DUPLICATES))


def test_spurgeon_source_in_combined_registry_only():
    from lampstand_corpus.commentaries import (
        COMMENTARY_SOURCES,
        all_commentary_sources,
    )

    # Not in the CCEL-only dict the snapshot loop iterates...
    assert "spurgeon" not in COMMENTARY_SOURCES
    # ...but present in the combined registry the build resolves by id.
    combined = all_commentary_sources()
    assert combined["spurgeon"].shortcode == "CHS"
    assert set(combined) == {"henry", "jfb", "calvin", "spurgeon"}


# --- built-DB spot checks ----------------------------------------------------
@pytest.mark.skipif(not DB.exists(),
                    reason="commentaries.sqlite not built; run build-commentaries")
class TestSpurgeonDB:
    @pytest.fixture(scope="class")
    def conn(self):
        c = sqlite3.connect(DB)
        yield c
        c.close()

    def test_spurgeon_present_psalms_only(self, conn):
        books = {r[0] for r in conn.execute(
            "SELECT DISTINCT book FROM comment WHERE commentator='spurgeon'")}
        assert books == {"PSA"}

    def test_psalm_coverage_excludes_missing_volume(self, conn):
        psalms = {r[0] for r in conn.execute(
            "SELECT DISTINCT chapter FROM comment WHERE commentator='spurgeon'")}
        # The 104-118 volume is absent from the scan set.
        assert not (psalms & set(range(104, 119)))
        # But the available ranges are well-covered.
        assert {1, 23, 119, 150} <= psalms

    def test_four_components_present(self, conn):
        comps = {r[0] for r in conn.execute(
            "SELECT DISTINCT component FROM comment WHERE commentator='spurgeon'")}
        assert {"exposition", "notes", "hints", "works"} <= comps

    def test_no_verse_anchor_exceeds_psalm_length(self, conn):
        from lampstand_corpus import books

        counts = books.VERSE_COUNTS["PSA"]
        rows = conn.execute(
            "SELECT chapter, verse_start FROM comment "
            "WHERE commentator='spurgeon' AND verse_start>0"
        ).fetchall()
        assert rows
        for chap, v in rows:
            assert v <= counts[chap - 1], (chap, v)
