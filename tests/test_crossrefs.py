"""Treasury of Scripture Knowledge cross-reference parsing + validation tests.

Unit tests run without snapshots (they parse synthetic OSIS lines). The DB-backed
checks query the built crossrefs.sqlite / bibles.sqlite and skip automatically
when those aren't present.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus.crossrefs import (
    CROSSREFS_ATTRIBUTION,
    CROSSREFS_LICENSE,
    OSIS_TO_USFM,
    CanonicalPoint,
    parse_crossrefs,
    point_resolves,
)
from lampstand_corpus.schema import Provenance
from lampstand_corpus.validate_crossrefs import validate_crossrefs

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "output" / "crossrefs.sqlite"
BIBLES_DB = REPO_ROOT / "output" / "bibles.sqlite"

HEADER = "From Verse\tTo Verse\tVotes\t#www.openbible.info CC-BY 2026-06-08"


def _prov() -> Provenance:
    return Provenance(
        source="openbible:tsk",
        version="test",
        license=CROSSREFS_LICENSE,
        retrieved="2026-06-10",
        url="https://a.openbible.info/data/cross-references.zip",
        checksum="0" * 64,
    )


def _parse(*lines: str):
    return parse_crossrefs("\n".join([HEADER, *lines]), _prov())


# --- OSIS book map -----------------------------------------------------------
def test_osis_map_is_complete_and_canonical():
    from lampstand_corpus import books

    # Exactly the 66-book canon, mapping onto valid USFM ids.
    assert len(OSIS_TO_USFM) == 66
    assert set(OSIS_TO_USFM.values()) == set(books.ORDER)
    # No duplicate USFM target.
    assert len(set(OSIS_TO_USFM.values())) == 66


# --- parsing -----------------------------------------------------------------
def test_parse_single_verse_target():
    res = _parse("Gen.1.1\tEccl.12.1\t26")
    assert len(res.refs) == 1
    cr = res.refs[0]
    assert cr.source.as_tuple() == ("GEN", 1, 1)
    assert cr.target_start.as_tuple() == ("ECC", 12, 1)
    assert cr.target_end.as_tuple() == ("ECC", 12, 1)
    assert not cr.is_range
    assert cr.votes == 26
    assert cr.rank == 1


def test_parse_range_target_keeps_both_endpoints():
    res = _parse("Rev.22.21\tEph.6.23-Eph.6.24\t2")
    cr = res.refs[0]
    assert cr.is_range
    assert cr.target_start.as_tuple() == ("EPH", 6, 23)
    assert cr.target_end.as_tuple() == ("EPH", 6, 24)


def test_parse_book_crossing_range():
    res = _parse("Num.3.1\tLev.27.34-Num.1.1\t5")
    cr = res.refs[0]
    assert cr.is_range
    assert cr.target_start.as_tuple() == ("LEV", 27, 34)
    assert cr.target_end.as_tuple() == ("NUM", 1, 1)


def test_negative_votes_sign_preserved():
    res = _parse("Gen.1.1\tJohn.1.1\t-86")
    assert res.refs[0].votes == -86


def test_rank_is_per_source_and_in_file_order():
    res = _parse(
        "Gen.1.1\tNeh.9.6\t100",
        "Gen.1.1\tRev.14.7\t81",
        "Gen.1.2\tPs.104.30\t60",
    )
    ranks = {(cr.source.as_tuple(), cr.target_start.as_tuple()): cr.rank
             for cr in res.refs}
    assert ranks[(("GEN", 1, 1), ("NEH", 9, 6))] == 1
    assert ranks[(("GEN", 1, 1), ("REV", 14, 7))] == 2
    assert ranks[(("GEN", 1, 2), ("PSA", 104, 30))] == 1


def test_unparseable_lines_are_recorded_not_dropped_silently():
    res = _parse(
        "Gen.1.1\tEccl.12.1\tnotanumber",   # bad votes
        "Gen.1.1\tEccl.12.1",               # too few columns
        "Foo.1.1\tEccl.12.1\t3",            # unknown source book
        "Gen.1.1\tFoo.2.2\t3",              # unknown target book
    )
    assert res.refs == []
    assert len(res.unparsed) == 4


# --- point resolution --------------------------------------------------------
def test_point_resolves_against_kjv_spine():
    assert point_resolves(CanonicalPoint("GEN", 1, 1))
    assert point_resolves(CanonicalPoint("PSA", 119, 176))
    # Out of range: Psalm 119 has 176 verses, not 177.
    assert not point_resolves(CanonicalPoint("PSA", 119, 177))
    # 3 John has 14 verses in the KJV spine — v.15 does not resolve.
    assert point_resolves(CanonicalPoint("3JN", 1, 14))
    assert not point_resolves(CanonicalPoint("3JN", 1, 15))
    # Unknown book / chapter.
    assert not point_resolves(CanonicalPoint("GEN", 51, 1))


def test_omitted_variant_targets_resolve():
    # Matt 18:11 is a textual-variant omission carried as an omitted=1 row in the
    # bibles; it is within the KJV verse count, so a cross-ref TO it resolves.
    assert point_resolves(CanonicalPoint("MAT", 18, 11))


# --- validation --------------------------------------------------------------
def test_validation_flags_nonresolving_source_and_target():
    res = _parse(
        "Gen.1.1\tEccl.12.1\t26",          # both resolve
        "3John.1.15\tJohn.10.3\t2",        # source 3JN 1:15 does NOT resolve
        "Acts.11.30\t2John.1.1-3John.1.15\t2",  # target end 3JN 1:15 doesn't resolve
    )
    rep = validate_crossrefs(res)
    assert rep.n_refs == 3
    assert rep.n_nonresolving_source == 1
    assert rep.n_nonresolving_target == 1
    # Nothing is dropped — every ref is still present.
    assert rep.n_refs == len(res.refs)


def test_validation_counts_ranges_and_book_crossings():
    res = _parse(
        "Gen.1.1\tEccl.12.1\t26",
        "Num.3.1\tLev.27.34-Num.1.1\t5",   # book-crossing range
        "Rev.22.21\tEph.6.23-Eph.6.24\t2",  # in-book range
    )
    rep = validate_crossrefs(res)
    assert rep.n_ranges == 2
    assert rep.n_single == 1
    assert len(rep.cross_book_ranges) == 1


def test_validation_reports_vote_extremes_and_negative_count():
    res = _parse(
        "Gen.1.1\tNeh.9.6\t100",
        "Gen.1.1\tJohn.1.1\t-86",
        "Gen.1.2\tPs.104.30\t0",
    )
    rep = validate_crossrefs(res)
    assert rep.votes_min == -86
    assert rep.votes_max == 100
    assert rep.n_negative_votes == 1
    assert rep.n_zero_votes == 1


# --- against the built DB ----------------------------------------------------
@pytest.mark.skipif(not DB.exists(), reason="crossrefs.sqlite not built")
class TestCrossRefsAgainstDB:
    @pytest.fixture(scope="class")
    def conn(self):
        c = sqlite3.connect(DB)
        yield c
        c.close()

    def test_source_provenance_row_carries_license_and_attribution(self, conn):
        row = conn.execute(
            "SELECT license, attribution FROM source WHERE id='tsk'"
        ).fetchone()
        assert row is not None
        assert "CC-BY" in row[0]
        assert row[1] == CROSSREFS_ATTRIBUTION

    def test_votes_keep_negative_sign(self, conn):
        lo = conn.execute("SELECT MIN(votes) FROM crossref").fetchone()[0]
        assert lo < 0

    def test_every_row_marks_resolution(self, conn):
        # src_resolves / tgt_resolves are populated (0/1) for every row.
        n_bad = conn.execute(
            "SELECT COUNT(*) FROM crossref WHERE src_resolves NOT IN (0,1) "
            "OR tgt_resolves NOT IN (0,1)"
        ).fetchone()[0]
        assert n_bad == 0

    def test_nonresolving_rows_kept_not_dropped(self, conn):
        # The known 3JN 1:15 non-resolving source survives in the DB.
        row = conn.execute(
            "SELECT src_resolves FROM crossref WHERE src_book='3JN' "
            "AND src_chapter=1 AND src_verse=15"
        ).fetchone()
        assert row is not None
        assert row[0] == 0

    @pytest.mark.skipif(not BIBLES_DB.exists(), reason="bibles.sqlite not built")
    def test_resolving_targets_exist_in_bibles(self, conn):
        # Sample resolving single-verse targets and confirm a KJV verse row exists.
        bibles = sqlite3.connect(BIBLES_DB)
        try:
            rows = conn.execute(
                "SELECT tgt_book, tgt_chapter, tgt_verse FROM crossref "
                "WHERE is_range=0 AND tgt_resolves=1 "
                "ORDER BY votes DESC LIMIT 50"
            ).fetchall()
            for b, c, v in rows:
                hit = bibles.execute(
                    "SELECT 1 FROM verse WHERE translation='kjv' AND book=? "
                    "AND chapter=? AND verse_start<=? AND verse_end>=?",
                    (b, c, v, v),
                ).fetchone()
                assert hit is not None, f"resolving target {b} {c}:{v} not in KJV"
        finally:
            bibles.close()
