"""TSK crossref-pack tests — verse keys, blobs, expansion aggregation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus import crossref_pack as cp
from lampstand_corpus.pack_codec import zigzag_decode, zigzag_encode


# --- zigzag ------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, -1, 2, -2, 63, -64, 1278, -86, 2**31])
def test_zigzag_round_trip(n):
    assert zigzag_decode(zigzag_encode(n)) == n


def test_zigzag_small_magnitudes_stay_small():
    # -86..1278 (the TSK vote range) must fit in <= 2 varint bytes.
    assert zigzag_encode(-86) == 171
    assert zigzag_encode(1278) == 2556  # < 2^14 -> 2-byte uvarint


# --- verse keys ---------------------------------------------------------------------
def test_verse_key_round_trip_and_monotone():
    for ref in (("GEN", 1, 1), ("PSA", 119, 176), ("MAL", 4, 6), ("REV", 22, 21)):
        assert cp.verse_key_parts(cp.verse_key(*ref)) == ref
    # Canonical order is preserved by the arithmetic key.
    assert cp.verse_key("GEN", 50, 26) < cp.verse_key("EXO", 1, 1)
    assert cp.verse_key("MAL", 4, 6) < cp.verse_key("MAT", 1, 1)


def test_verse_key_rejects_out_of_range():
    with pytest.raises(ValueError):
        cp.verse_key("GEN", 0, 5)
    with pytest.raises(ValueError):
        cp.verse_key("GEN", 1, 1000)
    with pytest.raises(KeyError):
        cp.verse_key("TOB", 1, 1)  # off-canon book


# --- blobs ---------------------------------------------------------------------------
def test_targets_round_trip_with_ranges_and_negative_votes():
    targets = [
        (cp.verse_key("JHN", 1, 1), cp.verse_key("JHN", 1, 5), 120),
        (cp.verse_key("PSA", 33, 6), cp.verse_key("PSA", 33, 6), -86),
        # Book-crossing range (Lev 27:34 - Num 1:1, a real TSK shape).
        (cp.verse_key("LEV", 27, 34), cp.verse_key("NUM", 1, 1), 3),
    ]
    assert cp.decode_targets(cp.encode_targets(targets)) == targets


def test_targets_reject_reversed_range():
    with pytest.raises(ValueError):
        cp.encode_targets([(cp.verse_key("JHN", 1, 5), cp.verse_key("JHN", 1, 1), 1)])


def test_neighbors_round_trip_and_validation():
    nbrs = [(42, 120), (7, 3), (175_442, 1)]
    assert cp.decode_neighbors(cp.encode_neighbors(nbrs)) == nbrs
    with pytest.raises(ValueError):
        cp.encode_neighbors([(1, 0)])


# --- edge rows -------------------------------------------------------------------------
def _mk_crossrefs(path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE crossref(src_book TEXT, src_chapter INT, src_verse INT,"
        " tgt_book TEXT, tgt_chapter INT, tgt_verse INT, tgt_end_book TEXT,"
        " tgt_end_chapter INT, tgt_end_verse INT, is_range INT, votes INT,"
        " rank INT, src_resolves INT, tgt_resolves INT)")
    conn.execute(
        "CREATE TABLE source(id TEXT PRIMARY KEY, license TEXT, attribution TEXT)")
    conn.execute("INSERT INTO source VALUES ('tsk','CC-BY 4.0','openbible')")
    conn.executemany(
        "INSERT INTO crossref VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_build_edge_rows_orders_and_filters(tmp_path):
    db = tmp_path / "crossrefs.sqlite"
    _mk_crossrefs(db, [
        # Deliberately inserted out of order; rank defines the target order.
        ("GEN", 1, 2, "JHN", 1, 3, "JHN", 1, 3, 0, 5, 1, 1, 1),
        ("GEN", 1, 1, "PSA", 33, 6, "PSA", 33, 6, 0, 2, 2, 1, 1),
        ("GEN", 1, 1, "JHN", 1, 1, "JHN", 1, 1, 0, 90, 1, 1, 1),
        ("GEN", 1, 1, "REV", 99, 1, "REV", 99, 1, 0, 9, 3, 1, 0),  # non-resolving
    ])
    rows, stats = cp.build_edge_rows(db)
    assert stats == {"n_sources": 2, "n_edges": 3}
    # Sources in canonical spine order.
    assert [r[0] for r in rows] == [cp.verse_key("GEN", 1, 1),
                                    cp.verse_key("GEN", 1, 2)]
    # Targets in rank (file/vote) order, non-resolving edge dropped.
    targets = cp.decode_targets(rows[0][2])
    assert [t[2] for t in targets] == [90, 2]
    # Deterministic across rebuilds.
    rows2, _ = cp.build_edge_rows(db)
    assert rows == rows2


# --- expansion --------------------------------------------------------------------------
def _refs(*rows) -> list[cp.ScriptureChunkRef]:
    return [cp.ScriptureChunkRef(*r) for r in rows]


def test_expansion_aggregates_votes_and_excludes_self(tmp_path):
    db = tmp_path / "crossrefs.sqlite"
    _mk_crossrefs(db, [
        # Two verses of the SAME source pericope both point at JHN 1:1-10:
        # votes must SUM (10 + 15 = 25).
        ("GEN", 1, 1, "JHN", 1, 1, "JHN", 1, 1, 0, 10, 1, 1, 1),
        ("GEN", 1, 2, "JHN", 1, 2, "JHN", 1, 2, 0, 15, 1, 1, 1),
        # Self-reference within the pericope: excluded for every translation.
        ("GEN", 1, 3, "GEN", 1, 5, "GEN", 1, 5, 0, 99, 1, 1, 1),
        # Net-negative target: dropped.
        ("GEN", 1, 4, "PSA", 33, 6, "PSA", 33, 6, 0, -7, 1, 1, 1),
    ])
    chunks = _refs(
        ("b_gen", "bsb", "GEN", 1, 1, 10),
        ("k_gen", "kjv", "GEN", 1, 1, 10),
        ("b_jhn", "bsb", "JHN", 1, 1, 10),
        ("b_psa", "bsb", "PSA", 33, 1, 10),
    )
    exp = cp.build_expansion(chunks, db)
    # Both translations of the source pericope get the SAME bsb neighbor.
    assert exp["b_gen"] == [("b_jhn", 25)]
    assert exp["k_gen"] == [("b_jhn", 25)]
    # Chunks with no positive-weight neighbors are absent.
    assert "b_jhn" not in exp and "b_psa" not in exp
    # Deterministic.
    assert exp == cp.build_expansion(chunks, db)


def test_expansion_top_n_cut_is_by_weight_then_id(tmp_path):
    db = tmp_path / "crossrefs.sqlite"
    rows = []
    # GEN 1:1 -> ten different PSA chapters with descending votes.
    for i in range(10):
        rows.append(("GEN", 1, 1, "PSA", i + 1, 1, "PSA", i + 1, 1, 0,
                     100 - i, i + 1, 1, 1))
    _mk_crossrefs(db, rows)
    chunks = _refs(("b_gen", "bsb", "GEN", 1, 1, 10)) + _refs(
        *((f"b_psa{i+1:02d}", "bsb", "PSA", i + 1, 1, 6) for i in range(10)))
    exp = cp.build_expansion(chunks, db, top_n=3)
    assert exp["b_gen"] == [("b_psa01", 100), ("b_psa02", 99), ("b_psa03", 98)]
