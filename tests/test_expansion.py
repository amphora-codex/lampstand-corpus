"""Rank-7 expansion-mining tests — tiny parallel-verse fixtures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from lampstand_corpus import expansion as ex


def _mk_bibles(path: Path, verses: dict[str, dict[tuple, str]]) -> None:
    """verses: {translation: {(book, ch, vs): text}}."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE verse (translation TEXT, book TEXT, chapter INT,"
        " verse_start INT, verse_end INT, text TEXT)")
    for tid, by_ref in verses.items():
        for (b, ch, vs), text in by_ref.items():
            conn.execute("INSERT INTO verse VALUES (?,?,?,?,?,?)",
                         (tid, b, ch, vs, vs, text))
    conn.commit()
    conn.close()


def _parallel_fixture(tmp_path: Path) -> Path:
    """KJV uses 'believeth' where BSB uses 'believes' across enough verses."""
    kjv, bsb = {}, {}
    for v in range(1, 7):
        kjv[("JHN", 3, v)] = f"whosoever believeth in him verse {v}"
        bsb[("JHN", 3, v)] = f"everyone who believes in him verse {v}"
    # A near-stopword partner trap: 'the' co-occurs everywhere.
    kjv[("JHN", 3, 7)] = "the wind bloweth where it listeth"
    bsb[("JHN", 3, 7)] = "the wind blows where it wishes"
    # Filler verses give partner tokens a realistic base rate (the lift filter
    # compares P(partner | archaic verses) against P(partner) overall).
    for v in range(1, 41):
        kjv[("GEN", 1, v)] = f"the earth was formless and void number {v}"
        bsb[("GEN", 1, v)] = f"the earth was formless and void number {v}"
    db = tmp_path / "bibles.sqlite"
    _mk_bibles(db, {"kjv": kjv, "bsb": bsb})
    return db


# --- archaic mining ------------------------------------------------------------
def test_mine_archaic_finds_inflection_pair(tmp_path):
    pairs = ex.mine_archaic_pairs(_parallel_fixture(tmp_path))
    by = {p["archaic"]: p for p in pairs}
    assert by["believeth"]["modern"] == "believes"
    assert by["believeth"]["mechanical"] is True
    assert by["believeth"]["score"] == 1.0
    # 'the' must never be mined as anyone's partner (lift filter).
    assert all(p["modern"] != "the" for p in pairs)


def test_mine_archaic_is_deterministic(tmp_path):
    db = _parallel_fixture(tmp_path)
    assert ex.mine_archaic_pairs(db) == ex.mine_archaic_pairs(db)


# --- suffix classes ---------------------------------------------------------------
def test_suffix_classes_group_inflections():
    vocab = {"justify": 40, "justified": 30, "justification": 25,
             "believe": 50, "believeth": 9, "believes": 22,
             "rare": 1, "rarest": 1}
    classes = ex.suffix_classes(vocab)
    assert set(classes["justify"]) == {"justify", "justified", "justification"}
    assert set(classes["believe"]) == {"believe", "believeth", "believes"}
    # df floor: 'rare/rarest' never form a class.
    assert "rare" not in classes


def test_suffix_classes_never_invent_words():
    # 'walking' strips to 'walk' only if 'walk' is IN the vocabulary.
    assert ex.suffix_classes({"walking": 10, "walked": 10}) == {}
    got = ex.suffix_classes({"walking": 10, "walk": 10})
    assert set(got["walk"]) == {"walk", "walking"}


# --- shipped rows + DRAFT file -------------------------------------------------------
def test_build_expansion_rows_bidirectional_and_draft(tmp_path):
    db = _parallel_fixture(tmp_path)
    vocab = {"justify": 40, "justified": 30, "believeth": 9, "believe": 22}
    rows, stats = ex.build_expansion_rows(db, vocab, tmp_path)
    as_set = {(r[0], r[1], r[2]) for r in rows}
    # Mined archaic pair ships bidirectionally.
    assert ("believeth", "believes", "archaic") in as_set
    assert ("believes", "believeth", "archaic") in as_set
    # Suffix class members expand to each other.
    assert ("justify", "justified", "suffix") in as_set
    assert ("justified", "justify", "suffix") in as_set
    # DRAFT synonym file written, and NOT shipped while DRAFT.
    syn = tmp_path / "data" / "eval" / "theological_synonyms_v1.json"
    assert syn.exists()
    payload = json.loads(syn.read_text(encoding="utf-8"))
    assert "DRAFT" in payload["status"]
    assert all(kind != "synonym" for _t, _e, kind in as_set)
    # Deterministic.
    rows2, _ = ex.build_expansion_rows(db, vocab, tmp_path)
    assert rows == rows2


def test_approved_synonyms_are_folded_in(tmp_path):
    db = _parallel_fixture(tmp_path)
    syn = tmp_path / "data" / "eval" / "theological_synonyms_v1.json"
    syn.parent.mkdir(parents=True)
    syn.write_text(json.dumps({
        "status": "APPROVED by advisor",
        "pairs": [{"archaic": "charity", "modern": "love", "score": 0.9}],
    }), encoding="utf-8")
    rows, _ = ex.build_expansion_rows(db, {}, tmp_path)
    as_set = {(r[0], r[1], r[2]) for r in rows}
    assert ("charity", "love", "synonym") in as_set
    assert ("love", "charity", "synonym") in as_set


def test_architect_only_approval_keeps_synonyms_unwired(tmp_path):
    """Architect clearance (status=approved, approved_by=architect) does NOT
    fold synonyms into the shipped table — the wiring gate is the advisor's."""
    db = _parallel_fixture(tmp_path)
    syn = tmp_path / "data" / "eval" / "theological_synonyms_v1.json"
    syn.parent.mkdir(parents=True)
    syn.write_text(json.dumps({
        "status": "approved",
        "approved_by": "architect",
        "pairs": [{"archaic": "charity", "modern": "love", "score": 0.9}],
    }), encoding="utf-8")
    assert ex.load_approved_synonyms(syn) == []
    rows, _ = ex.build_expansion_rows(db, {}, tmp_path)
    as_set = {(r[0], r[1], r[2]) for r in rows}
    assert all(kind != "synonym" for _t, _e, kind in as_set)


def test_advisor_approval_by_field_wires_synonyms(tmp_path):
    """An `approved_by` naming the advisor DOES wire the pairs."""
    db = _parallel_fixture(tmp_path)
    syn = tmp_path / "data" / "eval" / "theological_synonyms_v1.json"
    syn.parent.mkdir(parents=True)
    syn.write_text(json.dumps({
        "status": "approved",
        "approved_by": "theological-advisor",
        "pairs": [{"archaic": "charity", "modern": "love", "score": 0.9}],
    }), encoding="utf-8")
    assert len(ex.load_approved_synonyms(syn)) == 1
    rows, _ = ex.build_expansion_rows(db, {}, tmp_path)
    as_set = {(r[0], r[1], r[2]) for r in rows}
    assert ("charity", "love", "synonym") in as_set


def test_load_expansion_map_shape():
    rows = [("a", "b", "archaic", 0.9), ("a", "c", "suffix", 1.0),
            ("b", "a", "archaic", 0.9)]
    m = ex.load_expansion_map(rows)
    assert m == {"a": ["b", "c"], "b": ["a"]}


def test_duplicate_pairs_across_sources_dedupe_with_precedence(tmp_path):
    """believeth↔believes arrives from BOTH the archaic miner and the suffix
    class (believe/believes/believeth) — the table PK is (term, expansion), so
    exactly one row must survive, by documented precedence (weight desc, then
    synonym > archaic > suffix), independent of arrival order."""
    db = _parallel_fixture(tmp_path)
    vocab = {"believe": 50, "believes": 22, "believeth": 9}
    rows, _ = ex.build_expansion_rows(db, vocab, tmp_path)
    keys = [(r[0], r[1]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate (term, expansion) rows"
    by_key = {(r[0], r[1]): (r[2], r[3]) for r in rows}
    # Fixture archaic score is 1.0 == suffix weight 1.0 -> tie -> archaic wins
    # (mined outranks rule-derived at equal weight).
    assert by_key[("believeth", "believes")] == ("archaic", 1.0)
    assert by_key[("believes", "believeth")] == ("archaic", 1.0)
    # Pure suffix pairs (no archaic twin) stay suffix.
    assert by_key[("believe", "believes")][0] == "suffix"


# --- directional gloss (GAP 2: tap-to-gloss) --------------------------------------
def test_gloss_rows_are_one_way_archaic_to_modern(tmp_path):
    """The gloss table is DIRECTIONAL: the archaic surface form -> modern gloss,
    and ONLY that direction (unlike the symmetric expansion table)."""
    db = _parallel_fixture(tmp_path)
    rows, stats = ex.build_gloss_rows(db, tmp_path)
    by = {r[0]: (r[1], r[2]) for r in rows}
    # Mechanical mined pair: archaic -> modern present, reverse ABSENT.
    assert by["believeth"] == ("believes", "archaic")
    assert "believes" not in by
    # Term is the primary key: at most one gloss per surface form.
    assert len({r[0] for r in rows}) == len(rows)
    assert stats["direction"] == "archaic-to-modern"
    # Deterministic.
    assert ex.build_gloss_rows(db, tmp_path) == (rows, stats)


def test_gloss_folds_in_advisor_approved_synonyms_one_way(tmp_path):
    db = _parallel_fixture(tmp_path)
    syn = tmp_path / "data" / "eval" / "theological_synonyms_v1.json"
    syn.parent.mkdir(parents=True)
    syn.write_text(json.dumps({
        "status": "APPROVED by advisor",
        "pairs": [{"archaic": "sepulchre", "modern": "tomb", "score": 0.9}],
    }), encoding="utf-8")
    rows, stats = ex.build_gloss_rows(db, tmp_path)
    by = {r[0]: (r[1], r[2]) for r in rows}
    assert by["sepulchre"] == ("tomb", "synonym")
    # One-way only — the modern term does NOT gloss back to the archaic.
    assert "tomb" not in by
    assert stats["n_synonym_approved"] == 1


def test_gloss_excludes_unapproved_synonyms(tmp_path):
    """DRAFT (advisor-unsigned) synonyms must NOT appear in the gloss table."""
    db = _parallel_fixture(tmp_path)
    syn = tmp_path / "data" / "eval" / "theological_synonyms_v1.json"
    syn.parent.mkdir(parents=True)
    syn.write_text(json.dumps({
        "status": "DRAFT",
        "pairs": [{"archaic": "sepulchre", "modern": "tomb", "score": 0.9}],
    }), encoding="utf-8")
    rows, _ = ex.build_gloss_rows(db, tmp_path)
    assert all(r[2] != "synonym" for r in rows)
    assert "sepulchre" not in {r[0] for r in rows}
