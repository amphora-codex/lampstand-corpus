"""Unit tests for the human-citation -> VerseRef parser (scripref)."""

from __future__ import annotations

from lampstand_corpus.scripref import parse_proof_block


def _triples(block: str) -> list[tuple[str, int, int, int | None]]:
    res = parse_proof_block(block)
    return [(r.book, r.chapter, r.verse_start, r.verse_end) for r in res.refs]


def test_basic_semicolon_block():
    refs = _triples("Eph. 1:11; Rom. 11:33")
    assert refs == [("EPH", 1, 11, None), ("ROM", 11, 33, None)]


def test_full_book_names_and_ranges():
    refs = _triples("2 Timothy 3:15-17; Isaiah 8:20")
    assert refs == [("2TI", 3, 15, 17), ("ISA", 8, 20, None)]


def test_verse_list_shares_chapter():
    refs = _triples("Rom. 9:15, 18")
    assert refs == [("ROM", 9, 15, None), ("ROM", 9, 18, None)]


def test_connector_with_switches_book():
    refs = _triples("Gen. 2:7 with Eccles. 12:7")
    assert ("GEN", 2, 7, None) in refs
    assert ("ECC", 12, 7, None) in refs


def test_one_chapter_book_verse_only():
    # "Jude 4" names a verse, not a chapter -> Jude 1:4.
    refs = _triples("Jude 4; 2 John 10, 11")
    assert ("JUD", 1, 4, None) in refs
    assert ("2JN", 1, 10, None) in refs
    assert ("2JN", 1, 11, None) in refs


def test_chapter_only_is_unparsed_not_guessed():
    # "Lev. 18" has no verse — must NOT be invented as 18:1.
    res = parse_proof_block("Lev. 18")
    assert res.refs == []
    assert res.unparsed  # surfaced for the human


def test_unknown_book_is_unparsed():
    res = parse_proof_block("Tobit 1:1")
    assert res.refs == []
    assert "Tobit 1:1" in res.unparsed


def test_repeated_book_after_comma():
    refs = _triples("Job 14:4, Job 15:14")
    assert ("JOB", 14, 4, None) in refs
    assert ("JOB", 15, 14, None) in refs
