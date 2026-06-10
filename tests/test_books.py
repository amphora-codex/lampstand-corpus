"""Canonical book-table sanity tests."""

from __future__ import annotations

from lampstand_corpus import books


def test_canon_is_66_books():
    assert len(books.ORDER) == 66
    assert len(books.CANON) == 66
    assert len(set(books.ORDER)) == 66


def test_every_book_has_name_and_chapter_count():
    for b in books.ORDER:
        assert b in books.NAMES
        assert b in books.CHAPTER_COUNTS
        assert books.CHAPTER_COUNTS[b] >= 1


def test_verse_counts_match_chapter_counts():
    # Where a verse-count array exists, its length must equal the chapter count.
    for b, counts in books.VERSE_COUNTS.items():
        assert len(counts) == books.CHAPTER_COUNTS[b], b


def test_order_index_is_dense_and_sorted():
    assert books.ORDER_INDEX["GEN"] == 0
    assert books.ORDER_INDEX["REV"] == 65
    assert books.ORDER_INDEX["MAT"] == 39  # first NT book


def test_known_chapter_counts():
    assert books.CHAPTER_COUNTS["PSA"] == 150
    assert books.CHAPTER_COUNTS["OBA"] == 1
    assert books.CHAPTER_COUNTS["JHN"] == 21
