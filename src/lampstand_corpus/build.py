"""Build ``bibles.sqlite`` from parsed USFM books.

The schema is verse-addressable for display and carries full provenance per the
normalized-schema contract. Output is deterministic: rows are inserted in a fixed
canonical order, no timestamps or autoincrement-dependent ids leak in, and the
SQLite file is built with stable settings so the same snapshots produce a
bit-for-bit identical database.

Red-letter (words-of-Christ) data is stored as a JSON array of ``[start, end]``
character offsets on each verse row, plus a boolean flag for cheap filtering.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import books
from .schema import Provenance
from .usfm import ParsedBook

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE translation (
    id          TEXT PRIMARY KEY,   -- 'bsb','kjv','asv','web'
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    license     TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    retrieved   TEXT NOT NULL,      -- ISO date
    checksum    TEXT NOT NULL       -- SHA-256 of the source snapshot
);

CREATE TABLE book (
    id          TEXT PRIMARY KEY,   -- USFM id, 'GEN'..'REV'
    name        TEXT NOT NULL,
    ord         INTEGER NOT NULL    -- canonical order index
);

CREATE TABLE verse (
    translation  TEXT NOT NULL REFERENCES translation(id),
    book         TEXT NOT NULL REFERENCES book(id),
    chapter      INTEGER NOT NULL,
    verse_start  INTEGER NOT NULL,
    verse_end    INTEGER NOT NULL,  -- == verse_start unless a bridge
    text         TEXT NOT NULL,
    red_letter   INTEGER NOT NULL DEFAULT 0,  -- 0/1: any words-of-Christ
    wj_spans     TEXT,              -- JSON [[start,end],...] or NULL
    PRIMARY KEY (translation, book, chapter, verse_start)
);

CREATE INDEX idx_verse_ref ON verse (book, chapter, verse_start);
CREATE INDEX idx_verse_translation ON verse (translation);
"""


def _connect(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def write_bibles(
    parsed: dict[str, dict[str, ParsedBook]],
    provenance: dict[str, Provenance],
    out_path: Path,
) -> None:
    """Write all translations to ``out_path``.

    ``parsed`` is ``{translation_id: {book_id: ParsedBook}}``.
    ``provenance`` is ``{translation_id: Provenance}``.
    """
    conn = _connect(out_path)
    try:
        # Deterministic build identity, no wall-clock.
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("resource_type", "scripture"),
                ("canon", "protestant-66"),
            ],
        )

        names = {
            "bsb": "Berean Standard Bible",
            "kjv": "King James Version",
            "asv": "American Standard Version",
            "web": "World English Bible",
        }
        for tid in sorted(provenance):
            p = provenance[tid]
            conn.execute(
                "INSERT INTO translation VALUES (?,?,?,?,?,?,?)",
                (tid, names.get(tid, tid.upper()), p.version, p.license,
                 p.url, p.retrieved, p.checksum),
            )

        for book_id in books.ORDER:
            conn.execute(
                "INSERT INTO book VALUES (?,?,?)",
                (book_id, books.NAMES[book_id], books.ORDER_INDEX[book_id]),
            )

        # Verses in fixed order: translation, then canonical book, chapter, verse.
        for tid in sorted(parsed):
            for book_id in books.ORDER:
                pb = parsed[tid].get(book_id)
                if pb is None:
                    continue
                for v in sorted(pb.verses, key=lambda x: (x.chapter, x.verse_start)):
                    spans = [[s.start, s.end] for s in v.wj_spans]
                    conn.execute(
                        "INSERT INTO verse VALUES (?,?,?,?,?,?,?,?)",
                        (tid, book_id, v.chapter, v.verse_start, v.verse_end,
                         v.text, 1 if spans else 0,
                         json.dumps(spans, separators=(",", ":")) if spans else None),
                    )
        conn.commit()
    finally:
        conn.close()
