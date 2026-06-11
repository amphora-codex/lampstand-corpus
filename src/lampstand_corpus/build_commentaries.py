"""Build ``commentaries.sqlite`` from normalized commentary chunks.

One row per paragraph-level chunk, each anchored to the ``VerseRef`` (single
verse or verse range / pericope) of the governing ``scripCom``. Full provenance
per chunk so the app can cite ``[MH Rom 9]`` / ``[JFB Rom 9]`` / ``[Calvin Comm.
Rom 9.23]`` and resolve back to the exact CCEL source + snapshot checksum.

Output is deterministic: commentators and chunks are written in a fixed canonical
order (book, chapter, verse, paragraph), no wall-clock timestamps leak in, so the
same snapshots yield a bit-identical file. ``commentaries.sqlite`` is gitignored
(a built artifact), never committed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import books
from .commentaries import ParsedCommentary, all_commentary_sources

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE commentator (
    id          TEXT PRIMARY KEY,   -- 'henry','jfb','calvin'
    name        TEXT NOT NULL,
    shortcode   TEXT NOT NULL,      -- citation prefix: MH / JFB / Calvin Comm.
    author      TEXT NOT NULL,
    work        TEXT NOT NULL,
    version     TEXT NOT NULL,
    license     TEXT NOT NULL,
    retrieved   TEXT NOT NULL,
    source_urls TEXT NOT NULL       -- space-joined CCEL volume URLs
);

CREATE TABLE comment (
    commentator   TEXT NOT NULL REFERENCES commentator(id),
    key           TEXT NOT NULL,    -- 'BOOK.chap.vstart[-vend]#p<n>'
    ord           INTEGER NOT NULL, -- stable global display order
    book          TEXT NOT NULL REFERENCES book(id),
    chapter       INTEGER NOT NULL,
    verse_start   INTEGER NOT NULL, -- 0 for a whole-chapter (intro) note
    verse_end     INTEGER NOT NULL, -- == verse_start unless a range / pericope
    chapter_level INTEGER NOT NULL DEFAULT 0,  -- 1 = chapter-introduction note
    para_index    INTEGER NOT NULL, -- paragraph ordinal within the verse anchor
    passage       TEXT,             -- CCEL's human-readable passage label
    component     TEXT,             -- Spurgeon Treasury section (exposition/notes/
                                    -- hints/works/title); NULL for CCEL commentators
    volume        TEXT NOT NULL,    -- source volume stem (CCEL stem or Treasury vol)
    text          TEXT NOT NULL,
    checksum      TEXT NOT NULL,    -- SHA-256 of the volume snapshot this came from
    PRIMARY KEY (commentator, key)
);

CREATE TABLE book (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    ord  INTEGER NOT NULL
);

CREATE INDEX idx_comment_ref ON comment (book, chapter, verse_start);
CREATE INDEX idx_comment_commentator ON comment (commentator);
"""


def _connect(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def write_commentaries(
    parsed: dict[str, ParsedCommentary], out_path: Path
) -> None:
    """Write all parsed commentaries to ``out_path`` (deterministic order)."""
    sources = all_commentary_sources()
    conn = _connect(out_path)
    try:
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [("schema_version", "1"), ("resource_type", "commentary"),
             ("canon", "protestant-66")],
        )

        for book_id in books.ORDER:
            conn.execute(
                "INSERT INTO book VALUES (?,?,?)",
                (book_id, books.NAMES[book_id], books.ORDER_INDEX[book_id]),
            )

        for cid in sorted(parsed):
            pc = parsed[cid]
            if not pc.chunks:
                continue
            src = sources[cid]
            # All chunks of a commentator share name/license; provenance varies by
            # volume (different checksum). Pull stable fields from the first chunk.
            first = pc.chunks[0]
            prov = first.provenance
            urls = " ".join(src.url(v) for v in src.volumes)
            conn.execute(
                "INSERT INTO commentator VALUES (?,?,?,?,?,?,?,?,?)",
                (cid, src.name, src.shortcode, src.author, src.work,
                 src.version, src.license, prov.retrieved, urls),
            )
            for ord_i, ch in enumerate(pc.chunks):
                r = ch.ref
                m = ch.meta
                conn.execute(
                    "INSERT INTO comment VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cid, ch.key, ord_i,
                        r.book, r.chapter, r.verse_start,
                        r.verse_end if r.verse_end else r.verse_start,
                        1 if m.get("chapter_level") else 0,
                        m.get("para_index", 0),
                        m.get("passage"),
                        m.get("component"),
                        m.get("volume", ""),
                        ch.text,
                        ch.provenance.checksum,
                    ),
                )
        conn.commit()
    finally:
        conn.close()
