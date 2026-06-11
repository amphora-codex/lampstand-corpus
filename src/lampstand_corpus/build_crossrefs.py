"""Build ``crossrefs.sqlite`` from normalized Treasury-of-Scripture-Knowledge
cross-references (OpenBible.info, CC-BY).

Tables
------
* ``meta`` — schema/version + the CC-BY license and required attribution string.
* ``source`` — one provenance row for the dataset (url, license, attribution,
  retrieval date, checksum, original header line).
* ``crossref`` — one row per (source verse -> target) edge. The source is a single
  canonical verse ``(src_book, src_chapter, src_verse)``. The target is a verse OR
  a range, stored as its full start AND end coordinates
  (``tgt_book/tgt_chapter/tgt_verse`` .. ``tgt_end_book/tgt_end_chapter/tgt_end_verse``)
  so chapter- and book-crossing ranges (e.g. Gen.11.32-Gen.12.1, Lev.27.34-Num.1.1)
  round-trip exactly. ``votes`` keeps the signed relevance weight; ``rank`` is the
  1-based position of the target within its source verse. ``src_resolves`` /
  ``tgt_resolves`` flag whether each endpoint lands on a real verse of the
  canonical (KJV) spine — non-resolving rows are KEPT (flagged), never dropped.

Output is deterministic: rows are written in a fixed canonical order (source book
order, chapter, verse, then rank), no wall-clock timestamps leak in, so identical
snapshots yield a bit-identical file. ``crossrefs.sqlite`` is gitignored.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import books
from .crossrefs import ParsedCrossRefs, point_resolves

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE source (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    license       TEXT NOT NULL,
    attribution   TEXT NOT NULL,
    source_url    TEXT NOT NULL,
    retrieved     TEXT NOT NULL,
    checksum      TEXT NOT NULL,
    header        TEXT NOT NULL   -- original data-file header line (provenance)
);

CREATE TABLE crossref (
    src_book        TEXT NOT NULL,
    src_chapter     INTEGER NOT NULL,
    src_verse       INTEGER NOT NULL,
    tgt_book        TEXT NOT NULL,
    tgt_chapter     INTEGER NOT NULL,
    tgt_verse       INTEGER NOT NULL,
    tgt_end_book    TEXT NOT NULL,
    tgt_end_chapter INTEGER NOT NULL,
    tgt_end_verse   INTEGER NOT NULL,
    is_range        INTEGER NOT NULL,
    votes           INTEGER NOT NULL,   -- signed relevance weight (may be negative)
    rank            INTEGER NOT NULL,   -- 1-based rank within the source verse
    src_resolves    INTEGER NOT NULL,   -- 1 if source verse exists on KJV spine
    tgt_resolves    INTEGER NOT NULL    -- 1 if BOTH target endpoints exist
);

CREATE INDEX idx_crossref_src ON crossref (src_book, src_chapter, src_verse);
CREATE INDEX idx_crossref_tgt ON crossref (tgt_book, tgt_chapter, tgt_verse);
"""


def _connect(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _sort_key(row: tuple) -> tuple:
    """Deterministic ordering: source canonical order, then target, then rank."""
    (sb, sc, sv, tb, tc, tv, *_rest, rank) = row[:6] + (row[-1],)
    return (
        books.ORDER_INDEX.get(sb, 1 << 30), sc, sv,
        books.ORDER_INDEX.get(tb, 1 << 30), tc, tv, rank,
    )


def write_crossrefs(parsed: ParsedCrossRefs, out_path: Path) -> None:
    """Write all normalized cross-references to ``out_path`` deterministically."""
    conn = _connect(out_path)
    try:
        prov = parsed.provenance
        meta = [
            ("schema_version", "1"),
            ("resource_type", "crossref"),
            ("license", prov.license if prov else ""),
        ]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta)

        if prov is not None:
            from .crossrefs import CROSSREFS_ATTRIBUTION, CROSSREFS_NAME
            conn.execute(
                "INSERT INTO source VALUES (?,?,?,?,?,?,?,?)",
                ("tsk", CROSSREFS_NAME, prov.license, CROSSREFS_ATTRIBUTION,
                 prov.url, prov.retrieved, prov.checksum, parsed.header),
            )

        rows: list[tuple] = []
        for cr in parsed.refs:
            s, ts, te = cr.source, cr.target_start, cr.target_end
            src_ok = point_resolves(s)
            tgt_ok = point_resolves(ts) and point_resolves(te)
            rows.append((
                s.book, s.chapter, s.verse,
                ts.book, ts.chapter, ts.verse,
                te.book, te.chapter, te.verse,
                1 if cr.is_range else 0,
                cr.votes, cr.rank,
                1 if src_ok else 0, 1 if tgt_ok else 0,
            ))

        rows.sort(key=_sort_key)
        conn.executemany(
            "INSERT INTO crossref VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        conn.commit()
    finally:
        conn.close()
