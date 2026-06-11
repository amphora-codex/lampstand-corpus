"""Build ``lexicons.sqlite`` from normalized lexicon entries (+ optional OSHB
Strong's-tagged Hebrew text).

Tables
------
* ``lexicon`` — one row per dictionary source (Strong's Greek/Hebrew, BDB) with
  full provenance (version, edition license, underlying-text license, url,
  checksum, retrieval date).
* ``entry`` — one row per dictionary entry, keyed ``(lexicon, strongs)``. Greek
  and Hebrew are distinguished by ``language`` and by the ``G``/``H`` prefix on
  ``strongs``. BDB entries that the LexicalIndex did not link to a Strong's number
  are stored keyed by their BDB id (``raw_key``) with an empty ``strongs`` and
  ``strongs_linked=0`` so the app/validator can see them without a fabricated key.
* ``tagged_word`` — (P4b, only when OSHB is ingested) one row per Strong's-tagged
  Hebrew word, anchored to ``(book, chapter, verse, position)`` on the canonical
  spine, with its Strong's H-numbers (JSON array), surface form, morphology, and
  the raw OSHB lemma.
* ``tagged_source`` — provenance for each ingested tagged-text source.

Output is deterministic: sources and rows are written in fixed sorted order, no
wall-clock timestamps leak in, so identical snapshots yield a bit-identical file.
``lexicons.sqlite`` is gitignored (a built artifact), never committed.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .lexicons import TAGGED_TEXT_SOURCES, ParsedLexicon, ParsedTaggedText

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE lexicon (
    id            TEXT PRIMARY KEY,   -- strongs-greek / strongs-hebrew / bdb
    name          TEXT NOT NULL,
    language      TEXT NOT NULL,      -- greek / hebrew
    lexicon       TEXT NOT NULL,      -- strongs / bdb
    version       TEXT NOT NULL,
    license       TEXT NOT NULL,      -- OpenScriptures edition license
    text_license  TEXT NOT NULL,      -- underlying dictionary text license (PD)
    source_url    TEXT NOT NULL,
    retrieved     TEXT NOT NULL,
    checksum      TEXT NOT NULL
);

CREATE TABLE entry (
    lexicon         TEXT NOT NULL REFERENCES lexicon(id),
    strongs         TEXT NOT NULL,    -- 'G####' / 'H####' (or '' for unlinked BDB)
    language        TEXT NOT NULL,
    raw_key         TEXT,             -- BDB entry id where it differs from strongs
    strongs_linked  INTEGER NOT NULL, -- 1 if keyed by a Strong's number, else 0
    lemma           TEXT,
    translit        TEXT,
    pronunciation   TEXT,
    definition      TEXT,
    derivation      TEXT,
    kjv_def         TEXT,
    PRIMARY KEY (lexicon, strongs, raw_key)
);

CREATE INDEX idx_entry_strongs ON entry (strongs);
CREATE INDEX idx_entry_language ON entry (language);

CREATE TABLE tagged_source (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    language      TEXT NOT NULL,
    version       TEXT NOT NULL,
    license       TEXT NOT NULL,
    attribution   TEXT NOT NULL,
    source_url    TEXT NOT NULL,
    retrieved     TEXT NOT NULL,
    checksum      TEXT NOT NULL
);

CREATE TABLE tagged_word (
    source     TEXT NOT NULL REFERENCES tagged_source(id),
    book       TEXT NOT NULL,
    chapter    INTEGER NOT NULL,
    verse      INTEGER NOT NULL,
    position   INTEGER NOT NULL,   -- 1-based word index within the verse
    surface    TEXT NOT NULL,
    strongs    TEXT NOT NULL,      -- JSON array of 'H####' (may be empty)
    morph      TEXT,
    lemma_raw  TEXT NOT NULL,
    PRIMARY KEY (source, book, chapter, verse, position)
);

CREATE INDEX idx_tagged_ref ON tagged_word (book, chapter, verse);
"""


def _connect(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _entry_sort_key(strongs: str, raw_key: str | None) -> tuple[int, str]:
    """Sort entries by numeric Strong's value, then raw_key — deterministic."""
    if strongs and len(strongs) > 1 and strongs[1:].isdigit():
        return (int(strongs[1:]), raw_key or "")
    return (1 << 30, raw_key or strongs)


def write_lexicons(
    lexicons: dict[str, ParsedLexicon],
    out_path: Path,
    *,
    tagged: dict[str, ParsedTaggedText] | None = None,
) -> None:
    """Write all parsed lexicons (+ optional tagged text) to ``out_path``."""
    conn = _connect(out_path)
    tagged = tagged or {}
    try:
        meta = [("schema_version", "1"), ("resource_type", "lexicon")]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta)

        for lid in sorted(lexicons):
            pl = lexicons[lid]
            prov = pl.provenance
            conn.execute(
                "INSERT INTO lexicon VALUES (?,?,?,?,?,?,?,?,?,?)",
                (lid, _lexicon_name(pl), pl.language, pl.lexicon,
                 prov.version, prov.license, _text_license(prov),
                 prov.url, prov.retrieved, prov.checksum),
            )
            rows = sorted(
                pl.entries, key=lambda e: _entry_sort_key(e.strongs, e.raw_key)
            )
            for e in rows:
                conn.execute(
                    "INSERT INTO entry VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (lid, e.strongs, e.language, e.raw_key,
                     1 if e.strongs else 0,
                     e.lemma, e.translit, e.pronunciation,
                     e.definition, e.derivation, e.kjv_def),
                )

        for sid in sorted(tagged):
            pt = tagged[sid]
            src = TAGGED_TEXT_SOURCES[sid]
            prov = pt.provenance
            conn.execute(
                "INSERT INTO tagged_source VALUES (?,?,?,?,?,?,?,?,?)",
                (sid, src.name, pt.language, prov.version, prov.license,
                 src.attribution, prov.url, prov.retrieved, prov.checksum),
            )
            words = sorted(
                pt.words,
                key=lambda w: (w.ref.book, w.ref.chapter, w.ref.verse_start,
                               w.position),
            )
            for w in words:
                conn.execute(
                    "INSERT INTO tagged_word VALUES (?,?,?,?,?,?,?,?,?)",
                    (sid, w.ref.book, w.ref.chapter, w.ref.verse_start,
                     w.position, w.surface,
                     json.dumps(w.strongs, separators=(",", ":")),
                     w.morph, w.lemma_raw),
                )
        conn.commit()
    finally:
        conn.close()


def _lexicon_name(pl: ParsedLexicon) -> str:
    from .lexicons import LEXICON_SOURCES

    src = LEXICON_SOURCES.get(pl.id)
    return src.name if src else pl.id


def _text_license(prov) -> str:
    # The Provenance model carries one license field; the underlying-text license
    # is recorded on the source. We thread it through meta to keep Provenance
    # unchanged — pull from LEXICON_SOURCES by matching source id.
    from .lexicons import LEXICON_SOURCES

    for src in LEXICON_SOURCES.values():
        if src.url == prov.url:
            return src.text_license
    return prov.license
