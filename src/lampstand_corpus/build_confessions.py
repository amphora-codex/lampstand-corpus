"""Build ``confessions.sqlite`` from normalized confession chunks.

One row per section / Q&A, keyed by ``document`` + ``key`` (chapter.section for
the WCF, question number for the catechisms). Scripture proof-texts are stored as
a JSON array of ``{book,chapter,verse_start[,verse_end]}`` objects so the app can
render and resolve them against the verse spine. Full provenance per chunk.

Output is deterministic: documents and chunks are written in a fixed order, no
wall-clock timestamps leak in, so the same snapshots yield a bit-identical file.
``confessions.sqlite`` is gitignored (a built artifact), never committed.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .confessions import CONFESSION_SOURCES, ParsedConfession

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE document (
    id          TEXT PRIMARY KEY,   -- 'wcf','wlc','wsc','heidelberg'
    name        TEXT NOT NULL,
    shortcode   TEXT NOT NULL,      -- citation prefix: WCF / WLC / WSC / HC
    version     TEXT NOT NULL,
    license     TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    retrieved   TEXT NOT NULL,
    checksum    TEXT NOT NULL
);

CREATE TABLE section (
    document     TEXT NOT NULL REFERENCES document(id),
    key          TEXT NOT NULL,     -- 'chapter.section' or question number
    ord          INTEGER NOT NULL,  -- stable display order within the document
    chapter      INTEGER,           -- WCF chapter / NULL for catechisms
    section      INTEGER,           -- WCF section / NULL for catechisms
    question     INTEGER,           -- catechism question / NULL for WCF
    lords_day    INTEGER,           -- Heidelberg Lord's Day / NULL otherwise
    title        TEXT,              -- chapter title where present
    text         TEXT NOT NULL,
    proof_texts  TEXT,              -- JSON [{book,chapter,verse_start,...}] or NULL
    PRIMARY KEY (document, key)
);

CREATE INDEX idx_section_document ON section (document);
"""


def _connect(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def write_confessions(
    parsed: dict[str, ParsedConfession], out_path: Path
) -> None:
    """Write all parsed confessions to ``out_path`` (deterministic order)."""
    conn = _connect(out_path)
    try:
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [("schema_version", "1"), ("resource_type", "confession")],
        )

        for did in sorted(parsed):
            pc = parsed[did]
            src = CONFESSION_SOURCES[did]
            # All chunks of a document share one provenance; take the first.
            prov = pc.chunks[0].provenance if pc.chunks else None
            if prov is None:
                continue
            conn.execute(
                "INSERT INTO document VALUES (?,?,?,?,?,?,?,?)",
                (did, src.name, src.shortcode, prov.version, prov.license,
                 prov.url, prov.retrieved, prov.checksum),
            )
            for ord_i, ch in enumerate(pc.chunks):
                m = ch.meta
                proofs = m.get("proof_texts") or []
                conn.execute(
                    "INSERT INTO section VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        did, ch.key, ord_i,
                        m.get("chapter"), m.get("section"),
                        m.get("question"), m.get("lords_day"),
                        m.get("chapter_title"),
                        ch.text,
                        json.dumps(proofs, separators=(",", ":"), sort_keys=True)
                        if proofs else None,
                    ),
                )
        conn.commit()
    finally:
        conn.close()
