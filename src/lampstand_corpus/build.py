"""Build ``bibles.sqlite`` from parsed USFM books.

The schema is verse-addressable for display and carries full provenance per the
normalized-schema contract. Output is deterministic: rows are inserted in a fixed
canonical order, no timestamps or autoincrement-dependent ids leak in, and the
SQLite file is built with stable settings so the same snapshots produce a
bit-for-bit identical database.

Red-letter (words-of-Christ) data is stored as a JSON array of ``[start, end]``
character offsets on each verse row, plus a boolean flag for cheap filtering.

Disputed critical-text omissions (books.OMITTED_VARIANTS) are standardized here:
every such reference resolves to a row in every translation — a real verse where
the translation includes it, or an ``omitted=1`` empty row where it does not — so
the app can render a uniform footnote / tap-to-explain affordance. ``source_note``
carries the source's textual footnote on an omitted verse when one is present.
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
    text         TEXT NOT NULL,     -- "" for an omitted (critical-text) verse
    red_letter   INTEGER NOT NULL DEFAULT 0,  -- 0/1: any words-of-Christ
    wj_spans     TEXT,              -- JSON [[start,end],...] or NULL
    omitted      INTEGER NOT NULL DEFAULT 0,  -- 0/1: critical-text omission, empty body
    source_note  TEXT,              -- textual footnote on an omitted verse, or NULL
    superscription TEXT,            -- Hebrew Psalm superscription (\d) on this verse, or NULL
    para_start   INTEGER NOT NULL DEFAULT 0,  -- 0/1: verse opens a prose paragraph (\p)
    PRIMARY KEY (translation, book, chapter, verse_start)
);

CREATE INDEX idx_verse_ref ON verse (book, chapter, verse_start);
CREATE INDEX idx_verse_translation ON verse (translation);

-- Section headings (\s-family) attached to the verse they precede (Rank 8a).
-- Only the BSB carries a real heading apparatus (3,096 headings); the other
-- translations' stray heading lines are kept verbatim, never judged.
CREATE TABLE heading (
    translation  TEXT NOT NULL REFERENCES translation(id),
    book         TEXT NOT NULL REFERENCES book(id),
    chapter      INTEGER NOT NULL,
    verse_start  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    PRIMARY KEY (translation, book, chapter, verse_start)
);
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
                ("schema_version", "2"),
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
        # As we write, we standardize the disputed critical-text omissions so every
        # reference in books.OMITTED_VARIANTS resolves to a row in every translation:
        #   * a parsed verse whose body is empty AND whose ref is a known variant
        #     is marked omitted=1 (ASV/WEB empties);
        #   * a known variant with no parsed row at all gets an injected omitted=1
        #     empty row in canonical position (BSB drops these rows entirely).
        for tid in sorted(parsed):
            present_variants: set[tuple[str, int, int]] = set()
            for book_id in books.ORDER:
                pb = parsed[tid].get(book_id)
                if pb is None:
                    continue
                # Build the verse list for this book, then splice in any injected
                # omitted rows so ordering stays canonical and deterministic.
                rows: list[tuple] = []
                for v in sorted(pb.verses, key=lambda x: (x.chapter, x.verse_start)):
                    ref = (book_id, v.chapter, v.verse_start)
                    is_variant = ref in books.OMITTED_VARIANT_SET
                    if is_variant:
                        present_variants.add(ref)
                    spans = [[s.start, s.end] for s in v.wj_spans]
                    omitted = 1 if (is_variant and v.text == "") else 0
                    rows.append((
                        tid, book_id, v.chapter, v.verse_start, v.verse_end,
                        v.text, 1 if spans else 0,
                        json.dumps(spans, separators=(",", ":")) if spans else None,
                        omitted, v.source_note, v.superscription,
                        1 if v.para_start else 0,
                    ))

                # Inject omitted=1 empty rows for variants this translation dropped
                # entirely (no parsed verse). BSB carries the omitted verse's wording
                # on the *preceding* verse's footnote via an embedded \fv marker; we
                # recover it (pb.omission_notes) and attach it as source_note. When no
                # \fv segment is recoverable the note stays NULL — never fabricated —
                # and the validator flags it for human review.
                injected = [
                    (tid, b, ch, vs, vs, "", 0, None, 1,
                     pb.omission_notes.get((ch, vs)), None, 0)
                    for (b, ch, vs) in books.OMITTED_VARIANTS
                    if b == book_id and (b, ch, vs) not in present_variants
                ]
                rows.extend(injected)
                rows.sort(key=lambda r: (r[2], r[3]))  # (chapter, verse_start)
                conn.executemany(
                    "INSERT INTO verse VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)

                # Section headings, deduped per (chapter, verse) keeping the
                # first (multiple \s levels before one verse join with ' · ').
                by_ref: dict[tuple[int, int], list[str]] = {}
                for ch, vs, text in pb.headings:
                    by_ref.setdefault((ch, vs), []).append(text)
                conn.executemany(
                    "INSERT INTO heading VALUES (?,?,?,?,?)",
                    [
                        (tid, book_id, ch, vs, " · ".join(texts))
                        for (ch, vs), texts in sorted(by_ref.items())
                    ],
                )
        conn.commit()
    finally:
        conn.close()
