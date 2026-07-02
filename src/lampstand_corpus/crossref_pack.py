"""TSK cross-reference pack (``bundled_crossrefs.sqlite``) — Rank-13 corpus half.

Emits the resolving TSK edges (crossrefs.sqlite; signed votes preserved) into a
compact bundled pack using the pack-diet encoding discipline, plus a
per-pericope expansion table the app's HybridRetriever uses to pull
TSK-adjacent pericopes as extra Ask grounding candidates.

Two tables (contract: docs/crossrefs-pack.md):

  crossref        one row per SOURCE verse, keyed by an arithmetic verse key
                  (below); ``targets`` BLOB holds the vote-ranked target list
                  in the OpenBible file order (already vote-sorted per source):
                      [ start_key uvarint ][ (end_key−start_key) uvarint ]
                      [ zigzag(votes) uvarint ]        × n_targets
  chunk_crossref  one row per SCRIPTURE retrieval chunk (every translation),
                  keyed by the pack-diet INTEGER chunk id; ``neighbors`` BLOB
                  holds the top-N cross-referenced BSB pericopes by aggregated
                  vote weight:
                      [ neighbor_chunk_id uvarint ][ weight uvarint ]  × n

Verse key (no lookup table needed; the app computes it from its BookId order):

    verse_key = book_ord * 1_000_000 + chapter * 1_000 + verse

with ``book_ord`` the 0-based index in the canonical 66-book order
(books.ORDER). Chapters ≤ 150 and verses ≤ 176 on the KJV spine, so the
components never collide; keys are monotone in canonical Scripture order.

Only edges with ``src_resolves=1 AND tgt_resolves=1`` are packed (344,794 of
344,799) — the 5 non-resolving refs remain in the build DB and its validation
flags, per the never-drop rule; a runtime pack must not point at verses that
don't exist.

Expansion aggregation (top-N, N=8): for each Scripture chunk, every TSK edge
whose source verse falls inside the chunk's verse window contributes its
target(s); a target range maps to the BSB pericope chunk(s) covering it, and
votes are SUMMED per target chunk (signed — community downvotes subtract).
The chunk's own verse range is excluded (self-reference), targets with a
non-positive aggregate are dropped (downvoted refs must not become grounding),
and the top 8 by (weight desc, chunk string-id asc) are kept. N=8 keeps the
per-hit injection budget small next to the app's fusion depths (bm25PerType=20
/ denseDepth=20): boosting the top ~5 Scripture hits injects ≤ 40 candidates,
comparable to one retriever's contribution. Neighbors are BSB chunks so the
same row set works against both the bundled (BSB-only) and full indexes; the
app's translation dedup treats a BSB pericope as the same row as its KJV/ASV/
WEB twins. Source rows exist for EVERY translation's chunk so whichever
translation surfaces as a hit still finds its neighbors.

Deterministic throughout: fixed orders, no RNG, no timestamps.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import books
from .pack_codec import (
    read_uvarint,
    write_uvarint,
    zigzag_decode,
    zigzag_encode,
)

CROSSREFS_PACK_FORMAT = "crossrefs-pack-v1"
VERSE_KEY_FORMAT = "book_ord*1000000 + chapter*1000 + verse (book_ord 0-based)"
TARGET_FORMAT = "uvarint start_key, end_key_delta, zigzag_votes"
NEIGHBOR_FORMAT = "uvarint chunk_id, weight"
EXPANSION_TOP_N = 8

_BOOK_MUL = 1_000_000
_CHAPTER_MUL = 1_000


# --- verse keys -----------------------------------------------------------------
def verse_key(book: str, chapter: int, verse: int) -> int:
    """Arithmetic verse key on the canonical spine (see module docstring)."""
    ord_ = books.ORDER_INDEX[book]
    if not (0 < chapter < _CHAPTER_MUL and 0 < verse < _CHAPTER_MUL):
        raise ValueError(f"chapter/verse out of key range: {book} {chapter}:{verse}")
    return ord_ * _BOOK_MUL + chapter * _CHAPTER_MUL + verse


def verse_key_parts(key: int) -> tuple[str, int, int]:
    """Inverse of :func:`verse_key` → (book, chapter, verse)."""
    return (books.ORDER[key // _BOOK_MUL],
            (key % _BOOK_MUL) // _CHAPTER_MUL,
            key % _CHAPTER_MUL)


# --- target blobs -----------------------------------------------------------------
def encode_targets(targets: list[tuple[int, int, int]]) -> bytes:
    """Encode ``[(start_key, end_key, votes)]`` in the given (vote-rank) order."""
    buf = bytearray()
    for start, end, votes in targets:
        if end < start:
            raise ValueError(f"reversed target range: {start}..{end}")
        write_uvarint(buf, start)
        write_uvarint(buf, end - start)
        write_uvarint(buf, zigzag_encode(votes))
    return bytes(buf)


def decode_targets(blob: bytes) -> list[tuple[int, int, int]]:
    """Decode a target blob back to ``[(start_key, end_key, votes)]``."""
    out: list[tuple[int, int, int]] = []
    pos = 0
    while pos < len(blob):
        start, pos = read_uvarint(blob, pos)
        delta, pos = read_uvarint(blob, pos)
        zz, pos = read_uvarint(blob, pos)
        out.append((start, start + delta, zigzag_decode(zz)))
    return out


# --- neighbor blobs -----------------------------------------------------------------
def encode_neighbors(neighbors: list[tuple[int, int]]) -> bytes:
    """Encode ``[(chunk_int_id, weight)]`` (weight >= 1) in rank order."""
    buf = bytearray()
    for cid, weight in neighbors:
        if weight <= 0:
            raise ValueError(f"neighbor weight must be >= 1 (got {weight})")
        write_uvarint(buf, cid)
        write_uvarint(buf, weight)
    return bytes(buf)


def decode_neighbors(blob: bytes) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = 0
    while pos < len(blob):
        cid, pos = read_uvarint(blob, pos)
        w, pos = read_uvarint(blob, pos)
        out.append((cid, w))
    return out


# --- edge rows ------------------------------------------------------------------------
def build_edge_rows(crossrefs_db: Path) -> tuple[list[tuple[int, int, bytes]], dict]:
    """One packed row per source verse, canonical spine order.

    Targets keep the OpenBible file order (``rank`` — already vote-sorted per
    source verse upstream); signed votes are preserved via zigzag. Returns
    ``(rows, stats)``.
    """
    conn = sqlite3.connect(crossrefs_db)
    try:
        rows = conn.execute(
            "SELECT src_book, src_chapter, src_verse, tgt_book, tgt_chapter, "
            "tgt_verse, tgt_end_book, tgt_end_chapter, tgt_end_verse, votes, rank "
            "FROM crossref WHERE src_resolves=1 AND tgt_resolves=1"
        ).fetchall()
    finally:
        conn.close()

    by_source: dict[int, list[tuple[int, int, int, int]]] = {}
    for (sb, sc, sv, tb, tc, tv, teb, tec, tev, votes, rank) in rows:
        skey = verse_key(sb, sc, sv)
        by_source.setdefault(skey, []).append(
            (rank, verse_key(tb, tc, tv), verse_key(teb, tec, tev), votes))

    out: list[tuple[int, int, bytes]] = []
    n_edges = 0
    for skey in sorted(by_source):
        targets = sorted(by_source[skey])  # rank ascending (file order)
        n_edges += len(targets)
        blob = encode_targets([(t[1], t[2], t[3]) for t in targets])
        out.append((skey, len(targets), blob))
    return out, {"n_sources": len(out), "n_edges": n_edges}


# --- per-pericope expansion --------------------------------------------------------------
@dataclass(frozen=True)
class ScriptureChunkRef:
    """The slice of chunk metadata the expansion needs (any translation)."""

    string_id: str
    source: str      # translation id (bsb/kjv/asv/web)
    book: str
    chapter: int
    verse_start: int
    verse_end: int


def load_scripture_chunk_refs(emb_db: Path) -> list[ScriptureChunkRef]:
    """Scripture RETRIEVAL chunks (indexed verse children) in id order.

    Context-only pericope parents are excluded — the expansion keys and its
    neighbors are retrieval units (the app expands a hit to its parent via
    ``chunk.parent_id`` separately). Neighbors are therefore BSB verse-level
    chunks under the Rank-8 re-chunk.
    """
    conn = sqlite3.connect(emb_db)
    try:
        rows = conn.execute(
            "SELECT id, source, book, chapter, verse_start, verse_end "
            "FROM chunk WHERE resource_type='scripture' AND indexed=1 "
            "ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [ScriptureChunkRef(*r) for r in rows]


def build_expansion(
    chunks: list[ScriptureChunkRef],
    crossrefs_db: Path,
    *,
    top_n: int = EXPANSION_TOP_N,
) -> dict[str, list[tuple[str, int]]]:
    """Top-N TSK-adjacent BSB pericopes per Scripture chunk, by summed votes.

    Returns ``{chunk_string_id: [(neighbor_string_id, weight), ...]}`` for every
    chunk with at least one positive-weight neighbor. See the module docstring
    for the aggregation rules (signed sum, self-range excluded, weight > 0).
    """
    # Edges grouped by source verse key.
    conn = sqlite3.connect(crossrefs_db)
    try:
        edge_rows = conn.execute(
            "SELECT src_book, src_chapter, src_verse, tgt_book, tgt_chapter, "
            "tgt_verse, tgt_end_book, tgt_end_chapter, tgt_end_verse, votes "
            "FROM crossref WHERE src_resolves=1 AND tgt_resolves=1"
        ).fetchall()
    finally:
        conn.close()
    edges_by_verse: dict[tuple[str, int, int], list[tuple]] = {}
    for e in edge_rows:
        edges_by_verse.setdefault((e[0], e[1], e[2]), []).append(e[3:])

    # BSB pericopes indexed per (book, chapter) for target mapping, plus a
    # verse-range key so any translation's chunk knows its BSB twin's range.
    bsb_by_chapter: dict[tuple[str, int], list[ScriptureChunkRef]] = {}
    for c in chunks:
        if c.source == "bsb":
            bsb_by_chapter.setdefault((c.book, c.chapter), []).append(c)

    def bsb_covering(book: str, ch: int, vs: int, ve: int) -> list[ScriptureChunkRef]:
        return [
            c for c in bsb_by_chapter.get((book, ch), [])
            if c.verse_start <= ve and c.verse_end >= vs
        ]

    out: dict[str, list[tuple[str, int]]] = {}
    for chunk in chunks:
        own_range = (chunk.book, chunk.chapter, chunk.verse_start, chunk.verse_end)
        weights: dict[str, int] = {}
        for v in range(chunk.verse_start, chunk.verse_end + 1):
            for (tb, tc, tv, teb, tec, tev, votes) in edges_by_verse.get(
                    (chunk.book, chunk.chapter, v), ()):
                for target in _bsb_chunks_for_range(
                        bsb_covering, tb, tc, tv, teb, tec, tev):
                    trange = (target.book, target.chapter,
                              target.verse_start, target.verse_end)
                    if trange == own_range:
                        continue  # self-reference (any translation)
                    weights[target.string_id] = (
                        weights.get(target.string_id, 0) + votes)
        ranked = sorted(
            ((sid, w) for sid, w in weights.items() if w > 0),
            key=lambda e: (-e[1], e[0]))[:top_n]
        if ranked:
            out[chunk.string_id] = ranked
    return out


def _bsb_chunks_for_range(
    bsb_covering, tb: str, tc: int, tv: int, teb: str, tec: int, tev: int
) -> list[ScriptureChunkRef]:
    """BSB chunks overlapping a (possibly chapter/book-crossing) target range."""
    if (tb, tc) == (teb, tec):
        return bsb_covering(tb, tc, tv, tev)
    # Rare multi-chapter/book range: walk the canonical spine.
    out: list[ScriptureChunkRef] = []
    start_ord, end_ord = books.ORDER_INDEX[tb], books.ORDER_INDEX[teb]
    for bo in range(start_ord, end_ord + 1):
        book = books.ORDER[bo]
        counts = books.VERSE_COUNTS.get(book, [])
        ch_first = tc if bo == start_ord else 1
        ch_last = tec if bo == end_ord else len(counts)
        for ch in range(ch_first, ch_last + 1):
            lo = tv if (bo, ch) == (start_ord, tc) else 1
            hi = tev if (bo, ch) == (end_ord, tec) else 10_000
            out.extend(bsb_covering(book, ch, lo, hi))
    return out


# --- pack writer ------------------------------------------------------------------------
_CROSSREFS_SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE crossref (
    src_verse INTEGER PRIMARY KEY,  -- verse key (see meta.verse_key)
    n_targets INTEGER NOT NULL,
    targets   BLOB NOT NULL         -- see meta.target_format (vote-rank order)
);
CREATE TABLE chunk_crossref (
    chunk_id    INTEGER PRIMARY KEY, -- pack-diet int chunk id (search pack)
    n_neighbors INTEGER NOT NULL,
    neighbors   BLOB NOT NULL        -- see meta.neighbor_format (weight desc)
);
"""


def build_bundled_crossrefs(
    crossrefs_db: Path,
    emb_db: Path,
    id_map: dict[str, int],
    dst_path: Path,
) -> dict:
    """Write ``bundled_crossrefs.sqlite`` (edge table + expansion). Returns stats."""
    conn = sqlite3.connect(crossrefs_db)
    try:
        src_meta = conn.execute(
            "SELECT license, attribution FROM source WHERE id='tsk'").fetchone()
    finally:
        conn.close()
    license_, attribution = src_meta if src_meta else ("", "")

    edge_rows, stats = build_edge_rows(crossrefs_db)
    chunks = load_scripture_chunk_refs(emb_db)
    expansion = build_expansion(chunks, crossrefs_db)
    expansion_rows = [
        (id_map[sid], len(nbrs),
         encode_neighbors([(id_map[n], w) for n, w in nbrs]))
        for sid, nbrs in sorted(expansion.items(), key=lambda kv: id_map[kv[0]])
    ]

    if dst_path.exists():
        dst_path.unlink()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(dst_path)
    try:
        dst.executescript(_CROSSREFS_SCHEMA)
        dst.executemany("INSERT INTO crossref VALUES (?,?,?)", edge_rows)
        dst.executemany("INSERT INTO chunk_crossref VALUES (?,?,?)", expansion_rows)
        dst.executemany(
            "INSERT INTO meta VALUES (?,?)",
            [
                ("schema_version", "2"),
                ("format", CROSSREFS_PACK_FORMAT),
                ("resource_type", "crossrefs"),
                ("verse_key", VERSE_KEY_FORMAT),
                ("target_format", TARGET_FORMAT),
                ("neighbor_format", NEIGHBOR_FORMAT),
                ("id_assignment", "ascending-string-chunk-id-1based"),
                ("expansion_top_n", str(EXPANSION_TOP_N)),
                ("expansion_neighbor_scope", "bsb pericope chunks"),
                ("scope", "resolving TSK edges (src_resolves=1 AND tgt_resolves=1)"),
                ("n_sources", str(stats["n_sources"])),
                ("n_edges", str(stats["n_edges"])),
                ("n_expanded_chunks", str(len(expansion_rows))),
                ("license", license_),
                ("attribution", attribution),
            ],
        )
        dst.commit()
    finally:
        dst.close()
    return {
        "format": CROSSREFS_PACK_FORMAT,
        "n_sources": stats["n_sources"],
        "n_edges": stats["n_edges"],
        "n_expanded_chunks": len(expansion_rows),
        "expansion_top_n": EXPANSION_TOP_N,
        "license": license_,
        "attribution": attribution,
    }
