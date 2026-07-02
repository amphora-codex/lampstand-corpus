"""P7 — packaging: split the built per-resource DBs into a bundled pack (ships in
the app binary) and free on-demand packs (downloaded on first launch).

Architect-locked split:

  * **Bundled** (ships in the binary; target well under ~150-200 MB):
      - BSB Bible (``bundled_bibles.sqlite``, schema-identical to ``bibles.sqlite``
        but scoped to the single ``bsb`` translation),
      - Westminster Shorter Catechism (``bundled_confessions.sqlite``, scoped to
        the ``wsc`` document),
      - a BSB-scoped dense + BM25 search index (``bundled_search.sqlite``): the
        BSB-scripture and WSC-confession chunks only, with the **BM25 statistics
        recomputed over that subset** so the app scores correctly against the
        bundled corpus (a filtered copy of the global BM25 would carry the wrong
        ``avgdl`` / document frequencies).

  * **On-demand** (free, downloaded on first launch): KJV/ASV/WEB, all
    commentaries, the remaining confessions, lexicons + tagged text, cross-refs,
    and the full embeddings index. These are byte-faithful **filtered copies** of
    the built DBs (same schema), so the on-demand format == the built format and
    nothing new has to be re-validated.

Every pack file is built deterministically (fixed row order, no wall-clock, the
same sqlite settings as the upstream builders) and hashed (SHA-256). A committed
``corpus_manifest.json`` records pack contents, per-file SHA-256 + byte size, the
rolled-up source licenses/attributions (the acknowledgements data the app renders),
and a corpus-version placeholder. The pack ``.sqlite`` files themselves are written
under ``output/packs/`` and are **gitignored** — only the manifest is committed.

The pipeline never marks a version ship-ready: this produces a *candidate* set of
packs + the manifest; the architect's 23-point spot-check still gates ship.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .build_embeddings import BM25_B, BM25_K1, build_bm25
from .embeddings import EMBED_DIM, Chunk
from .pack_codec import assign_int_ids, encode_postings, quantize_int8

# Corpus version placeholder. The pipeline NEVER finalizes a version — tagging
# happens only after the architect's 23-point spot-check passes. This string is a
# placeholder the human replaces (and tags) at ship time.
# v2: the Rank-8/14/7 re-chunk release — every chunk id changed (dual
# granularity, structural headers, commentary splits, BSB-only dense).
CORPUS_VERSION_PLACEHOLDER = "corpus-v2.0.0-candidate"

# Bundled-pack scope (architect-locked).
BUNDLED_TRANSLATION = "bsb"
BUNDLED_CONFESSION = "wsc"

# Size targets (from the spec / architect note), in bytes, for the FLAG check.
BUNDLED_TARGET_MAX = 200 * 1024 * 1024  # well under ~150-200 MB
# Pack-diet target for the on-demand SEARCH pack (with display text). If the
# built pack exceeds this, the display-text decision is FLAGGED for the
# architect (documented fallback: drop text and resolve display text from the
# per-resource packs).
SEARCH_PACK_TARGET_MAX = 250 * 1024 * 1024

# --- Pack-diet v2 format identifiers (the app-side contract; docs/pack-diet.md)
SEARCH_PACK_FORMAT = "search-pack-v2"
VECTORS_PACK_FORMAT = "vectors-pack-v2"
POSTING_FORMAT = "uvarint-gap-tf-v1"
ID_ASSIGNMENT = "ascending-string-chunk-id-1based"
VECTOR_FORMAT_INT8 = "int8-symmetric-per-vector-le"
VECTOR_FORMAT_FP32 = "float32-le"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_tree(root: Path) -> str:
    """Deterministic SHA-256 of a directory tree (for the .mlpackage bundle).

    A ``.mlpackage`` is a *directory*, so a single-file ``shasum`` cannot hash it.
    This folds every regular file's relative POSIX path and content hash into one
    digest, walking paths in sorted order so the result is platform-stable and
    reproducible. The iOS sync side mirrors this exact recipe
    (``find -s <dir> -type f | sorted relpath + per-file sha256``) so the manifest
    tree hash verifies on both ends.

    Recipe (must match the shell mirror):
      for each regular file under ``root``, sorted by POSIX relpath:
        update(relpath + "\\0" + sha256(file_bytes) + "\\n")
    Symlinks are resolved to their targets (HF/Core ML stores no symlinks inside a
    freshly written .mlpackage, but resolving keeps the hash content-addressed if
    one ever appears). Empty directories do not affect the hash.
    """
    if root.is_file():
        return _sha256_file(root)
    entries: list[tuple[str, str]] = []
    for p in sorted(root.rglob("*"), key=lambda x: x.relative_to(root).as_posix()):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        entries.append((rel, _sha256_file(p.resolve())))
    h = hashlib.sha256()
    for rel, file_hash in sorted(entries):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(file_hash.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _copy_schema(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    """Copy every table/index DDL from ``src`` to ``dst`` verbatim.

    Preserves the upstream schema exactly so an on-demand / bundled pack is the
    same shape as the DB it was filtered from (the app reads one format).
    """
    rows = src.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
        "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name"
    ).fetchall()
    for _type, _name, sql in rows:
        dst.execute(sql)


def _new_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _copy_filtered_table(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    *,
    where: str = "",
    params: tuple = (),
    order_by: str = "",
) -> int:
    """Copy rows of ``table`` from src→dst, optionally filtered, in a fixed order.

    Returns the row count written. Column order is taken from the source schema so
    the copy is faithful regardless of declaration order.
    """
    cols = _table_columns(src, table)
    col_list = ", ".join(cols)
    sql = f"SELECT {col_list} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    rows = src.execute(sql, params).fetchall()
    placeholders = ", ".join("?" for _ in cols)
    dst.executemany(
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})", rows
    )
    return len(rows)


# --- Bibles ------------------------------------------------------------------
def _filter_bibles(
    src_path: Path, dst_path: Path, translations: list[str]
) -> dict:
    """Write a bibles DB scoped to ``translations`` (schema-identical)."""
    src = sqlite3.connect(src_path)
    dst = _new_db(dst_path)
    try:
        _copy_schema(src, dst)
        _copy_filtered_table(src, dst, "meta", order_by="key")
        _copy_filtered_table(src, dst, "book", order_by="ord")
        tset = ",".join(f"'{t}'" for t in sorted(translations))
        n_tr = _copy_filtered_table(
            src, dst, "translation", where=f"id IN ({tset})", order_by="id"
        )
        n_v = _copy_filtered_table(
            src, dst, "verse",
            where=f"translation IN ({tset})",
            order_by="translation, book, chapter, verse_start",
        )
        dst.commit()
        return {"translations": sorted(translations), "n_translations": n_tr,
                "n_verses": n_v}
    finally:
        src.close()
        dst.close()


# --- Confessions -------------------------------------------------------------
def _filter_confessions(
    src_path: Path, dst_path: Path, documents: list[str]
) -> dict:
    """Write a confessions DB scoped to ``documents`` (schema-identical)."""
    src = sqlite3.connect(src_path)
    dst = _new_db(dst_path)
    try:
        _copy_schema(src, dst)
        _copy_filtered_table(src, dst, "meta", order_by="key")
        dset = ",".join(f"'{d}'" for d in sorted(documents))
        n_d = _copy_filtered_table(
            src, dst, "document", where=f"id IN ({dset})", order_by="id"
        )
        n_s = _copy_filtered_table(
            src, dst, "section",
            where=f"document IN ({dset})", order_by="document, ord",
        )
        dst.commit()
        return {"documents": sorted(documents), "n_documents": n_d,
                "n_sections": n_s}
    finally:
        src.close()
        dst.close()


# --- Generic full copy (on-demand byte-faithful packs) -----------------------
def _copy_whole_db(src_path: Path, dst_path: Path) -> None:
    """Deterministic full copy of a built DB into a pack file.

    Re-inserts every table's rows in a stable order (primary-key / rowid order) so
    the output is bit-for-bit reproducible and free of any prior file's page
    layout. Indexes are rebuilt from the copied schema.
    """
    src = sqlite3.connect(src_path)
    dst = _new_db(dst_path)
    try:
        _copy_schema(src, dst)
        tables = [
            r[0] for r in src.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            # Order by the table's own rowid/PK for a stable, reproducible copy.
            order = _stable_order(src, table)
            _copy_filtered_table(src, dst, table, order_by=order)
        dst.commit()
    finally:
        src.close()
        dst.close()


def _stable_order(conn: sqlite3.Connection, table: str) -> str:
    """Pick a deterministic ORDER BY for a table (PK cols, else all cols)."""
    info = list(conn.execute(f"PRAGMA table_info({table})"))
    pk_cols = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5] > 0]
    if pk_cols:
        return ", ".join(pk_cols)
    return ", ".join(r[1] for r in info)


# --- Pack-diet v2 search / vectors packs ---------------------------------------
def _read_chunks_for(
    emb_path: Path, predicate_sql: str, params: tuple
) -> tuple[list[Chunk], dict[str, bytes]]:
    """Load a chunk scope + its float32 vector blobs from embeddings.sqlite.

    Returns ``(chunks, vectors_by_string_id)`` ordered by string chunk id
    ascending (== integer-id ascending under ``assign_int_ids``), so every pack
    write is reproducible. Vector bytes are the exact stored float32 blobs —
    never re-encoded; quantization (if any) happens at pack-write time.
    """
    conn = sqlite3.connect(emb_path)
    try:
        rows = conn.execute(
            "SELECT c.id, c.resource_type, c.source, c.anchor, c.book, c.chapter, "
            "c.verse_start, c.verse_end, c.key, c.text, c.text_checksum, "
            "c.truncated, c.header, c.parent_id, c.indexed, c.question, "
            "c.lords_day, c.conf_chapter, c.conf_section, c.article, e.vector "
            "FROM chunk c LEFT JOIN embedding e ON e.chunk_id = c.id "
            f"WHERE {predicate_sql} "
            "ORDER BY c.id",
            params,
        ).fetchall()
    finally:
        conn.close()
    chunks: list[Chunk] = []
    vectors: dict[str, bytes] = {}
    for r in rows:
        (cid, rtype, source, anchor, book, chapter, vs, ve, key, text,
         tcs, trunc, header, parent_id, indexed, question, lords_day,
         conf_chapter, conf_section, article, vec) = r
        chunks.append(Chunk(
            id=cid, resource_type=rtype, source=source, anchor=anchor,
            book=book, chapter=chapter, verse_start=vs, verse_end=ve,
            key=key, text=text, text_checksum=tcs, truncated=bool(trunc),
            header=header or "", parent=parent_id, indexed=bool(indexed),
            embed=vec is not None, question=question, lords_day=lords_day,
            conf_chapter=conf_chapter, conf_section=conf_section,
            article=article,
        ))
        if vec is not None:
            vectors[cid] = vec
    return chunks, vectors


# The v2 search-pack schema (docs/pack-diet.md is the app-side contract).
# vs v1: chunk.id is the stable INTEGER id (string_id kept for provenance),
# bm25_doc is folded into chunk.doc_len, bm25_posting rows + idx_posting_chunk
# are replaced by one varint-delta BLOB per term.
_SEARCH_SCHEMA_V2 = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE chunk (
    id            INTEGER PRIMARY KEY,  -- stable int id (per corpus version)
    string_id     TEXT NOT NULL UNIQUE, -- content-addressed id (provenance)
    resource_type TEXT NOT NULL,
    source        TEXT NOT NULL,
    anchor        TEXT NOT NULL,
    book          TEXT,
    chapter       INTEGER,
    verse_start   INTEGER,
    verse_end     INTEGER,
    key           TEXT,
    text          TEXT,                 -- display text (NULL when excluded)
    text_checksum TEXT NOT NULL,
    truncated     INTEGER NOT NULL DEFAULT 0,
    doc_len       INTEGER NOT NULL,     -- BM25 token count (0 for parents)
    header        TEXT NOT NULL DEFAULT '', -- structural header (Rank 8e)
    parent_id     INTEGER,              -- pericope parent int id (children)
    indexed       INTEGER NOT NULL DEFAULT 1, -- 0 = context-only parent
    question      INTEGER,              -- Rank 14 catechism metadata
    lords_day     INTEGER,
    conf_chapter  INTEGER,
    conf_section  INTEGER,
    article       INTEGER
);
CREATE INDEX idx_chunk_resource ON chunk (resource_type);
CREATE INDEX idx_chunk_source ON chunk (source);
CREATE INDEX idx_chunk_ref ON chunk (book, chapter, verse_start);
CREATE INDEX idx_chunk_parent ON chunk (parent_id);
CREATE TABLE bm25_term (
    term_id  INTEGER PRIMARY KEY,
    term     TEXT NOT NULL UNIQUE,
    doc_freq INTEGER NOT NULL,
    postings BLOB NOT NULL              -- [gap uvarint, tf uvarint] * doc_freq
);
CREATE TABLE bm25_stats (
    key   TEXT PRIMARY KEY,
    value REAL NOT NULL
);
CREATE TABLE expansion (
    term      TEXT NOT NULL,   -- query token (pipeline tokenizer form)
    expansion TEXT NOT NULL,   -- token to ALSO score, down-weighted at query time
    kind      TEXT NOT NULL,   -- archaic | suffix | synonym (approved only)
    weight    REAL NOT NULL,   -- mining containment (informational)
    PRIMARY KEY (term, expansion)
);
"""

# Embedding table shared by ondemand_vectors.sqlite and (embedded) the bundled
# search pack. chunk_id is the SAME integer id space as the search pack.
_EMBEDDING_SCHEMA_V2 = """
CREATE TABLE embedding (
    chunk_id INTEGER PRIMARY KEY,
    vector   BLOB NOT NULL,   -- dim int8 (int8 format) or dim float32-le
    scale    REAL NOT NULL    -- per-vector dequant scale (1.0 for float32)
);
"""


def _vector_meta(model_meta: dict, vector_format: str, n: int) -> list[tuple[str, str]]:
    return [
        ("model_name", model_meta.get("model_name", "")),
        ("model_revision", model_meta.get("model_revision", "")),
        ("model_combined_sha256", model_meta.get("model_combined_sha256", "")),
        ("embedding_dim", str(EMBED_DIM)),
        ("vector_format", vector_format),
        ("query_instruction",
         model_meta.get(
             "query_instruction",
             "Represent this sentence for searching relevant passages: ")),
        ("n_vectors", str(n)),
    ]


def _vector_rows(
    chunks: list[Chunk], vectors: dict[str, bytes], id_map: dict[str, int],
    vector_format: str,
) -> list[tuple[int, bytes, float]]:
    """(int_id, blob, scale) rows in int-id order, quantized per the format."""
    rows: list[tuple[int, bytes, float]] = []
    for c in sorted(chunks, key=lambda c: id_map[c.id]):
        if c.id not in vectors:
            continue  # BM25-only children / context parents carry no vector
        blob = vectors[c.id]
        if vector_format == VECTOR_FORMAT_INT8:
            q, scale = quantize_int8(np.frombuffer(blob, dtype="<f4"))
            rows.append((id_map[c.id], q, scale))
        else:
            rows.append((id_map[c.id], blob, 1.0))
    return rows


def _build_search_pack(
    dst_path: Path,
    chunks: list[Chunk],
    id_map: dict[str, int],
    *,
    scope: str,
    include_text: bool = True,
    vectors: dict[str, bytes] | None = None,
    vector_format: str = VECTOR_FORMAT_INT8,
    model_meta: dict | None = None,
    bm25: dict | None = None,
    expansion_rows: list[tuple[str, str, str, float]] | None = None,
) -> dict:
    """Write a v2 search pack: chunk metadata + varint-delta BM25 posting blobs.

    BM25 statistics are recomputed over ``chunks`` with the same ``build_bm25``
    that produced the built index (recompute == stored for the full scope; for
    the bundled subset avgdl/df/N correctly reflect only that subset). When
    ``vectors`` is given (the bundled pack) the embedding table is embedded in
    the same file so the bundled index stays a single file.
    """
    indexed_chunks = [c for c in chunks if c.indexed]
    if bm25 is None:
        bm25 = build_bm25(indexed_chunks)
    ordered = sorted(chunks, key=lambda c: id_map[c.id])
    dst = _new_db(dst_path)
    try:
        dst.executescript(_SEARCH_SCHEMA_V2)

        def _display_text(c: Chunk) -> str | None:
            # Pack-diet: context-only pericope PARENTS (indexed=0) store NULL
            # display text — their text is exactly the concatenation of their
            # children's verse text, which is ALSO in this pack, so it is
            # redundant. The app reconstructs a parent's text by concatenating
            # its children's `text` in child order (docs/pack-diet.md contract).
            # Children (indexed=1) keep full display text when text is included.
            if not include_text or not c.indexed:
                return None
            return c.text

        dst.executemany(
            "INSERT INTO chunk VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (id_map[c.id], c.id, c.resource_type, c.source, c.anchor,
                 c.book, c.chapter, c.verse_start, c.verse_end, c.key,
                 _display_text(c), c.text_checksum,
                 1 if c.truncated else 0, bm25["doc_lengths"].get(c.id, 0),
                 c.header, id_map.get(c.parent) if c.parent else None,
                 1 if c.indexed else 0, c.question, c.lords_day,
                 c.conf_chapter, c.conf_section, c.article)
                for c in ordered
            ],
        )
        term_rows = []
        n_postings = 0
        for tid, (term, df) in enumerate(bm25["terms"]):
            postings = sorted(
                (id_map[cid], tf) for cid, tf in bm25["postings"][term].items())
            n_postings += len(postings)
            term_rows.append((tid, term, df, encode_postings(postings)))
        dst.executemany("INSERT INTO bm25_term VALUES (?,?,?,?)", term_rows)
        dst.executemany(
            "INSERT INTO bm25_stats VALUES (?,?)",
            [
                ("n_docs", float(bm25["n_docs"])),
                ("avgdl", float(bm25["avgdl"])),
                ("vocab_size", float(len(bm25["terms"]))),
                ("total_tokens", float(bm25["total_tokens"])),
                ("k1", BM25_K1),
                ("b", BM25_B),
            ],
        )
        if expansion_rows:
            dst.executemany(
                "INSERT INTO expansion VALUES (?,?,?,?)", expansion_rows)
        meta_rows = [
            ("schema_version", "2"),
            ("format", SEARCH_PACK_FORMAT),
            ("resource_type", "search"),
            ("scope", scope),
            ("bm25_tokenizer", "nfkc-casefold-alnum-no-stemming"),
            ("posting_format", POSTING_FORMAT),
            ("id_assignment", ID_ASSIGNMENT),
            ("text_included", "1" if include_text else "0"),
            ("n_chunks", str(len(ordered))),
            ("expansion_format", "expansion-v1"),
            ("n_expansion_rows", str(len(expansion_rows or []))),
        ]
        if vectors is not None:
            dst.executescript(_EMBEDDING_SCHEMA_V2)
            dst.executemany(
                "INSERT INTO embedding VALUES (?,?,?)",
                _vector_rows(ordered, vectors, id_map, vector_format))
            n_vec = dst.execute("SELECT count(*) FROM embedding").fetchone()[0]
            meta_rows += _vector_meta(model_meta or {}, vector_format, n_vec)
        dst.executemany("INSERT INTO meta VALUES (?,?)", meta_rows)
        dst.commit()
        return {
            "format": SEARCH_PACK_FORMAT,
            "n_chunks": len(ordered),
            "vocab_size": len(bm25["terms"]),
            "n_postings": n_postings,
            "posting_format": POSTING_FORMAT,
            "text_included": include_text,
            **({"vector_format": vector_format} if vectors is not None else {}),
        }
    finally:
        dst.close()


def _build_vectors_pack(
    dst_path: Path,
    chunks: list[Chunk],
    vectors: dict[str, bytes],
    id_map: dict[str, int],
    *,
    vector_format: str,
    model_meta: dict,
) -> dict:
    """Write the v2 vectors pack (int8 by default; float32 behind the flag)."""
    dst = _new_db(dst_path)
    try:
        dst.executescript(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            + _EMBEDDING_SCHEMA_V2)
        dst.executemany(
            "INSERT INTO embedding VALUES (?,?,?)",
            _vector_rows(chunks, vectors, id_map, vector_format))
        n_vec = dst.execute("SELECT count(*) FROM embedding").fetchone()[0]
        dst.executemany(
            "INSERT INTO meta VALUES (?,?)",
            [
                ("schema_version", "2"),
                ("format", VECTORS_PACK_FORMAT),
                ("resource_type", "vectors"),
                ("id_assignment", ID_ASSIGNMENT),
            ] + _vector_meta(model_meta, vector_format, n_vec),
        )
        dst.commit()
        return {
            "format": VECTORS_PACK_FORMAT,
            "n_vectors": n_vec,
            "vector_format": vector_format,
            "embedding_dim": EMBED_DIM,
        }
    finally:
        dst.close()


# --- Acknowledgements roll-up ------------------------------------------------
def _emb_model_meta(emb_path: Path) -> dict:
    conn = sqlite3.connect(emb_path)
    try:
        return dict(conn.execute("SELECT key, value FROM meta").fetchall())
    finally:
        conn.close()


def _ack_from_translations(bibles_path: Path) -> list[dict]:
    conn = sqlite3.connect(bibles_path)
    try:
        rows = conn.execute(
            "SELECT id, name, version, license, source_url, retrieved, checksum "
            "FROM translation ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "name": r[1], "resource_type": "scripture",
         "version": r[2], "license": r[3], "attribution": None,
         "source_url": r[4], "retrieved": r[5], "source_checksum": r[6]}
        for r in rows
    ]


def _ack_from_confessions(conf_path: Path) -> list[dict]:
    conn = sqlite3.connect(conf_path)
    try:
        rows = conn.execute(
            "SELECT id, name, version, license, source_url, retrieved, checksum "
            "FROM document ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "name": r[1], "resource_type": "confession",
         "version": r[2], "license": r[3], "attribution": None,
         "source_url": r[4], "retrieved": r[5], "source_checksum": r[6]}
        for r in rows
    ]


def _ack_from_commentaries(comm_path: Path) -> list[dict]:
    conn = sqlite3.connect(comm_path)
    try:
        rows = conn.execute(
            "SELECT id, name, version, license, retrieved, source_urls "
            "FROM commentator ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0], "name": r[1], "resource_type": "commentary",
            "version": r[2], "license": r[3], "attribution": None,
            "source_url": None, "retrieved": r[4], "source_checksum": None,
        })
    return out


def _ack_from_lexicons(lex_path: Path) -> list[dict]:
    conn = sqlite3.connect(lex_path)
    out: list[dict] = []
    try:
        for r in conn.execute(
            "SELECT id, name, version, license, source_url, retrieved, checksum "
            "FROM lexicon ORDER BY id"
        ):
            out.append({
                "id": r[0], "name": r[1], "resource_type": "lexicon",
                "version": r[2], "license": r[3], "attribution": None,
                "source_url": r[4], "retrieved": r[5], "source_checksum": r[6]})
        for r in conn.execute(
            "SELECT id, name, version, license, attribution, source_url, "
            "retrieved, checksum FROM tagged_source ORDER BY id"
        ):
            out.append({
                "id": r[0], "name": r[1], "resource_type": "tagged-text",
                "version": r[2], "license": r[3], "attribution": r[4],
                "source_url": r[5], "retrieved": r[6], "source_checksum": r[7]})
    finally:
        conn.close()
    return out


def _ack_from_crossrefs(cr_path: Path) -> list[dict]:
    conn = sqlite3.connect(cr_path)
    try:
        rows = conn.execute(
            "SELECT id, name, license, attribution, source_url, retrieved, checksum "
            "FROM source ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "name": r[1], "resource_type": "crossref",
         "version": None, "license": r[2], "attribution": r[3],
         "source_url": r[4], "retrieved": r[5], "source_checksum": r[6]}
        for r in rows
    ]


def _ack_embedding_model(emb_meta: dict) -> dict:
    return {
        "id": "embedding-model",
        "name": emb_meta.get("model_name", ""),
        "resource_type": "embedding-model",
        "version": emb_meta.get("model_revision", ""),
        "license": "MIT (BAAI/bge-small-en-v1.5)",
        "attribution": None,
        "source_url": "https://huggingface.co/BAAI/bge-small-en-v1.5",
        "retrieved": None,
        "source_checksum": emb_meta.get("model_combined_sha256", ""),
    }


def build_acknowledgements(output_dir: Path) -> list[dict]:
    """Roll up every source's license + attribution from the built DBs.

    The built DBs are the source of truth for provenance (each carries
    license/url/version/checksum per resource), so the acknowledgements the app
    renders are derived from them rather than re-stated by hand.
    """
    emb_meta = _emb_model_meta(output_dir / "embeddings.sqlite")
    acks: list[dict] = []
    acks += _ack_from_translations(output_dir / "bibles.sqlite")
    acks += _ack_from_confessions(output_dir / "confessions.sqlite")
    acks += _ack_from_commentaries(output_dir / "commentaries.sqlite")
    acks += _ack_from_lexicons(output_dir / "lexicons.sqlite")
    acks += _ack_from_crossrefs(output_dir / "crossrefs.sqlite")
    acks.append(_ack_embedding_model(emb_meta))
    return acks


# --- On-demand download tiering (F5 decision) --------------------------------
# The architect's F5 decision: ``ondemand_vectors.sqlite`` (int8 dense vectors)
# ships as part of the DEFAULT on-demand download set — NOT a separate opt-in —
# so the first-launch download yields full hybrid (BM25 + dense) retrieval out
# of the box. The retrieval index is therefore two default-tier packs grouped
# together: ``ondemand_search`` (BM25 + chunk metadata + display text) and
# ``ondemand_vectors`` (the int8 vectors). If ``ondemand_vectors`` is ever
# absent, retrieval degrades GRACEFULLY to BM25-only (the reverse fallback) —
# ``ondemand_search`` is self-sufficient for lexical search. No pack bytes or
# schema change: this is grouping/metadata only.
#
# tier: "default"  -> part of the first-launch default download set
#       "optional" -> user opts in from Manage-downloads (none today)
# download_group: coarse grouping for the app's Manage-downloads UI.
_ONDEMAND_TIER: dict[str, str] = {
    "ondemand_search.sqlite": "default",
    "ondemand_vectors.sqlite": "default",   # F5: default, not opt-in
    "ondemand_bibles.sqlite": "default",
    "ondemand_confessions.sqlite": "default",
    "ondemand_commentaries.sqlite": "default",
    "ondemand_lexicons.sqlite": "default",
    "ondemand_crossrefs.sqlite": "default",
}
_ONDEMAND_GROUP: dict[str, str] = {
    # The default retrieval index: search + vectors move together.
    "ondemand_search.sqlite": "retrieval-index",
    "ondemand_vectors.sqlite": "retrieval-index",
    "ondemand_bibles.sqlite": "content",
    "ondemand_confessions.sqlite": "content",
    "ondemand_commentaries.sqlite": "content",
    "ondemand_lexicons.sqlite": "content",
    "ondemand_crossrefs.sqlite": "content",
}


# --- Orchestration -----------------------------------------------------------
@dataclass
class PackFile:
    pack: str           # 'bundled' | 'on-demand'
    name: str           # file name under output/packs/
    role: str           # 'bibles'|'confessions'|'search'|'commentaries'|...
    bytes: int
    sha256: str
    contents: dict = field(default_factory=dict)

    @property
    def tier(self) -> str | None:
        """On-demand download tier ('default'/'optional'); None for bundled."""
        if self.pack != "on-demand":
            return None
        return _ONDEMAND_TIER.get(self.name, "default")

    @property
    def download_group(self) -> str | None:
        if self.pack != "on-demand":
            return None
        return _ONDEMAND_GROUP.get(self.name, "content")


@dataclass
class PackagingResult:
    files: list[PackFile]
    manifest: dict
    bundled_bytes: int
    ondemand_bytes: int
    flags: list[str]


# What lands in each pack. On-demand resources are byte-faithful filtered copies
# of the built DBs (same schema) — only the bibles are split (Bsb-out → on-demand
# KJV/ASV/WEB), everything else is the whole built DB.
_ONDEMAND_OTHER_TRANSLATIONS = ["asv", "kjv", "web"]
_ONDEMAND_CONFESSIONS = ["belgic", "dort", "heidelberg", "lbcf", "wcf", "wlc"]


def package_corpus(
    output_dir: Path, packs_dir: Path, *,
    vector_format: str = VECTOR_FORMAT_INT8,
    repo_root: Path | None = None,
) -> PackagingResult:
    """Produce the bundled + on-demand packs and the corpus manifest (in memory).

    Writes the pack ``.sqlite`` files under ``packs_dir`` (gitignored) and returns
    a ``PackagingResult`` carrying per-file SHA-256 + sizes, the manifest dict, and
    any size FLAGs. The manifest is written to disk by the caller (CLI).

    Pack-diet v2: the former ``ondemand_embeddings.sqlite`` byte-copy is replaced
    by ``ondemand_search.sqlite`` (chunk metadata + display text + varint-delta
    posting blobs over stable integer chunk ids) plus ``ondemand_vectors.sqlite``
    (int8 vectors + per-vector scale by default; ``vector_format=VECTOR_FORMAT_FP32``
    keeps float32). ``bundled_search.sqlite`` keeps its name but adopts the same
    v2 encoding with its vectors embedded.
    """
    packs_dir.mkdir(parents=True, exist_ok=True)
    emb_meta = _emb_model_meta(output_dir / "embeddings.sqlite")
    files: list[PackFile] = []

    def register(pack: str, name: str, role: str, contents: dict) -> None:
        p = packs_dir / name
        files.append(PackFile(
            pack=pack, name=name, role=role,
            bytes=p.stat().st_size, sha256=_sha256_file(p), contents=contents))

    # One integer-id space over the WHOLE corpus (bundled subset included), so a
    # chunk keeps one id across every pack of a corpus version.
    all_chunks, all_vectors = _read_chunks_for(
        output_dir / "embeddings.sqlite", "1=1", ())
    id_map = assign_int_ids([c.id for c in all_chunks])

    # Rank 7: BM25 over the full indexed corpus is computed ONCE here (reused
    # by the on-demand search pack) so the expansion mining can see the real
    # vocabulary; the bundled pack still recomputes scope-correct stats.
    from .expansion import build_expansion_rows
    bm25_full = build_bm25([c for c in all_chunks if c.indexed])
    expansion_rows, expansion_stats = build_expansion_rows(
        output_dir / "bibles.sqlite", dict(bm25_full["terms"]), repo_root)

    # --- Bundled pack ---
    c = _filter_bibles(
        output_dir / "bibles.sqlite",
        packs_dir / "bundled_bibles.sqlite", [BUNDLED_TRANSLATION])
    register("bundled", "bundled_bibles.sqlite", "bibles", c)

    c = _filter_confessions(
        output_dir / "confessions.sqlite",
        packs_dir / "bundled_confessions.sqlite", [BUNDLED_CONFESSION])
    register("bundled", "bundled_confessions.sqlite", "confessions", c)

    bundled_chunks = [
        c for c in all_chunks
        if (c.resource_type == "scripture" and c.source == BUNDLED_TRANSLATION)
        or (c.resource_type == "confession" and c.source == BUNDLED_CONFESSION)
    ]
    c = _build_search_pack(
        packs_dir / "bundled_search.sqlite", bundled_chunks, id_map,
        scope="bundled (bsb-scripture + wsc-confession)",
        vectors={ch.id: all_vectors[ch.id] for ch in bundled_chunks
                 if ch.id in all_vectors},
        vector_format=vector_format, model_meta=emb_meta,
        expansion_rows=expansion_rows)
    register("bundled", "bundled_search.sqlite", "search", c)

    # TSK cross-reference layer (Rank 13): edge table + per-pericope expansion,
    # BUNDLED — small, always available, and a single-purpose reader that never
    # touches the search index (contract: docs/crossrefs-pack.md).
    from .crossref_pack import build_bundled_crossrefs
    c = build_bundled_crossrefs(
        output_dir / "crossrefs.sqlite", output_dir / "embeddings.sqlite",
        id_map, packs_dir / "bundled_crossrefs.sqlite",
        confessions_db=output_dir / "confessions.sqlite")
    register("bundled", "bundled_crossrefs.sqlite", "crossrefs", c)

    # --- On-demand packs ---
    c = _filter_bibles(
        output_dir / "bibles.sqlite",
        packs_dir / "ondemand_bibles.sqlite", _ONDEMAND_OTHER_TRANSLATIONS)
    register("on-demand", "ondemand_bibles.sqlite", "bibles", c)

    c = _filter_confessions(
        output_dir / "confessions.sqlite",
        packs_dir / "ondemand_confessions.sqlite", _ONDEMAND_CONFESSIONS)
    register("on-demand", "ondemand_confessions.sqlite", "confessions", c)

    for role, fname in [
        ("commentaries", "commentaries.sqlite"),
        ("lexicons", "lexicons.sqlite"),
        ("crossrefs", "crossrefs.sqlite"),
    ]:
        dst_name = f"ondemand_{fname}"
        _copy_whole_db(output_dir / fname, packs_dir / dst_name)
        register("on-demand", dst_name, role, {"copied_from": fname})

    c = _build_search_pack(
        packs_dir / "ondemand_search.sqlite", all_chunks, id_map,
        scope="full corpus", model_meta=emb_meta, bm25=bm25_full,
        expansion_rows=expansion_rows)
    c["expansion"] = expansion_stats
    register("on-demand", "ondemand_search.sqlite", "search", c)

    c = _build_vectors_pack(
        packs_dir / "ondemand_vectors.sqlite", all_chunks, all_vectors, id_map,
        vector_format=vector_format, model_meta=emb_meta)
    register("on-demand", "ondemand_vectors.sqlite", "vectors", c)

    # The v1 pack this split supersedes; remove a stale copy so a mixed pack
    # directory can never ship both formats.
    stale = packs_dir / "ondemand_embeddings.sqlite"
    if stale.exists():
        stale.unlink()

    bundled_bytes = sum(f.bytes for f in files if f.pack == "bundled")
    ondemand_bytes = sum(f.bytes for f in files if f.pack == "on-demand")

    flags: list[str] = []
    if bundled_bytes > BUNDLED_TARGET_MAX:
        flags.append(
            f"BUNDLED pack total {bundled_bytes:,} B exceeds the "
            f"~{BUNDLED_TARGET_MAX // (1024*1024)} MB target — review the split.")
    search_bytes = next(
        f.bytes for f in files if f.name == "ondemand_search.sqlite")
    if search_bytes > SEARCH_PACK_TARGET_MAX:
        flags.append(
            f"ondemand_search.sqlite is {search_bytes:,} B (> "
            f"{SEARCH_PACK_TARGET_MAX // (1024*1024)} MiB target). Parent "
            f"pericope display text is already NULL (children keep text; see "
            f"docs/pack-diet.md); remaining bulk is commentary/lexicon child "
            f"text + BM25 postings. Next lever if further shrink is required: "
            f"drop child display text and resolve it from the per-resource "
            f"packs (architect decision).")

    manifest = _build_manifest(output_dir, files, bundled_bytes, ondemand_bytes)
    return PackagingResult(
        files=files, manifest=manifest,
        bundled_bytes=bundled_bytes, ondemand_bytes=ondemand_bytes, flags=flags)


def _build_manifest(
    output_dir: Path, files: list[PackFile], bundled_bytes: int,
    ondemand_bytes: int,
) -> dict:
    acks = build_acknowledgements(output_dir)

    def _file_entry(f: PackFile) -> OrderedDict:
        entry = OrderedDict([
            ("name", f.name), ("role", f.role),
            ("bytes", f.bytes), ("sha256", f.sha256),
        ])
        # On-demand files carry the F5 download tiering so the app's
        # Manage-downloads model can decide what to fetch on first launch.
        if f.tier is not None:
            entry["tier"] = f.tier
            entry["download_group"] = f.download_group
        entry["contents"] = f.contents
        return entry

    def pack_files(pack: str) -> tuple[int, list]:
        pf = [f for f in files if f.pack == pack]
        return sum(f.bytes for f in pf), [
            _file_entry(f) for f in sorted(pf, key=lambda x: x.name)
        ]

    bundled_total, bundled_files = pack_files("bundled")
    ondemand_total, ondemand_files = pack_files("on-demand")
    default_bytes = sum(
        f.bytes for f in files
        if f.pack == "on-demand" and f.tier == "default")

    return OrderedDict([
        ("corpus_version", CORPUS_VERSION_PLACEHOLDER),
        ("ship_ready", False),
        ("note",
         "CANDIDATE. The pipeline never marks a corpus version ship-ready — "
         "the architect's 23-point spot-check gates ship. Pack .sqlite files are "
         "gitignored (output/packs/); only this manifest is committed."),
        ("format_migration",
         "Pack diet v1→v2: ondemand_embeddings.sqlite (byte-copy of the built "
         "embeddings.sqlite; float32 vectors + one SQL row per BM25 posting "
         "keyed by TEXT chunk id) is REPLACED by ondemand_search.sqlite (chunk "
         "metadata + display text + one varint-delta posting BLOB per term over "
         "stable INTEGER chunk ids; bm25_doc folded into chunk.doc_len; "
         "idx_posting_chunk dropped) and ondemand_vectors.sqlite (int8 vectors "
         "+ per-vector scale, keyed by the same integer ids). "
         "bundled_search.sqlite keeps its name but adopts the same v2 encoding "
         "with its vectors embedded. Schema contract: docs/pack-diet.md."),
        ("embedding_dim", EMBED_DIM),
        ("packs", OrderedDict([
            ("bundled", OrderedDict([
                ("description",
                 "Ships in the app binary. BSB Bible + Westminster Shorter "
                 "Catechism + a BSB/WSC-scoped dense+BM25 search index."),
                ("delivery", "app-binary"),
                ("license_class", "public-domain / CC0 only"),
                ("total_bytes", bundled_total),
                ("files", bundled_files),
            ])),
            ("on_demand", OrderedDict([
                ("description",
                 "Free, downloaded on first launch. KJV/ASV/WEB, all "
                 "commentaries, the remaining confessions, lexicons + tagged "
                 "text, cross-references, and the full search + vectors packs "
                 "(pack-diet v2)."),
                ("delivery", "first-launch-download"),
                # F5 decision: every on-demand pack is DEFAULT tier — in
                # particular ondemand_vectors.sqlite (int8 dense vectors) ships
                # in the default first-launch set, NOT as a separate opt-in — so
                # the app has full hybrid (BM25+dense) retrieval out of the box.
                ("default_note",
                 "F5: the default first-launch download set = every file with "
                 "tier=\"default\" (all on-demand packs today). ondemand_search "
                 "+ ondemand_vectors form the default 'retrieval-index' group "
                 "(download_group); ondemand_vectors is DEFAULT, not opt-in. If "
                 "ondemand_vectors is absent, retrieval degrades GRACEFULLY to "
                 "BM25-only — ondemand_search is self-sufficient for lexical "
                 "search and is never gated on the vectors pack."),
                ("app_reader",
                 "To honor the default grouping: fetch every packs.on_demand."
                 "files[] whose tier == \"default\". Treat download_group == "
                 "\"retrieval-index\" (ondemand_search + ondemand_vectors) as one "
                 "unit; the dense arm requires ondemand_vectors, BM25 requires "
                 "only ondemand_search. tier == \"optional\" (none today) is the "
                 "user-opt-in slot."),
                ("tiers", OrderedDict([
                    ("default", OrderedDict([
                        ("bytes", default_bytes),
                        ("groups", ["retrieval-index", "content"]),
                    ])),
                ])),
                ("total_bytes", ondemand_total),
                ("files", ondemand_files),
            ])),
        ])),
        ("totals", OrderedDict([
            ("bundled_bytes", bundled_bytes),
            ("ondemand_bytes", ondemand_bytes),
            ("all_bytes", bundled_bytes + ondemand_bytes),
        ])),
        ("acknowledgements", acks),
    ])


# --- M4: packs.models subtree (Core ML query model + tokenizer vocab) ---------
def build_models_pack(
    *,
    mlpackage_path: Path,
    mlpackage_tree_sha256: str,
    mlpackage_bytes: int,
    vocab_path: Path,
    vocab_sha256: str,
    model_name: str,
    model_revision: str,
    model_combined_sha256: str,
    precision: str,
    seq_len: str,
) -> OrderedDict:
    """Build the ``packs.models`` subtree (sibling to bundled/on_demand).

    Nesting the model + vocab under ``.packs`` lets ``sync-corpus.sh verify()``
    pick them up with its existing jq walk (``.packs | to_entries[] |
    .value.files[]``) — **zero jq changes**. The ``.mlpackage`` is a directory, so
    its ``sha256`` is the deterministic *tree* hash (``_sha256_tree``), which the
    iOS sync side mirrors. The top-level ``acknowledgements[].embedding-model``
    entry is kept separately as the human-readable license/provenance record.
    """
    model_bytes = int(mlpackage_bytes)
    vocab_bytes = vocab_path.stat().st_size
    return OrderedDict([
        ("description",
         "On-device Tier-1 query-embedding model + tokenizer vocab. Synced "
         "(never committed) like the corpus packs; the app loads the .mlpackage "
         "strictly from disk (no runtime download). The .mlpackage sha256 is a "
         "deterministic directory-tree hash."),
        ("delivery", "app-binary"),
        ("license_class", "MIT (model) / Apache-2.0 (BERT vocab); static weights"),
        ("total_bytes", model_bytes + vocab_bytes),
        ("files", [
            OrderedDict([
                ("name", mlpackage_path.name),
                ("role", "embedding-model"),
                ("bytes", model_bytes),
                ("sha256", mlpackage_tree_sha256),
                ("model_name", model_name),
                ("model_revision", model_revision),
                ("model_combined_sha256", model_combined_sha256),
                ("precision", precision),
                ("seq_len", seq_len),
            ]),
            OrderedDict([
                ("name", vocab_path.name),
                ("role", "tokenizer-vocab"),
                ("bytes", vocab_bytes),
                ("sha256", vocab_sha256),
            ]),
        ]),
    ])


def preserve_models_subtree(manifest_path: Path, manifest: dict) -> bool:
    """Carry an existing manifest's ``packs.models`` into a fresh ``manifest``.

    ``package`` rebuilds the manifest from the packs it just wrote, but the
    Core ML query model + vocab entries are added later by ``coreml-export``
    (``update_manifest_models``) — without this carry-over, every ``package``
    run would silently drop them until the next model re-export. Returns True
    when a models subtree was preserved.
    """
    import json

    if not manifest_path.exists():
        return False
    try:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    models = old.get("packs", {}).get("models")
    if not models:
        return False
    manifest.setdefault("packs", {})["models"] = models
    return True


def update_manifest_models(manifest_path: Path, models_pack: OrderedDict) -> None:
    """Insert/replace the ``packs.models`` subtree in an existing manifest, on disk.

    The model export runs after ``package`` (which writes the rest of the manifest),
    so this surgically updates only ``packs.models`` — leaving bundled/on_demand,
    totals, and acknowledgements untouched. ``models`` is placed last under
    ``.packs`` (after bundled/on_demand) for readability; jq does not care about
    ordering.
    """
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packs = manifest.setdefault("packs", {})
    packs["models"] = models_pack
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
