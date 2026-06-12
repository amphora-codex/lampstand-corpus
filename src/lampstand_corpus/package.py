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

from .build_embeddings import BM25_B, BM25_K1, build_bm25
from .embeddings import EMBED_DIM, Chunk

# Corpus version placeholder. The pipeline NEVER finalizes a version — tagging
# happens only after the architect's 23-point spot-check passes. This string is a
# placeholder the human replaces (and tags) at ship time.
CORPUS_VERSION_PLACEHOLDER = "corpus-v1.0.0-candidate"

# Bundled-pack scope (architect-locked).
BUNDLED_TRANSLATION = "bsb"
BUNDLED_CONFESSION = "wsc"

# Size targets (from the spec / architect note), in bytes, for the FLAG check.
BUNDLED_TARGET_MAX = 200 * 1024 * 1024  # well under ~150-200 MB


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
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


# --- Bundled search index (BSB scripture + WSC confession; BM25 recomputed) ---
def _read_chunks_for(
    emb_path: Path, predicate_sql: str, params: tuple
) -> tuple[list[Chunk], dict[str, bytes]]:
    """Load the bundled chunk subset + their vector blobs from embeddings.sqlite.

    Returns ``(chunks, vectors_by_id)``. Chunks are returned in a fixed order
    (resource_type, source, anchor) so the bundled index is reproducible. We reuse
    the exact stored vector bytes — no re-encoding — and recompute only the BM25
    statistics over this subset.
    """
    conn = sqlite3.connect(emb_path)
    try:
        rows = conn.execute(
            "SELECT c.id, c.resource_type, c.source, c.anchor, c.book, c.chapter, "
            "c.verse_start, c.verse_end, c.key, c.text, c.text_checksum, "
            "c.truncated, e.vector "
            "FROM chunk c JOIN embedding e ON e.chunk_id = c.id "
            f"WHERE {predicate_sql} "
            "ORDER BY c.resource_type, c.source, c.anchor",
            params,
        ).fetchall()
    finally:
        conn.close()
    chunks: list[Chunk] = []
    vectors: dict[str, bytes] = {}
    for r in rows:
        (cid, rtype, source, anchor, book, chapter, vs, ve, key, text,
         tcs, trunc, vec) = r
        chunks.append(Chunk(
            id=cid, resource_type=rtype, source=source, anchor=anchor,
            book=book, chapter=chapter, verse_start=vs, verse_end=ve,
            key=key, text=text, text_checksum=tcs, truncated=bool(trunc),
        ))
        vectors[cid] = vec
    return chunks, vectors


_SEARCH_SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE chunk (
    id            TEXT PRIMARY KEY,
    resource_type TEXT NOT NULL,
    source        TEXT NOT NULL,
    anchor        TEXT NOT NULL,
    book          TEXT,
    chapter       INTEGER,
    verse_start   INTEGER,
    verse_end     INTEGER,
    key           TEXT,
    text          TEXT NOT NULL,
    text_checksum TEXT NOT NULL,
    truncated     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_chunk_resource ON chunk (resource_type);
CREATE INDEX idx_chunk_source ON chunk (source);
CREATE INDEX idx_chunk_ref ON chunk (book, chapter, verse_start);
CREATE TABLE embedding (
    chunk_id TEXT PRIMARY KEY REFERENCES chunk(id),
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);
CREATE TABLE bm25_doc (
    chunk_id TEXT PRIMARY KEY REFERENCES chunk(id),
    length   INTEGER NOT NULL
);
CREATE TABLE bm25_term (
    term_id  INTEGER PRIMARY KEY,
    term     TEXT NOT NULL UNIQUE,
    doc_freq INTEGER NOT NULL
);
CREATE TABLE bm25_posting (
    term_id   INTEGER NOT NULL REFERENCES bm25_term(term_id),
    chunk_id  TEXT NOT NULL REFERENCES chunk(id),
    term_freq INTEGER NOT NULL,
    PRIMARY KEY (term_id, chunk_id)
);
CREATE INDEX idx_posting_chunk ON bm25_posting (chunk_id);
CREATE TABLE bm25_stats (
    key   TEXT PRIMARY KEY,
    value REAL NOT NULL
);
"""


def _build_bundled_search(
    emb_path: Path, dst_path: Path, model_meta: dict
) -> dict:
    """Build the BSB+WSC scoped search index with BM25 recomputed over the subset.

    Reuses the exact stored float32 vectors (no re-encode — the bundled vectors are
    byte-identical to the full index) but rebuilds the BM25 tables so ``avgdl`` and
    document frequencies reflect ONLY the bundled corpus. The schema mirrors the
    full ``embeddings.sqlite`` so the app's retriever code is identical.
    """
    predicate = (
        "(c.resource_type='scripture' AND c.source=?) "
        "OR (c.resource_type='confession' AND c.source=?)"
    )
    chunks, vectors = _read_chunks_for(
        emb_path, predicate, (BUNDLED_TRANSLATION, BUNDLED_CONFESSION)
    )
    dst = _new_db(dst_path)
    try:
        dst.executescript(_SEARCH_SCHEMA)
        dst.executemany(
            "INSERT INTO chunk VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (c.id, c.resource_type, c.source, c.anchor, c.book, c.chapter,
                 c.verse_start, c.verse_end, c.key, c.text, c.text_checksum,
                 1 if c.truncated else 0)
                for c in chunks
            ],
        )
        dst.executemany(
            "INSERT INTO embedding VALUES (?,?,?)",
            [(c.id, EMBED_DIM, vectors[c.id]) for c in chunks],
        )

        bm25 = build_bm25(chunks)
        dst.executemany(
            "INSERT INTO bm25_doc VALUES (?,?)",
            sorted(bm25["doc_lengths"].items()),
        )
        term_ids: dict[str, int] = {}
        term_rows = []
        for tid, (term, df) in enumerate(bm25["terms"]):
            term_ids[term] = tid
            term_rows.append((tid, term, df))
        dst.executemany("INSERT INTO bm25_term VALUES (?,?,?)", term_rows)
        posting_rows = []
        for term, _df in bm25["terms"]:
            tid = term_ids[term]
            for chunk_id in sorted(bm25["postings"][term]):
                posting_rows.append((tid, chunk_id, bm25["postings"][term][chunk_id]))
        dst.executemany("INSERT INTO bm25_posting VALUES (?,?,?)", posting_rows)
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
        dst.executemany(
            "INSERT INTO meta VALUES (?,?)",
            [
                ("schema_version", "1"),
                ("resource_type", "embeddings"),
                ("scope", "bundled (bsb-scripture + wsc-confession)"),
                ("model_name", model_meta.get("model_name", "")),
                ("model_revision", model_meta.get("model_revision", "")),
                ("model_combined_sha256",
                 model_meta.get("model_combined_sha256", "")),
                ("embedding_dim", str(EMBED_DIM)),
                ("vector_format", "float32-le"),
                ("query_instruction",
                 model_meta.get(
                     "query_instruction",
                     "Represent this sentence for searching relevant passages: ")),
                ("bm25_tokenizer", "nfkc-casefold-alnum-no-stemming"),
                ("n_chunks", str(len(chunks))),
            ],
        )
        dst.commit()
        return {
            "n_chunks": len(chunks),
            "n_scripture": sum(1 for c in chunks if c.resource_type == "scripture"),
            "n_confession": sum(1 for c in chunks if c.resource_type == "confession"),
            "vocab_size": len(bm25["terms"]),
            "n_postings": len(posting_rows),
            "avgdl": bm25["avgdl"],
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


# --- Orchestration -----------------------------------------------------------
@dataclass
class PackFile:
    pack: str           # 'bundled' | 'on-demand'
    name: str           # file name under output/packs/
    role: str           # 'bibles'|'confessions'|'search'|'commentaries'|...
    bytes: int
    sha256: str
    contents: dict = field(default_factory=dict)


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


def package_corpus(output_dir: Path, packs_dir: Path) -> PackagingResult:
    """Produce the bundled + on-demand packs and the corpus manifest (in memory).

    Writes the pack ``.sqlite`` files under ``packs_dir`` (gitignored) and returns
    a ``PackagingResult`` carrying per-file SHA-256 + sizes, the manifest dict, and
    any size FLAGs. The manifest is written to disk by the caller (CLI).
    """
    packs_dir.mkdir(parents=True, exist_ok=True)
    emb_meta = _emb_model_meta(output_dir / "embeddings.sqlite")
    files: list[PackFile] = []

    def register(pack: str, name: str, role: str, contents: dict) -> None:
        p = packs_dir / name
        files.append(PackFile(
            pack=pack, name=name, role=role,
            bytes=p.stat().st_size, sha256=_sha256_file(p), contents=contents))

    # --- Bundled pack ---
    c = _filter_bibles(
        output_dir / "bibles.sqlite",
        packs_dir / "bundled_bibles.sqlite", [BUNDLED_TRANSLATION])
    register("bundled", "bundled_bibles.sqlite", "bibles", c)

    c = _filter_confessions(
        output_dir / "confessions.sqlite",
        packs_dir / "bundled_confessions.sqlite", [BUNDLED_CONFESSION])
    register("bundled", "bundled_confessions.sqlite", "confessions", c)

    c = _build_bundled_search(
        output_dir / "embeddings.sqlite",
        packs_dir / "bundled_search.sqlite", emb_meta)
    register("bundled", "bundled_search.sqlite", "search", c)

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
        ("search", "embeddings.sqlite"),
    ]:
        dst_name = f"ondemand_{fname}"
        _copy_whole_db(output_dir / fname, packs_dir / dst_name)
        register("on-demand", dst_name, role, {"copied_from": fname})

    bundled_bytes = sum(f.bytes for f in files if f.pack == "bundled")
    ondemand_bytes = sum(f.bytes for f in files if f.pack == "on-demand")

    flags: list[str] = []
    if bundled_bytes > BUNDLED_TARGET_MAX:
        flags.append(
            f"BUNDLED pack total {bundled_bytes:,} B exceeds the "
            f"~{BUNDLED_TARGET_MAX // (1024*1024)} MB target — review the split.")

    manifest = _build_manifest(output_dir, files, bundled_bytes, ondemand_bytes)
    return PackagingResult(
        files=files, manifest=manifest,
        bundled_bytes=bundled_bytes, ondemand_bytes=ondemand_bytes, flags=flags)


def _build_manifest(
    output_dir: Path, files: list[PackFile], bundled_bytes: int,
    ondemand_bytes: int,
) -> dict:
    acks = build_acknowledgements(output_dir)

    def pack_files(pack: str) -> tuple[int, list]:
        pf = [f for f in files if f.pack == pack]
        return sum(f.bytes for f in pf), [
            OrderedDict([
                ("name", f.name), ("role", f.role),
                ("bytes", f.bytes), ("sha256", f.sha256),
                ("contents", f.contents),
            ])
            for f in sorted(pf, key=lambda x: x.name)
        ]

    bundled_total, bundled_files = pack_files("bundled")
    ondemand_total, ondemand_files = pack_files("on-demand")

    return OrderedDict([
        ("corpus_version", CORPUS_VERSION_PLACEHOLDER),
        ("ship_ready", False),
        ("note",
         "CANDIDATE. The pipeline never marks a corpus version ship-ready — "
         "the architect's 23-point spot-check gates ship. Pack .sqlite files are "
         "gitignored (output/packs/); only this manifest is committed."),
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
                 "text, cross-references, and the full embeddings index."),
                ("delivery", "first-launch-download"),
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
