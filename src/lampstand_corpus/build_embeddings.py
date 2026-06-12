"""Build ``embeddings.sqlite`` — dense vectors + a BM25 keyword index.

A single, deliberately simple SQLite file the iOS app can read directly (no exotic
index formats):

  meta            build identity (model, dim, tokenizer, counts) — no timestamps
  chunk           one row per retrieval unit (id, resource_type, source, anchor,
                  verse anchor columns, key, text, text_checksum)
  embedding       chunk_id -> float32 vector blob (little-endian, dim columns)
  bm25_doc        chunk_id -> document length (token count) for BM25 scoring
  bm25_term       term -> term_id + document frequency
  bm25_posting    term_id -> chunk_id + term frequency in that chunk
  bm25_stats      corpus-level BM25 constants (N, avgdl, k1, b)

Output is deterministic: rows inserted in fixed order, vectors are exact float32
bytes from the encoder, term ids are assigned by sorted term order, and no
wall-clock or autoincrement-dependent value leaks in.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from .embeddings import EMBED_DIM, Chunk, tokenize

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE chunk (
    id            TEXT PRIMARY KEY,
    resource_type TEXT NOT NULL,   -- scripture|commentary|confession|lexicon
    source        TEXT NOT NULL,   -- bsb|henry|wcf|strongs-greek|...
    anchor        TEXT NOT NULL,   -- resolvable: "GEN 1:1-5","WCF 11.1","strongs-greek:G25"
    book          TEXT,            -- canonical book id when Scripture-anchored
    chapter       INTEGER,
    verse_start   INTEGER,
    verse_end     INTEGER,
    key           TEXT,            -- lexicon Strong's / confession section key
    text          TEXT NOT NULL,
    text_checksum TEXT NOT NULL,   -- SHA-256 of text (chunk-stability audit)
    truncated     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_chunk_resource ON chunk (resource_type);
CREATE INDEX idx_chunk_source ON chunk (source);
CREATE INDEX idx_chunk_ref ON chunk (book, chapter, verse_start);

CREATE TABLE embedding (
    chunk_id TEXT PRIMARY KEY REFERENCES chunk(id),
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL        -- dim * float32, little-endian
);

CREATE TABLE bm25_doc (
    chunk_id TEXT PRIMARY KEY REFERENCES chunk(id),
    length   INTEGER NOT NULL     -- token count (for BM25 length normalization)
);

CREATE TABLE bm25_term (
    term_id  INTEGER PRIMARY KEY,
    term     TEXT NOT NULL UNIQUE,
    doc_freq INTEGER NOT NULL     -- number of chunks containing the term
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

# BM25 free parameters (stored so the app uses identical constants).
BM25_K1 = 1.5
BM25_B = 0.75


def _connect(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _vector_blob(vec: np.ndarray) -> bytes:
    """Exact little-endian float32 bytes for one vector."""
    return np.ascontiguousarray(vec, dtype="<f4").tobytes()


def build_bm25(chunks: list[Chunk]) -> dict:
    """Compute the BM25 index over chunk texts (deterministic).

    Returns a dict of the tables to write:
      doc_lengths : {chunk_id: token_count}
      terms       : sorted [(term, doc_freq)]  (term_id == index, assigned later)
      postings    : {term: {chunk_id: term_freq}}
      avgdl, n_docs
    """
    doc_lengths: dict[str, int] = {}
    postings: dict[str, dict[str, int]] = defaultdict(dict)
    total_tokens = 0
    for c in chunks:
        toks = tokenize(c.text)
        doc_lengths[c.id] = len(toks)
        total_tokens += len(toks)
        tf: dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for term, freq in tf.items():
            postings[term][c.id] = freq
    n_docs = len(chunks)
    avgdl = (total_tokens / n_docs) if n_docs else 0.0
    # term_id assigned by sorted term for determinism + a stable, debuggable order.
    terms = sorted((term, len(docs)) for term, docs in postings.items())
    return {
        "doc_lengths": doc_lengths,
        "terms": terms,
        "postings": postings,
        "avgdl": avgdl,
        "n_docs": n_docs,
        "total_tokens": total_tokens,
    }


def write_embeddings(
    chunks: list[Chunk],
    vectors: np.ndarray,
    out_path: Path,
    *,
    model_provenance: dict,
    skipped: dict[str, list[str]] | None = None,
) -> dict:
    """Write the chunk table, vector blobs, and BM25 index to ``out_path``.

    ``vectors`` is an ``(len(chunks), EMBED_DIM)`` float32 array aligned to
    ``chunks`` by index. Returns BM25 stats for the report.
    """
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"vector/chunk count mismatch: {vectors.shape[0]} != {len(chunks)}")
    if chunks and vectors.shape[1] != EMBED_DIM:
        raise ValueError(f"unexpected embedding dim {vectors.shape[1]} != {EMBED_DIM}")

    conn = _connect(out_path)
    try:
        # Chunks + embeddings in the order given (already canonical).
        conn.executemany(
            "INSERT INTO chunk VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (c.id, c.resource_type, c.source, c.anchor, c.book, c.chapter,
                 c.verse_start, c.verse_end, c.key, c.text, c.text_checksum,
                 1 if c.truncated else 0)
                for c in chunks
            ],
        )
        conn.executemany(
            "INSERT INTO embedding VALUES (?,?,?)",
            [
                (c.id, EMBED_DIM, _vector_blob(vectors[i]))
                for i, c in enumerate(chunks)
            ],
        )

        bm25 = build_bm25(chunks)
        conn.executemany(
            "INSERT INTO bm25_doc VALUES (?,?)",
            sorted(bm25["doc_lengths"].items()),
        )
        term_ids: dict[str, int] = {}
        term_rows = []
        for tid, (term, df) in enumerate(bm25["terms"]):
            term_ids[term] = tid
            term_rows.append((tid, term, df))
        conn.executemany("INSERT INTO bm25_term VALUES (?,?,?)", term_rows)

        posting_rows = []
        for term, _df in bm25["terms"]:
            tid = term_ids[term]
            for chunk_id in sorted(bm25["postings"][term]):
                posting_rows.append((tid, chunk_id, bm25["postings"][term][chunk_id]))
        conn.executemany("INSERT INTO bm25_posting VALUES (?,?,?)", posting_rows)

        conn.executemany(
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

        n_truncated = sum(1 for c in chunks if c.truncated)
        n_skipped = sum(len(v) for v in (skipped or {}).values())
        meta_rows = [
            ("schema_version", "1"),
            ("resource_type", "embeddings"),
            ("model_name", model_provenance["name"]),
            ("model_revision", model_provenance["revision"]),
            ("model_combined_sha256", model_provenance["combined_sha256"]),
            ("embedding_dim", str(EMBED_DIM)),
            ("vector_format", "float32-le"),
            ("query_instruction",
             "Represent this sentence for searching relevant passages: "),
            ("bm25_tokenizer", "nfkc-casefold-alnum-no-stemming"),
            ("n_chunks", str(len(chunks))),
            ("n_truncated", str(n_truncated)),
            ("n_skipped", str(n_skipped)),
        ]
        conn.executemany("INSERT INTO meta VALUES (?,?)", meta_rows)
        conn.commit()
        return {
            "n_chunks": len(chunks),
            "vocab_size": len(bm25["terms"]),
            "n_postings": len(posting_rows),
            "avgdl": bm25["avgdl"],
            "n_truncated": n_truncated,
        }
    finally:
        conn.close()
