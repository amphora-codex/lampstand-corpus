"""embeddings.sqlite build determinism + schema integrity.

These tests don't need torch or the encoder: they feed fixed synthetic vectors
through ``write_embeddings`` and assert the SQLite output is bit-for-bit identical
across two runs (CLAUDE.md pipeline rule 6) and that the BM25 tables are coherent.
"""

from __future__ import annotations

import hashlib
import sqlite3

import numpy as np

from lampstand_corpus import embeddings
from lampstand_corpus.build_embeddings import write_embeddings

_PROV = {
    "name": embeddings.MODEL_NAME,
    "revision": embeddings.MODEL_REVISION,
    "combined_sha256": "0" * 64,
}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_chunks() -> list[embeddings.Chunk]:
    return [
        embeddings._mk_chunk("scripture", "bsb", "PSA 23:1-3",
                             "The LORD is my shepherd I shall not want",
                             book="PSA", chapter=23, verse_start=1, verse_end=3),
        embeddings._mk_chunk("confession", "wcf", "WCF 11.1",
                             "Justification is an act of God's free grace", key="11.1"),
        embeddings._mk_chunk("lexicon", "strongs-greek", "strongs-greek:G25",
                             "agapao to love", key="G25"),
    ]


def _fixed_vectors(n: int) -> np.ndarray:
    # Deterministic, normalized vectors — no RNG seed dependence.
    rows = []
    for i in range(n):
        v = np.array([((i + j) % 7) + 1 for j in range(embeddings.EMBED_DIM)],
                     dtype=np.float32)
        v /= np.linalg.norm(v)
        rows.append(v)
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)


def test_write_embeddings_is_bit_for_bit_deterministic(tmp_path):
    chunks = _sample_chunks()
    vecs = _fixed_vectors(len(chunks))
    a = tmp_path / "a.sqlite"
    b = tmp_path / "b.sqlite"
    write_embeddings(chunks, vecs, a, model_provenance=_PROV)
    write_embeddings(chunks, vecs, b, model_provenance=_PROV)
    assert _sha(a) == _sha(b)


def test_write_embeddings_roundtrips_vectors(tmp_path):
    chunks = _sample_chunks()
    vecs = _fixed_vectors(len(chunks))
    out = tmp_path / "e.sqlite"
    write_embeddings(chunks, vecs, out, model_provenance=_PROV)
    conn = sqlite3.connect(out)
    try:
        for i, c in enumerate(chunks):
            blob = conn.execute(
                "SELECT vector FROM embedding WHERE chunk_id=?", (c.id,)).fetchone()[0]
            got = np.frombuffer(blob, dtype="<f4")
            assert got.shape == (embeddings.EMBED_DIM,)
            assert np.array_equal(got, vecs[i])
        # meta records the dim + model identity.
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        assert meta["embedding_dim"] == str(embeddings.EMBED_DIM)
        assert meta["model_revision"] == embeddings.MODEL_REVISION
    finally:
        conn.close()


def test_bm25_tables_are_coherent(tmp_path):
    chunks = _sample_chunks()
    vecs = _fixed_vectors(len(chunks))
    out = tmp_path / "e.sqlite"
    stats = write_embeddings(chunks, vecs, out, model_provenance=_PROV)
    conn = sqlite3.connect(out)
    try:
        vocab = conn.execute("SELECT count(*) FROM bm25_term").fetchone()[0]
        assert vocab == stats["vocab_size"]
        # doc_freq per term equals the number of postings for that term.
        rows = conn.execute(
            "SELECT t.doc_freq, count(p.chunk_id) "
            "FROM bm25_term t JOIN bm25_posting p ON p.term_id = t.term_id "
            "GROUP BY t.term_id").fetchall()
        assert all(df == npost for df, npost in rows)
        # every chunk has a doc-length row.
        n_docs = conn.execute("SELECT count(*) FROM bm25_doc").fetchone()[0]
        assert n_docs == len(chunks)
        stats_tbl = dict(conn.execute("SELECT key, value FROM bm25_stats").fetchall())
        assert stats_tbl["n_docs"] == float(len(chunks))
        assert stats_tbl["k1"] > 0 and stats_tbl["b"] > 0
    finally:
        conn.close()
