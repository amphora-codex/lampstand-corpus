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


# --- Incremental re-encode (no torch: encoder is stubbed) --------------------
def _write_prior(tmp_path, chunks, vecs, *, revision):
    prov = dict(_PROV, revision=revision)
    db = tmp_path / "prior.sqlite"
    write_embeddings(chunks, vecs, db, model_provenance=prov)
    return db


def test_incremental_reuses_unchanged_and_encodes_changed(tmp_path, monkeypatch):
    from lampstand_corpus import encode

    prior_chunks = _sample_chunks()
    prior_vecs = _fixed_vectors(len(prior_chunks))
    db = _write_prior(tmp_path, prior_chunks, prior_vecs,
                      revision=embeddings.MODEL_REVISION)

    # New set: keep chunk 0 unchanged, change chunk 1's text (new id), add a new
    # chunk, and drop chunk 2 entirely.
    unchanged = prior_chunks[0]
    changed = embeddings._mk_chunk("confession", "wcf", "WCF 11.1",
                                   "Justification is an act of God's free grace, NEW",
                                   key="11.1")
    added = embeddings._mk_chunk("lexicon", "strongs-greek", "strongs-greek:G26",
                                 "agape love", key="G26")
    new_chunks = [unchanged, changed, added]

    encoded_texts = []

    def fake_encode_texts(model, texts, **kw):
        encoded_texts.extend(texts)
        return np.ascontiguousarray(
            np.vstack([np.full(embeddings.EMBED_DIM, float(i + 1), dtype=np.float32)
                       for i in range(len(texts))]))

    monkeypatch.setattr(encode, "load_model", lambda device="cpu": object())
    monkeypatch.setattr(encode, "encode_texts", fake_encode_texts)

    res = encode.encode_chunks_incremental(new_chunks, db, device="cpu")
    assert res.n_total == 3
    assert res.n_reused == 1            # only chunk 0 (unchanged id)
    assert res.n_encoded == 2           # changed + added
    assert res.n_dropped == 2           # chunks 1(old text) and 2 dropped
    assert not res.full_reencode
    # The reused vector is byte-identical to the prior one.
    assert res.vectors[0].tobytes() == prior_vecs[0].tobytes()
    # Only the changed + added texts went to the encoder.
    assert set(encoded_texts) == {changed.text, added.text}


def test_incremental_falls_back_to_full_on_model_mismatch(tmp_path, monkeypatch):
    from lampstand_corpus import encode

    chunks = _sample_chunks()
    vecs = _fixed_vectors(len(chunks))
    db = _write_prior(tmp_path, chunks, vecs, revision="OLD-REVISION-HASH")

    def fake_encode_texts(model, texts, **kw):
        return np.ascontiguousarray(
            np.ones((len(texts), embeddings.EMBED_DIM), dtype=np.float32))

    monkeypatch.setattr(encode, "load_model", lambda device="cpu": object())
    monkeypatch.setattr(encode, "encode_texts", fake_encode_texts)

    res = encode.encode_chunks_incremental(chunks, db, device="cpu")
    assert res.full_reencode is True
    assert res.n_reused == 0
    assert res.n_encoded == len(chunks)
    assert any("revision" in n for n in res.notes)


def test_incremental_full_encode_when_no_prior(tmp_path, monkeypatch):
    from lampstand_corpus import encode

    chunks = _sample_chunks()

    def fake_encode_texts(model, texts, **kw):
        return np.ascontiguousarray(
            np.ones((len(texts), embeddings.EMBED_DIM), dtype=np.float32))

    monkeypatch.setattr(encode, "load_model", lambda device="cpu": object())
    monkeypatch.setattr(encode, "encode_texts", fake_encode_texts)

    res = encode.encode_chunks_incremental(chunks, tmp_path / "missing.sqlite")
    assert res.full_reencode is True
    assert res.n_reused == 0 and res.n_encoded == len(chunks)
