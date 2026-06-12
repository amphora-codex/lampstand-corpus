"""P6 embeddings + BM25 tests.

Two layers:
  * Pure-logic tests (tokenizer, BM25 builder, chunk-id stability, pericope
    windowing) — always run, no torch / no built DBs needed.
  * DB-backed tests (chunk counts from the built per-resource DBs) — skipped when
    output/*.sqlite aren't present.
  * Encoder determinism — skipped unless the [embeddings] extra + the pinned model
    snapshot are both present; it re-encodes a tiny fixed batch twice and requires
    a bit-for-bit byte match on CPU.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus import build_embeddings, embeddings

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"

_DBS = ["bibles", "commentaries", "confessions", "lexicons"]
_have_dbs = all((OUTPUT_DIR / f"{d}.sqlite").exists() for d in _DBS)
db_required = pytest.mark.skipif(
    not _have_dbs, reason="built per-resource DBs not present; run the P1-P4 builds")


# --- Tokenizer (deterministic, documented) -----------------------------------
def test_tokenizer_lowercases_and_splits():
    assert embeddings.tokenize("The LORD Is My Shepherd") == [
        "the", "lord", "is", "my", "shepherd"]


def test_tokenizer_keeps_internal_apostrophe():
    assert embeddings.tokenize("the Lord's word") == ["the", "lord's", "word"]


def test_tokenizer_strips_punctuation_and_normalizes():
    # NFKC normalizes ligatures/full-width; punctuation is a separator.
    assert embeddings.tokenize("faith, hope—love!") == ["faith", "hope", "love"]


def test_tokenizer_is_deterministic_across_calls():
    text = "Justification by faith alone, sola fide; Romans 3:28."
    assert embeddings.tokenize(text) == embeddings.tokenize(text)


# --- chunk id stability ------------------------------------------------------
def test_chunk_id_is_content_addressed_and_stable():
    c1 = embeddings._mk_chunk("confession", "wcf", "WCF 11.1", "Hello world", key="11.1")
    c2 = embeddings._mk_chunk("confession", "wcf", "WCF 11.1", "Hello world", key="11.1")
    assert c1.id == c2.id
    assert c1.text_checksum == c2.text_checksum
    # Different text -> different id.
    c3 = embeddings._mk_chunk("confession", "wcf", "WCF 11.1", "Other", key="11.1")
    assert c3.id != c1.id


def test_chunk_truncation_flag():
    long_text = "x " * 5000  # well over MAX_CHARS
    c = embeddings._mk_chunk("commentary", "henry", "GEN 1", long_text)
    assert c.truncated is True
    assert len(c.text) == embeddings.MAX_CHARS


# --- pericope windowing ------------------------------------------------------
def test_window_chapter_groups_by_target():
    verses = [(v, v, f"v{v}") for v in range(1, 21)]  # 20 verses
    wins = list(embeddings._window_chapter(verses))
    # 20 verses, target 10 -> two 10-verse windows.
    assert wins[0][0] == 1 and wins[0][1] == 10
    assert wins[1][0] == 11 and wins[1][1] == 20


def test_window_chapter_folds_short_tail():
    verses = [(v, v, f"v{v}") for v in range(1, 13)]  # 12 verses
    wins = list(embeddings._window_chapter(verses))
    # 12 verses, target 10, tail of 2 (< MIN_TAIL) folds into the first window.
    assert len(wins) == 1
    assert wins[0][0] == 1 and wins[0][1] == 12


# --- BM25 builder (deterministic) --------------------------------------------
def _chunk(cid_text: str, text: str) -> embeddings.Chunk:
    return embeddings._mk_chunk("scripture", "bsb", cid_text, text)


def test_bm25_build_counts_terms_and_postings():
    chunks = [
        _chunk("a", "faith faith hope"),
        _chunk("b", "hope love"),
    ]
    idx = build_embeddings.build_bm25(chunks)
    terms = dict(idx["terms"])  # term -> doc_freq
    assert terms["faith"] == 1   # only in chunk a
    assert terms["hope"] == 2    # in both
    assert terms["love"] == 1
    # term frequency: 'faith' appears twice in chunk a.
    a_id = chunks[0].id
    assert idx["postings"]["faith"][a_id] == 2
    assert idx["doc_lengths"][a_id] == 3


def test_bm25_build_is_deterministic():
    chunks = [_chunk("a", "the lord is my shepherd"),
              _chunk("b", "i shall not want")]
    a = build_embeddings.build_bm25(chunks)
    b = build_embeddings.build_bm25(chunks)
    assert a["terms"] == b["terms"]
    assert a["avgdl"] == b["avgdl"]


# --- DB-backed chunk counts --------------------------------------------------
@db_required
def test_scripture_chunks_cover_four_translations():
    chunks, skipped = embeddings.scripture_chunks(OUTPUT_DIR / "bibles.sqlite")
    sources = {c.source for c in chunks}
    assert sources == {"bsb", "kjv", "asv", "web"}
    # Every scripture chunk anchors to a real book + chapter + verse window.
    for c in chunks[:50]:
        assert c.book is not None and c.chapter is not None
        assert c.verse_start is not None and c.verse_end >= c.verse_start


@db_required
def test_commentary_chunks_match_db_rows():
    chunks, skipped = embeddings.commentary_chunks(
        OUTPUT_DIR / "commentaries.sqlite")
    conn = sqlite3.connect(OUTPUT_DIR / "commentaries.sqlite")
    n_nonempty = conn.execute(
        "SELECT count(*) FROM comment WHERE trim(text) <> ''").fetchone()[0]
    conn.close()
    assert len(chunks) == n_nonempty
    assert len(chunks) + len(skipped) >= n_nonempty


@db_required
def test_confession_chunks_match_db_rows():
    chunks, skipped = embeddings.confession_chunks(
        OUTPUT_DIR / "confessions.sqlite")
    conn = sqlite3.connect(OUTPUT_DIR / "confessions.sqlite")
    n = conn.execute("SELECT count(*) FROM section WHERE trim(text) <> ''").fetchone()[0]
    conn.close()
    assert len(chunks) == n


@db_required
def test_lexicon_chunks_skip_only_english_empty_entries():
    chunks, skipped = embeddings.lexicon_chunks(OUTPUT_DIR / "lexicons.sqlite")
    # Every embedded lexicon chunk has English-bearing text beyond a bare head.
    assert all(c.text.strip() for c in chunks)
    # Skips are flagged with a reason, never silent.
    assert all("lemma-only" in s for s in skipped)


@db_required
def test_extract_all_ids_are_unique():
    ec = embeddings.extract_all(OUTPUT_DIR)
    ids = [c.id for c in ec.chunks]
    assert len(ids) == len(set(ids))
    assert ec.by_type()  # non-empty


# --- Encoder determinism (heavy; opt-in) -------------------------------------
def _model_present() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    from lampstand_corpus.embeddings import MODEL_NAME, MODEL_REVISION
    from lampstand_corpus.encode import MODEL_CACHE
    snap = (MODEL_CACHE / f"models--{MODEL_NAME.replace('/', '--')}"
            / "snapshots" / MODEL_REVISION)
    return snap.exists()


@pytest.mark.skipif(
    not _model_present(),
    reason="[embeddings] extra + pinned model snapshot not present")
def test_encoder_is_bit_for_bit_deterministic_on_cpu():
    from lampstand_corpus.encode import encode_chunks
    sample = [
        embeddings._mk_chunk("scripture", "bsb", "PSA 23:1", "The LORD is my shepherd"),
        embeddings._mk_chunk("confession", "wcf", "WCF 11.1",
                             "Justification is an act of God's free grace."),
    ]
    a = encode_chunks(sample, device="cpu")
    b = encode_chunks(sample, device="cpu")
    assert a.shape == (2, embeddings.EMBED_DIM)
    assert a.tobytes() == b.tobytes()


# --- Retrieval smoke test against the built artifact (tolerant to jitter) -----
_EMB_DB = OUTPUT_DIR / "embeddings.sqlite"


@pytest.mark.skipif(
    not (_EMB_DB.exists() and _model_present()),
    reason="embeddings.sqlite + model snapshot not present; run build-embeddings")
def test_smoke_shepherd_returns_psalm_23():
    from lampstand_corpus.encode import load_model
    from lampstand_corpus.retrieve import encode_query, load_matrix, topk

    matrix, meta = load_matrix(_EMB_DB)
    model = load_model(device="cpu")
    qvec = encode_query(model, "the LORD is my shepherd")
    hits = topk(matrix, meta, qvec, k=10)
    # Tolerant: Psalm 23 must appear somewhere in top-k, not necessarily at rank 1.
    assert any(h.book == "PSA" and h.chapter == 23 for h in hits)


@pytest.mark.skipif(
    not (_EMB_DB.exists() and _model_present()),
    reason="embeddings.sqlite + model snapshot not present; run build-embeddings")
def test_smoke_justification_returns_romans_or_galatians_or_confession():
    from lampstand_corpus.encode import load_model
    from lampstand_corpus.retrieve import encode_query, load_matrix, topk

    matrix, meta = load_matrix(_EMB_DB)
    model = load_model(device="cpu")
    qvec = encode_query(model, "justification by faith")
    hits = topk(matrix, meta, qvec, k=10)
    assert any(
        h.book in {"ROM", "GAL"}
        or h.resource_type == "confession"
        or "justif" in h.text.lower()
        for h in hits
    )
