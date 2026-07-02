"""P7 packaging tests — bundled/on-demand split, BM25 recompute, determinism.

Builds tiny synthetic per-resource DBs (so the suite stays fast and independent
of the full built artifacts) and exercises ``package_corpus``:

  * the bundled pack carries ONLY BSB + WSC + a BSB/WSC-scoped search index;
  * the on-demand packs carry everything else, with no overlap and no gaps;
  * the bundled search index reuses the exact stored vectors but recomputes BM25
    statistics over the bundled subset (N == bundled chunk count, not the global);
  * the whole thing is bit-for-bit reproducible across two runs;
  * the corpus manifest has the right shape + a rolled-up acknowledgements list.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from lampstand_corpus.embeddings import EMBED_DIM
from lampstand_corpus.pack_codec import decode_postings, quantize_int8
from lampstand_corpus.package import (
    CORPUS_VERSION_PLACEHOLDER,
    VECTOR_FORMAT_FP32,
    VECTOR_FORMAT_INT8,
    build_acknowledgements,
    package_corpus,
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _vec(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype("<f4")
    v /= np.linalg.norm(v)
    return v.tobytes()


def _make_bibles(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE translation(id TEXT PRIMARY KEY, name TEXT, version TEXT,"
        " license TEXT, source_url TEXT, retrieved TEXT, checksum TEXT);"
        "CREATE TABLE book(id TEXT PRIMARY KEY, name TEXT, ord INTEGER);"
        "CREATE TABLE verse(translation TEXT, book TEXT, chapter INTEGER,"
        " verse_start INTEGER, verse_end INTEGER, text TEXT,"
        " PRIMARY KEY(translation, book, chapter, verse_start));"
    )
    conn.execute("INSERT INTO meta VALUES('schema_version','1')")
    conn.execute("INSERT INTO book VALUES('GEN','Genesis',0)")
    for tid in ("asv", "bsb", "kjv", "web"):
        conn.execute(
            "INSERT INTO translation VALUES(?,?,?,?,?,?,?)",
            (tid, tid.upper(), "v1", "Public domain", "http://x", "2026-06-10", "ck"),
        )
        conn.execute(
            "INSERT INTO verse VALUES(?,?,?,?,?,?)",
            (tid, "GEN", 1, 1, 1, f"In the beginning ({tid})."),
        )
    conn.commit()
    conn.close()


def _make_confessions(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE document(id TEXT PRIMARY KEY, name TEXT, shortcode TEXT,"
        " version TEXT, license TEXT, source_url TEXT, retrieved TEXT, checksum TEXT);"
        "CREATE TABLE section(document TEXT, key TEXT, ord INTEGER, title TEXT,"
        " text TEXT, PRIMARY KEY(document, key));"
    )
    conn.execute("INSERT INTO meta VALUES('schema_version','1')")
    for did, name in (("wsc", "Westminster Shorter Catechism"),
                      ("wcf", "Westminster Confession of Faith")):
        conn.execute(
            "INSERT INTO document VALUES(?,?,?,?,?,?,?,?)",
            (did, name, did.upper(), "v1", "Public domain", "http://x",
             "2026-06-10", "ck"),
        )
        conn.execute(
            "INSERT INTO section VALUES(?,?,?,?,?)",
            (did, "1", 0, "Title", f"Body of {did}."),
        )
    conn.commit()
    conn.close()


def _make_simple_db(path: Path, table: str, rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES('schema_version','1')")
    conn.execute(f"CREATE TABLE {table}(id TEXT PRIMARY KEY, payload TEXT)")
    conn.executemany(f"INSERT INTO {table} VALUES(?,?)", rows)
    conn.commit()
    conn.close()


def _make_commentaries(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES('schema_version','1')")
    conn.execute(
        "CREATE TABLE commentator(id TEXT PRIMARY KEY, name TEXT, shortcode TEXT,"
        " author TEXT, work TEXT, version TEXT, license TEXT, retrieved TEXT,"
        " source_urls TEXT)")
    conn.execute(
        "INSERT INTO commentator VALUES('henry','Matthew Henry','MH','Henry',"
        "'Whole Bible','v1','Public domain (CCEL)','2026-06-10','[]')")
    conn.execute("CREATE TABLE comment(id TEXT PRIMARY KEY, text TEXT)")
    conn.execute("INSERT INTO comment VALUES('c1','A note.')")
    conn.commit()
    conn.close()


def _make_lexicons(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES('schema_version','1')")
    conn.execute(
        "CREATE TABLE lexicon(id TEXT PRIMARY KEY, name TEXT, language TEXT,"
        " lexicon TEXT, version TEXT, license TEXT, text_license TEXT,"
        " source_url TEXT, retrieved TEXT, checksum TEXT)")
    conn.execute(
        "INSERT INTO lexicon VALUES('strongs-greek',\"Strong's Greek\",'greek',"
        "'sg','v1','CC0','PD','http://x','2026-06-10','ck')")
    conn.execute(
        "CREATE TABLE tagged_source(id TEXT PRIMARY KEY, name TEXT, language TEXT,"
        " version TEXT, license TEXT, attribution TEXT, source_url TEXT,"
        " retrieved TEXT, checksum TEXT)")
    conn.execute(
        "INSERT INTO tagged_source VALUES('tagnt','TAGNT','greek','v1',"
        "'CC-BY-4.0','STEP Bible','http://x','2026-06-10','ck')")
    conn.commit()
    conn.close()


def _make_crossrefs(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES('schema_version','1')")
    conn.execute(
        "CREATE TABLE source(id TEXT PRIMARY KEY, name TEXT, license TEXT,"
        " attribution TEXT, source_url TEXT, retrieved TEXT, checksum TEXT,"
        " header TEXT)")
    conn.execute(
        "INSERT INTO source VALUES('tsk','TSK','CC-BY 4.0','courtesy openbible',"
        "'http://x','2026-06-10','ck','hdr')")
    conn.execute("CREATE TABLE crossref(id TEXT PRIMARY KEY, payload TEXT)")
    conn.execute("INSERT INTO crossref VALUES('x1','GEN 1:1->JHN 1:1')")
    conn.commit()
    conn.close()


def _make_embeddings(path: Path) -> None:
    """Tiny embeddings DB: bsb + asv scripture, wsc + wcf confession, 1 lexicon."""
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE chunk(id TEXT PRIMARY KEY, resource_type TEXT, source TEXT,"
        " anchor TEXT, book TEXT, chapter INTEGER, verse_start INTEGER,"
        " verse_end INTEGER, key TEXT, text TEXT, text_checksum TEXT,"
        " truncated INTEGER);"
        "CREATE TABLE embedding(chunk_id TEXT PRIMARY KEY, dim INTEGER, vector BLOB);"
    )
    rows = [
        ("scr_a", "scripture", "bsb", "bsb:GEN 1:1", "GEN", 1, 1, 1, None,
         "in the beginning bsb light", "h1", 0),
        ("scr_b", "scripture", "asv", "asv:GEN 1:1", "GEN", 1, 1, 1, None,
         "in the beginning asv light", "h2", 0),
        ("con_w", "confession", "wsc", "WSC 1", None, None, None, None, "1",
         "chief end of man glorify", "h3", 0),
        ("con_c", "confession", "wcf", "WCF 1.1", None, None, None, None, "1.1",
         "of the holy scripture light", "h4", 0),
        ("lex_g", "lexicon", "strongs-greek", "strongs-greek:G25", None, None,
         None, None, "G25", "agape love charity", "h5", 0),
    ]
    conn.executemany(
        "INSERT INTO chunk VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    for i, r in enumerate(rows):
        conn.execute("INSERT INTO embedding VALUES(?,?,?)",
                     (r[0], EMBED_DIM, _vec(i)))
    conn.executemany(
        "INSERT INTO meta VALUES(?,?)",
        [("model_name", "BAAI/bge-small-en-v1.5"),
         ("model_revision", "abc123"),
         ("model_combined_sha256", "deadbeef"),
         ("query_instruction", "Represent this sentence: ")],
    )
    conn.commit()
    conn.close()


def _build_fixture(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _make_bibles(output_dir / "bibles.sqlite")
    _make_confessions(output_dir / "confessions.sqlite")
    _make_commentaries(output_dir / "commentaries.sqlite")
    _make_lexicons(output_dir / "lexicons.sqlite")
    _make_crossrefs(output_dir / "crossrefs.sqlite")
    _make_embeddings(output_dir / "embeddings.sqlite")


def test_bundled_pack_is_bsb_and_wsc_only(tmp_path):
    out = tmp_path / "output"
    _build_fixture(out)
    result = package_corpus(out, out / "packs")

    bb = sqlite3.connect(out / "packs" / "bundled_bibles.sqlite")
    assert [r[0] for r in bb.execute("SELECT id FROM translation")] == ["bsb"]
    assert bb.execute("SELECT count(*) FROM verse").fetchone()[0] == 1
    bb.close()

    bc = sqlite3.connect(out / "packs" / "bundled_confessions.sqlite")
    assert [r[0] for r in bc.execute("SELECT id FROM document")] == ["wsc"]
    bc.close()

    # No size flag at fixture scale.
    assert result.flags == []


def test_ondemand_has_the_rest_with_no_overlap(tmp_path):
    out = tmp_path / "output"
    _build_fixture(out)
    package_corpus(out, out / "packs")

    ob = sqlite3.connect(out / "packs" / "ondemand_bibles.sqlite")
    assert sorted(r[0] for r in ob.execute("SELECT id FROM translation")) == [
        "asv", "kjv", "web"]
    ob.close()

    oc = sqlite3.connect(out / "packs" / "ondemand_confessions.sqlite")
    assert "wsc" not in {r[0] for r in oc.execute("SELECT id FROM document")}
    assert "wcf" in {r[0] for r in oc.execute("SELECT id FROM document")}
    oc.close()

    # Whole-DB copies present for the heavy resources; the v2 split replaces
    # the former ondemand_embeddings.sqlite byte-copy.
    for name in ("ondemand_commentaries.sqlite", "ondemand_lexicons.sqlite",
                 "ondemand_crossrefs.sqlite", "ondemand_search.sqlite",
                 "ondemand_vectors.sqlite"):
        assert (out / "packs" / name).exists()
    assert not (out / "packs" / "ondemand_embeddings.sqlite").exists()


def test_stale_v1_embeddings_pack_is_removed(tmp_path):
    out = tmp_path / "output"
    _build_fixture(out)
    packs = out / "packs"
    packs.mkdir(parents=True)
    (packs / "ondemand_embeddings.sqlite").write_bytes(b"stale v1 pack")
    package_corpus(out, packs)
    assert not (packs / "ondemand_embeddings.sqlite").exists()


def test_bundled_search_v2_scopes_quantizes_and_recomputes_bm25(tmp_path):
    out = tmp_path / "output"
    _build_fixture(out)
    package_corpus(out, out / "packs")

    full = sqlite3.connect(out / "embeddings.sqlite")
    bun = sqlite3.connect(out / "packs" / "bundled_search.sqlite")

    # Only bsb-scripture + wsc-confession survive into the bundled index.
    sources = sorted(r[0] for r in bun.execute(
        "SELECT DISTINCT source FROM chunk"))
    assert sources == ["bsb", "wsc"]

    # Embedded vectors are the int8 quantization of the EXACT stored float32
    # bytes (quantized at pack time, never re-encoded).
    for sid in ("scr_a", "con_w"):
        int_id, = bun.execute(
            "SELECT id FROM chunk WHERE string_id=?", (sid,)).fetchone()
        vb, scale = bun.execute(
            "SELECT vector, scale FROM embedding WHERE chunk_id=?",
            (int_id,)).fetchone()
        vf = full.execute(
            "SELECT vector FROM embedding WHERE chunk_id=?", (sid,)).fetchone()[0]
        exp_blob, exp_scale = quantize_int8(np.frombuffer(vf, dtype="<f4"))
        assert vb == exp_blob and scale == exp_scale

    # BM25 N reflects ONLY the bundled subset (2 docs), not the global 5.
    n_docs = bun.execute(
        "SELECT value FROM bm25_stats WHERE key='n_docs'").fetchone()[0]
    assert n_docs == 2.0
    # "light" appears in the bsb scripture chunk; doc_freq must be <= bundled N.
    df = bun.execute(
        "SELECT doc_freq FROM bm25_term WHERE term='light'").fetchone()
    assert df is not None and df[0] <= 2
    full.close()
    bun.close()


def test_ondemand_search_pack_v2_contract(tmp_path):
    """Int ids ascend with string ids, postings decode, doc_len is folded in."""
    out = tmp_path / "output"
    _build_fixture(out)
    package_corpus(out, out / "packs")

    sp = sqlite3.connect(out / "packs" / "ondemand_search.sqlite")
    rows = sp.execute(
        "SELECT id, string_id, text, doc_len FROM chunk ORDER BY id").fetchall()
    # 1-based int ids assigned by ascending string id.
    assert [r[0] for r in rows] == list(range(1, 6))
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
    # Display text included (the data-driven decision) + doc_len from BM25.
    by_sid = {r[1]: r for r in rows}
    assert by_sid["con_w"][2] == "chief end of man glorify"
    assert by_sid["con_w"][3] == 5
    # Meta declares the v2 contract.
    meta = dict(sp.execute("SELECT key, value FROM meta"))
    assert meta["format"] == "search-pack-v2"
    assert meta["posting_format"] == "uvarint-gap-tf-v1"
    assert meta["text_included"] == "1"

    # Posting blobs decode to (int id, tf) pairs consistent with the corpus:
    # "light" appears in scr_a, scr_b, con_c — three distinct chunks.
    df, blob = sp.execute(
        "SELECT doc_freq, postings FROM bm25_term WHERE term='light'").fetchone()
    postings = decode_postings(blob)
    assert len(postings) == df == 3
    ids_for_light = {
        by_sid[sid][0] for sid in ("scr_a", "scr_b", "con_c")}
    assert {cid for cid, _tf in postings} == ids_for_light
    assert all(tf == 1 for _cid, tf in postings)
    # No per-posting row table exists in v2.
    tables = {r[0] for r in sp.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "bm25_posting" not in tables and "bm25_doc" not in tables
    sp.close()


def test_vectors_pack_shares_int_ids_and_round_trips(tmp_path):
    out = tmp_path / "output"
    _build_fixture(out)
    package_corpus(out, out / "packs")

    sp = sqlite3.connect(out / "packs" / "ondemand_search.sqlite")
    vp = sqlite3.connect(out / "packs" / "ondemand_vectors.sqlite")
    full = sqlite3.connect(out / "embeddings.sqlite")

    meta = dict(vp.execute("SELECT key, value FROM meta"))
    assert meta["format"] == "vectors-pack-v2"
    assert meta["vector_format"] == VECTOR_FORMAT_INT8

    # Every search-pack chunk has a vector row under the SAME int id, and the
    # int8 dequantization is within half a quantization step of the fp32 truth.
    for sid in ("scr_a", "lex_g"):
        int_id, = sp.execute(
            "SELECT id FROM chunk WHERE string_id=?", (sid,)).fetchone()
        blob, scale = vp.execute(
            "SELECT vector, scale FROM embedding WHERE chunk_id=?",
            (int_id,)).fetchone()
        truth = np.frombuffer(full.execute(
            "SELECT vector FROM embedding WHERE chunk_id=?",
            (sid,)).fetchone()[0], dtype="<f4")
        back = np.frombuffer(blob, dtype=np.int8).astype(np.float32) * scale
        assert len(blob) == EMBED_DIM
        assert float(np.max(np.abs(back - truth))) <= scale / 2 + 1e-7
    sp.close()
    vp.close()
    full.close()


def test_fp32_flag_keeps_exact_vector_bytes(tmp_path):
    out = tmp_path / "output"
    _build_fixture(out)
    package_corpus(out, out / "packs", vector_format=VECTOR_FORMAT_FP32)

    sp = sqlite3.connect(out / "packs" / "ondemand_search.sqlite")
    vp = sqlite3.connect(out / "packs" / "ondemand_vectors.sqlite")
    full = sqlite3.connect(out / "embeddings.sqlite")
    meta = dict(vp.execute("SELECT key, value FROM meta"))
    assert meta["vector_format"] == VECTOR_FORMAT_FP32
    int_id, = sp.execute(
        "SELECT id FROM chunk WHERE string_id='scr_a'").fetchone()
    blob, scale = vp.execute(
        "SELECT vector, scale FROM embedding WHERE chunk_id=?",
        (int_id,)).fetchone()
    truth = full.execute(
        "SELECT vector FROM embedding WHERE chunk_id='scr_a'").fetchone()[0]
    assert blob == truth and scale == 1.0
    sp.close()
    vp.close()
    full.close()


def test_packaging_is_deterministic(tmp_path):
    out = tmp_path / "output"
    _build_fixture(out)
    package_corpus(out, out / "packs_a")
    package_corpus(out, out / "packs_b")
    for f in sorted((out / "packs_a").glob("*.sqlite")):
        assert _sha(f) == _sha(out / "packs_b" / f.name), f.name


def test_manifest_shape_and_acknowledgements(tmp_path):
    out = tmp_path / "output"
    _build_fixture(out)
    result = package_corpus(out, out / "packs")
    m = result.manifest

    assert m["corpus_version"] == CORPUS_VERSION_PLACEHOLDER
    assert m["ship_ready"] is False
    assert set(m["packs"]) == {"bundled", "on_demand"}
    # Per-file sha256 + bytes present.
    for f in m["packs"]["bundled"]["files"]:
        assert len(f["sha256"]) == 64 and f["bytes"] > 0

    acks = build_acknowledgements(out)
    rtypes = {a["resource_type"] for a in acks}
    assert {"scripture", "confession", "commentary", "lexicon", "tagged-text",
            "crossref", "embedding-model"} <= rtypes
    # CC-BY attribution rolled up where present.
    tagnt = next(a for a in acks if a["id"] == "tagnt")
    assert tagnt["attribution"] == "STEP Bible"

    # Manifest serializes cleanly to JSON.
    json.dumps(m)
