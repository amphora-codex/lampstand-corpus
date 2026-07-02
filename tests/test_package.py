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
        " text TEXT, proof_texts TEXT, PRIMARY KEY(document, key));"
    )
    conn.execute("INSERT INTO meta VALUES('schema_version','1')")
    for did, name in (("wsc", "Westminster Shorter Catechism"),
                      ("wcf", "Westminster Confession of Faith"),
                      ("heidelberg", "Heidelberg Catechism")):
        conn.execute(
            "INSERT INTO document VALUES(?,?,?,?,?,?,?,?)",
            (did, name, did.upper(), "v1", "Public domain", "http://x",
             "2026-06-10", "ck"),
        )
        conn.execute(
            "INSERT INTO section VALUES(?,?,?,?,?,NULL)",
            (did, "1", 0, "Title", f"Body of {did}."),
        )
    # A proof-texted section for the reverse index (HC 60 cites GEN 1:1-2).
    conn.execute(
        "INSERT INTO section VALUES('heidelberg','60',1,NULL,'Only by faith.',"
        "'[{\"book\":\"GEN\",\"chapter\":1,\"verse_start\":1,"
        "\"verse_end\":2}]')")
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
    conn.execute(
        "CREATE TABLE crossref(src_book TEXT, src_chapter INT, src_verse INT,"
        " tgt_book TEXT, tgt_chapter INT, tgt_verse INT, tgt_end_book TEXT,"
        " tgt_end_chapter INT, tgt_end_verse INT, is_range INT, votes INT,"
        " rank INT, src_resolves INT, tgt_resolves INT)")
    conn.executemany(
        "INSERT INTO crossref VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            # GEN 1:1 -> JHN 1:1 (maps to the fixture's bsb JHN chunk).
            ("GEN", 1, 1, "JHN", 1, 1, "JHN", 1, 1, 0, 50, 1, 1, 1),
            # GEN 1:1 -> PSA 33:6, community-downvoted (signed votes preserved).
            ("GEN", 1, 1, "PSA", 33, 6, "PSA", 33, 6, 0, -3, 2, 1, 1),
            # Non-resolving edge: must NOT be packed.
            ("GEN", 1, 1, "REV", 99, 1, "REV", 99, 1, 0, 7, 3, 1, 0),
        ])
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
        " truncated INTEGER, header TEXT NOT NULL DEFAULT '', parent_id TEXT,"
        " indexed INTEGER NOT NULL DEFAULT 1, question INTEGER,"
        " lords_day INTEGER, conf_chapter INTEGER, conf_section INTEGER,"
        " article INTEGER);"
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
        ("scr_j", "scripture", "bsb", "bsb:JHN 1:1", "JHN", 1, 1, 1, None,
         "in the beginning was the word", "h6", 0),
    ]
    conn.executemany(
        "INSERT INTO chunk (id, resource_type, source, anchor, book, chapter,"
        " verse_start, verse_end, key, text, text_checksum, truncated)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
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

    # BM25 N reflects ONLY the bundled subset (2 bsb scripture + 1 wsc = 3
    # docs), not the global 6.
    n_docs = bun.execute(
        "SELECT value FROM bm25_stats WHERE key='n_docs'").fetchone()[0]
    assert n_docs == 3.0
    # "light" appears in the bsb GEN chunk; doc_freq must be <= bundled N.
    df = bun.execute(
        "SELECT doc_freq FROM bm25_term WHERE term='light'").fetchone()
    assert df is not None and df[0] <= 3
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
    assert [r[0] for r in rows] == list(range(1, 7))
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


def test_bundled_crossrefs_pack_edges_and_expansion(tmp_path):
    from lampstand_corpus.crossref_pack import (
        decode_neighbors,
        decode_targets,
        verse_key,
    )

    out = tmp_path / "output"
    _build_fixture(out)
    package_corpus(out, out / "packs")

    cp = sqlite3.connect(out / "packs" / "bundled_crossrefs.sqlite")
    meta = dict(cp.execute("SELECT key, value FROM meta"))
    assert meta["format"] == "crossrefs-pack-v1"
    assert "CC-BY" in meta["license"]
    assert meta["attribution"] == "courtesy openbible"

    # Edge table: one source (GEN 1:1), TWO resolving targets in rank order —
    # the non-resolving edge is excluded; signed votes survive zigzag.
    n_targets, blob = cp.execute(
        "SELECT n_targets, targets FROM crossref WHERE src_verse=?",
        (verse_key("GEN", 1, 1),)).fetchone()
    targets = decode_targets(blob)
    assert n_targets == 2 and len(targets) == 2
    assert targets[0] == (verse_key("JHN", 1, 1), verse_key("JHN", 1, 1), 50)
    assert targets[1] == (verse_key("PSA", 33, 6), verse_key("PSA", 33, 6), -3)

    # Expansion: BOTH translations of GEN 1:1 point at the bsb JHN pericope
    # (the downvoted PSA target has no chunk and would be negative anyway).
    sp = sqlite3.connect(out / "packs" / "ondemand_search.sqlite")
    int_id = {sid: i for i, sid in sp.execute("SELECT id, string_id FROM chunk")}
    for src_sid in ("scr_a", "scr_b"):
        n, nblob = cp.execute(
            "SELECT n_neighbors, neighbors FROM chunk_crossref WHERE chunk_id=?",
            (int_id[src_sid],)).fetchone()
        assert n == 1
        assert decode_neighbors(nblob) == [(int_id["scr_j"], 50)]
    # JHN 1:1 has no outgoing edges -> no expansion row.
    assert cp.execute(
        "SELECT count(*) FROM chunk_crossref WHERE chunk_id=?",
        (int_id["scr_j"],)).fetchone()[0] == 0

    # Reverse proof-text index (Rank 14): HC 60 cites GEN 1:1 in the fixture.
    rows = cp.execute(
        "SELECT document, key FROM prooftext WHERE verse_key=?",
        (verse_key("GEN", 1, 1),)).fetchall()
    assert ("heidelberg", "60") in rows
    assert int(dict(cp.execute("SELECT key, value FROM meta"))
               ["n_prooftext_rows"]) >= 1
    sp.close()
    cp.close()


def test_context_parent_rows_have_no_postings_or_vectors(tmp_path):
    """A context-only pericope parent (indexed=0) lands in the search pack
    with doc_len 0, no vector row, and its children carry parent_id."""
    out = tmp_path / "output"
    _build_fixture(out)
    # Add a parent + re-point scr_a at it.
    conn = sqlite3.connect(out / "embeddings.sqlite")
    conn.execute(
        "INSERT INTO chunk (id, resource_type, source, anchor, book, chapter,"
        " verse_start, verse_end, key, text, text_checksum, truncated,"
        " header, indexed) VALUES ('scr_p1','scripture','bsb',"
        "'bsb:pericope GEN 1:1-2','GEN',1,1,2,NULL,"
        "'in the beginning bsb light and more','hp',0,'Genesis 1:1-2 — ',0)")
    conn.execute("UPDATE chunk SET parent_id='scr_p1' WHERE id='scr_a'")
    conn.commit()
    conn.close()
    package_corpus(out, out / "packs")

    sp = sqlite3.connect(out / "packs" / "ondemand_search.sqlite")
    pid, doc_len, indexed = sp.execute(
        "SELECT id, doc_len, indexed FROM chunk WHERE string_id='scr_p1'"
    ).fetchone()
    assert (doc_len, indexed) == (0, 0)
    child_parent, = sp.execute(
        "SELECT parent_id FROM chunk WHERE string_id='scr_a'").fetchone()
    assert child_parent == pid
    # The parent must appear in NO posting blob.
    from lampstand_corpus.pack_codec import decode_postings
    for blob, in sp.execute("SELECT postings FROM bm25_term"):
        assert all(cid != pid for cid, _tf in decode_postings(blob))
    sp.close()
    vp = sqlite3.connect(out / "packs" / "ondemand_vectors.sqlite")
    assert vp.execute(
        "SELECT count(*) FROM embedding WHERE chunk_id=?", (pid,)
    ).fetchone()[0] == 0
    vp.close()


def test_parent_text_is_null_children_keep_text(tmp_path):
    """Pack-diet: context-only pericope parents (indexed=0) store NULL display
    text (redundant with their children); children keep full text and stay
    self-sufficient for search-result rendering."""
    out = tmp_path / "output"
    _build_fixture(out)
    # Add a pericope parent + re-point two children (scr_a GEN1:1, and a new
    # GEN1:2 child) at it.
    conn = sqlite3.connect(out / "embeddings.sqlite")
    conn.execute(
        "INSERT INTO chunk (id, resource_type, source, anchor, book, chapter,"
        " verse_start, verse_end, key, text, text_checksum, truncated,"
        " header, indexed) VALUES ('scr_p1','scripture','bsb',"
        "'bsb:pericope GEN 1:1-2','GEN',1,1,2,NULL,"
        "'in the beginning bsb light and there was more','hp',0,"
        "'Genesis 1:1-2 — ',0)")
    conn.execute(
        "INSERT INTO chunk (id, resource_type, source, anchor, book, chapter,"
        " verse_start, verse_end, key, text, text_checksum, truncated,"
        " header, parent_id, indexed) VALUES ('scr_a2','scripture','bsb',"
        "'bsb:GEN 1:2','GEN',1,2,2,NULL,'and there was more','h7',0,"
        "'Genesis 1:2 — ','scr_p1',1)")
    conn.execute("UPDATE chunk SET parent_id='scr_p1' WHERE id='scr_a'")
    conn.commit()
    conn.close()
    package_corpus(out, out / "packs")

    sp = sqlite3.connect(out / "packs" / "ondemand_search.sqlite")
    # Parent: NULL text, but ids/linkage/metadata intact.
    ptext, pindexed, pcs = sp.execute(
        "SELECT text, indexed, text_checksum FROM chunk WHERE string_id='scr_p1'"
    ).fetchone()
    assert ptext is None and pindexed == 0
    assert pcs  # checksum (provenance) is preserved

    # Every indexed child keeps its (non-null) display text.
    child_rows = sp.execute(
        "SELECT string_id, text FROM chunk WHERE indexed=1").fetchall()
    assert child_rows  # sanity
    for sid, text in child_rows:
        assert text is not None and text != "", sid
    # Specifically the children of scr_p1 carry their own verse text.
    by_sid = {r[0]: r[1] for r in child_rows}
    assert by_sid["scr_a"] == "in the beginning bsb light"
    assert by_sid["scr_a2"] == "and there was more"

    # No NON-parent chunk was NULLed (guards against over-broad NULLing).
    n_null_indexed = sp.execute(
        "SELECT count(*) FROM chunk WHERE indexed=1 AND text IS NULL"
    ).fetchone()[0]
    assert n_null_indexed == 0
    sp.close()


def test_preserve_models_subtree_carries_coreml_entries(tmp_path):
    from lampstand_corpus.package import preserve_models_subtree

    manifest_path = tmp_path / "corpus_manifest.json"
    # No existing manifest -> nothing to preserve.
    fresh = {"packs": {"bundled": {}}}
    assert preserve_models_subtree(manifest_path, fresh) is False
    assert "models" not in fresh["packs"]
    # Existing manifest with a models subtree -> carried into the new one.
    manifest_path.write_text(json.dumps(
        {"packs": {"models": {"files": [{"name": "BGEQuery.mlpackage"}]}}}))
    assert preserve_models_subtree(manifest_path, fresh) is True
    assert fresh["packs"]["models"]["files"][0]["name"] == "BGEQuery.mlpackage"


def test_ondemand_vectors_is_default_tier_grouped_with_search(tmp_path):
    """F5: ondemand_vectors ships default-tier (not opt-in), grouped with
    ondemand_search as the 'retrieval-index'; bundled files carry no tier."""
    out = tmp_path / "output"
    _build_fixture(out)
    result = package_corpus(out, out / "packs")
    od = result.manifest["packs"]["on_demand"]
    by_name = {f["name"]: f for f in od["files"]}

    # Vectors + search are default-tier and share the retrieval-index group.
    for name in ("ondemand_vectors.sqlite", "ondemand_search.sqlite"):
        assert by_name[name]["tier"] == "default", name
        assert by_name[name]["download_group"] == "retrieval-index", name
    # Every on-demand file is default tier today (no opt-in slot used).
    assert all(f["tier"] == "default" for f in od["files"])
    # Content packs are grouped separately from the retrieval index.
    assert by_name["ondemand_commentaries.sqlite"]["download_group"] == "content"

    # The default-set size is the sum of tier=default files.
    assert od["tiers"]["default"]["bytes"] == sum(
        f["bytes"] for f in od["files"] if f["tier"] == "default")
    # The app-reader contract + graceful-degradation note are present.
    assert "tier == \"default\"" in od["app_reader"]
    assert "BM25-only" in od["default_note"]

    # Bundled files carry NO tier (tiering is on-demand only).
    for f in result.manifest["packs"]["bundled"]["files"]:
        assert "tier" not in f and "download_group" not in f

    # Grouping/metadata only: no pack bytes changed vs a run without reading
    # the manifest (the .sqlite files are identical across the two packs dirs).
    package_corpus(out, out / "packs_b")
    for f in sorted((out / "packs").glob("*.sqlite")):
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
