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
from lampstand_corpus.package import (
    CORPUS_VERSION_PLACEHOLDER,
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

    # Whole-DB copies present for the heavy resources.
    for name in ("ondemand_commentaries.sqlite", "ondemand_lexicons.sqlite",
                 "ondemand_crossrefs.sqlite", "ondemand_embeddings.sqlite"):
        assert (out / "packs" / name).exists()


def test_bundled_search_reuses_vectors_and_recomputes_bm25(tmp_path):
    out = tmp_path / "output"
    _build_fixture(out)
    package_corpus(out, out / "packs")

    full = sqlite3.connect(out / "embeddings.sqlite")
    bun = sqlite3.connect(out / "packs" / "bundled_search.sqlite")

    # Only bsb-scripture + wsc-confession survive into the bundled index.
    sources = sorted(r[0] for r in bun.execute(
        "SELECT DISTINCT source FROM chunk"))
    assert sources == ["bsb", "wsc"]

    # Vectors are byte-identical to the full index (reused, not re-encoded).
    for cid in ("scr_a", "con_w"):
        vb = bun.execute(
            "SELECT vector FROM embedding WHERE chunk_id=?", (cid,)).fetchone()[0]
        vf = full.execute(
            "SELECT vector FROM embedding WHERE chunk_id=?", (cid,)).fetchone()[0]
        assert vb == vf

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
