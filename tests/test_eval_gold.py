"""Gold-set builder tests — tiny synthetic DB fixtures, never the real corpus.

Covers the verse→chunk anchoring (incl. range/translation handling), the
query-text extractors, the data-driven crossref vote threshold, the DRAFT
hard-negative generator, and end-to-end determinism of ``build_gold``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus import eval_gold


# --- fixture builders ------------------------------------------------------------
def _mk_embeddings(path: Path, chunks: list[tuple]) -> None:
    """chunks: (id, resource_type, source, anchor, book, chapter, vs, ve, key).

    All fixture chunks are retrieval units (indexed=1); the schema carries the
    Rank-8 columns the builder queries.
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE chunk (id TEXT PRIMARY KEY, resource_type TEXT, source TEXT,"
        " anchor TEXT, book TEXT, chapter INTEGER, verse_start INTEGER,"
        " verse_end INTEGER, key TEXT, indexed INTEGER NOT NULL DEFAULT 1,"
        " parent_id TEXT);"
    )
    conn.executemany(
        "INSERT INTO chunk (id, resource_type, source, anchor, book, chapter,"
        " verse_start, verse_end, key) VALUES (?,?,?,?,?,?,?,?,?)", chunks)
    conn.executemany("INSERT INTO meta VALUES (?,?)", [
        ("model_revision", "testrev"), ("n_chunks", str(len(chunks)))])
    conn.commit()
    conn.close()


_SCRIPTURE_CHUNKS = [
    # Two translations of GEN 1:1-10 + 1:11-20, one PSA chunk.
    ("s_bsb_gen1a", "scripture", "bsb", "bsb:GEN 1:1-10", "GEN", 1, 1, 10, None),
    ("s_bsb_gen1b", "scripture", "bsb", "bsb:GEN 1:11-20", "GEN", 1, 11, 20, None),
    ("s_kjv_gen1a", "scripture", "kjv", "kjv:GEN 1:1-10", "GEN", 1, 1, 10, None),
    ("s_kjv_gen1b", "scripture", "kjv", "kjv:GEN 1:11-20", "GEN", 1, 11, 20, None),
    ("s_bsb_gen2", "scripture", "bsb", "bsb:GEN 2:1-10", "GEN", 2, 1, 10, None),
    ("s_bsb_psa23", "scripture", "bsb", "bsb:PSA 23:1-6", "PSA", 23, 1, 6, None),
]


@pytest.fixture
def scripture_index(tmp_path: Path) -> eval_gold.ScriptureChunkIndex:
    db = tmp_path / "embeddings.sqlite"
    _mk_embeddings(db, _SCRIPTURE_CHUNKS)
    return eval_gold.ScriptureChunkIndex(db)


# --- verse -> chunk anchoring --------------------------------------------------
def test_covering_returns_all_translations(scripture_index):
    assert scripture_index.covering("GEN", 1, 5) == {"s_bsb_gen1a", "s_kjv_gen1a"}


def test_covering_outside_any_window_is_empty(scripture_index):
    assert scripture_index.covering("GEN", 3, 1) == set()


def test_overlapping_within_one_chapter(scripture_index):
    got = scripture_index.overlapping(("GEN", 1, 8), ("GEN", 1, 12))
    assert got == {"s_bsb_gen1a", "s_kjv_gen1a", "s_bsb_gen1b", "s_kjv_gen1b"}


def test_overlapping_crosses_chapter_boundary(scripture_index):
    got = scripture_index.overlapping(("GEN", 1, 19), ("GEN", 2, 2))
    assert got == {"s_bsb_gen1b", "s_kjv_gen1b", "s_bsb_gen2"}


def test_overlapping_reversed_range_is_empty(scripture_index):
    assert scripture_index.overlapping(("GEN", 2, 1), ("GEN", 1, 1)) == set()


# --- query-text extraction --------------------------------------------------------
def test_split_catechism_wsc_shape():
    q, a = eval_gold.split_catechism(
        "Q33. What is justification?\nA33. Justification is an act of God's free grace.")
    assert q == "What is justification?"
    assert a.startswith("Justification is an act")


def test_split_catechism_heidelberg_shape():
    q, a = eval_gold.split_catechism(
        "Question 60. How are thou righteous before God? Answer. Only by a true faith.")
    assert q == "How are thou righteous before God?"
    assert a == "Only by a true faith."


def test_split_catechism_rejects_plain_prose():
    assert eval_gold.split_catechism("Those whom God effectually calleth.") is None


def test_opening_sentence_strips_citations_and_stops_at_boundary():
    text = ("Those whom God effectually calleth (Rom. 8:30; Rom. 3:24), He freely "
            "justifieth. Not by infusing righteousness into them.")
    s = eval_gold.opening_sentence(text)
    assert s == "Those whom God effectually calleth , He freely justifieth."
    assert "Rom." not in s


def test_truncate_words_cuts_at_word_boundary():
    out = eval_gold.truncate_words("alpha beta gamma delta", 12)
    assert out == "alpha beta"
    assert eval_gold.truncate_words("  spaced   text ", 100) == "spaced text"


# --- crossref vote threshold --------------------------------------------------------
def test_crossref_vote_threshold_is_data_driven(tmp_path: Path):
    db = tmp_path / "crossrefs.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE crossref (src_book TEXT, src_chapter INT, src_verse INT,"
        " tgt_book TEXT, tgt_chapter INT, tgt_verse INT, tgt_end_book TEXT,"
        " tgt_end_chapter INT, tgt_end_verse INT, votes INT,"
        " src_resolves INT, tgt_resolves INT)")
    rows = []
    for votes, n in ((100, 3), (50, 4), (10, 5)):
        for i in range(n):
            rows.append(("GEN", 1, i + 1, "PSA", 23, 1, "PSA", 23, 1, votes, 1, 1))
    conn.executemany(
        "INSERT INTO crossref VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    # pool_min 5 -> needs votes>=50 (3+4=7 >= 5); pool_min 3 -> votes>=100.
    assert eval_gold.crossref_vote_threshold(conn, pool_min=5) == 50
    assert eval_gold.crossref_vote_threshold(conn, pool_min=3) == 100
    conn.close()


# --- hard negatives ------------------------------------------------------------------
def _mk_confessions(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE document (id TEXT PRIMARY KEY, shortcode TEXT);"
        "CREATE TABLE section (document TEXT, key TEXT, ord INTEGER, text TEXT,"
        " proof_texts TEXT);"
    )
    conn.executemany("INSERT INTO document VALUES (?,?)", [
        ("wsc", "WSC"), ("wlc", "WLC"), ("heidelberg", "HC")])
    sections = [
        ("wsc", "33", 0,
         "Q33. What is justification?\n"
         "A33. Justification is an act of God's free grace, wherein he pardoneth "
         "all our sins and accepteth us as righteous, only for the righteousness "
         "of Christ imputed to us, received by faith alone.", None),
        ("wsc", "35", 1,
         "Q35. What is sanctification?\n"
         "A35. Sanctification is the work of God's free grace, whereby we are "
         "renewed in the whole man after the image of God, and are enabled more "
         "and more to die unto sin and live unto righteousness.", None),
        ("wsc", "98", 2,
         "Q98. What is prayer?\n"
         "A98. Prayer is an offering up of our desires unto God for things "
         "agreeable to his will, in the name of Christ, with confession of our "
         "sins and thankful acknowledgment of his mercies.", None),
        ("heidelberg", "60", 0,
         "Question 60. How are thou righteous before God? Answer. Only by a true "
         "faith in Jesus Christ.",
         json.dumps([{"book": "GEN", "chapter": 1, "verse_start": 1}])),
    ]
    conn.executemany("INSERT INTO section VALUES (?,?,?,?,?)", sections)
    conn.commit()
    conn.close()


def test_hard_negative_candidates_pair_adjacent_doctrines(tmp_path: Path):
    db = tmp_path / "confessions.sqlite"
    _mk_confessions(db)
    cands = eval_gold.generate_hard_negative_candidates(db, target=10)
    assert cands, "expected at least one candidate pair"
    top = cands[0]
    # Justification vs sanctification share the free-grace formula tokens.
    assert {top["relevant_anchor"], top["hard_negative_anchor"]} == {"WSC 33", "WSC 35"}
    assert top["query"] == "What is justification?"
    # Deterministic: same input -> identical output.
    assert cands == eval_gold.generate_hard_negative_candidates(db, target=10)


def test_write_hard_negatives_marks_draft(tmp_path: Path):
    out = tmp_path / "data" / "eval" / "hard_negatives_v1.json"
    eval_gold.write_hard_negatives(out, [])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "DRAFT" in payload["status"]
    assert "advisor" in payload["status"].lower() or "review" in payload["status"].lower()


# --- end-to-end determinism -----------------------------------------------------------
def _mk_full_fixture(root: Path) -> Path:
    """A complete minimal output/ dir for build_gold."""
    out = root / "output"
    out.mkdir()
    chunks = list(_SCRIPTURE_CHUNKS) + [
        ("c_wsc33", "confession", "wsc", "WSC 33", None, None, None, None, "33"),
        ("c_wsc35", "confession", "wsc", "WSC 35", None, None, None, None, "35"),
        ("c_wsc98", "confession", "wsc", "WSC 98", None, None, None, None, "98"),
        ("c_hc60", "confession", "heidelberg", "HC 60", None, None, None, None, "60"),
        ("m_h1", "commentary", "henry", "henry:GEN.1.1#p1#0", "GEN", 1, 1, 1,
         "GEN.1.1#p1"),
    ]
    _mk_embeddings(out / "embeddings.sqlite", chunks)
    _mk_confessions(out / "confessions.sqlite")

    conn = sqlite3.connect(out / "crossrefs.sqlite")
    conn.execute(
        "CREATE TABLE crossref (src_book TEXT, src_chapter INT, src_verse INT,"
        " tgt_book TEXT, tgt_chapter INT, tgt_verse INT, tgt_end_book TEXT,"
        " tgt_end_chapter INT, tgt_end_verse INT, votes INT,"
        " src_resolves INT, tgt_resolves INT)")
    conn.execute("INSERT INTO crossref VALUES ('GEN',1,1,'PSA',23,1,'PSA',23,1,120,1,1)")
    conn.commit()
    conn.close()

    conn = sqlite3.connect(out / "bibles.sqlite")
    conn.execute(
        "CREATE TABLE verse (translation TEXT, book TEXT, chapter INT,"
        " verse_start INT, verse_end INT, text TEXT)")
    conn.execute(
        "INSERT INTO verse VALUES ('bsb','GEN',1,1,1,"
        "'In the beginning God created the heavens and the earth.')")
    conn.commit()
    conn.close()

    conn = sqlite3.connect(out / "commentaries.sqlite")
    conn.executescript(
        "CREATE TABLE commentator (id TEXT PRIMARY KEY);"
        "CREATE TABLE comment (commentator TEXT, key TEXT, ord INT, book TEXT,"
        " chapter INT, verse_start INT, verse_end INT, text TEXT)")
    conn.execute("INSERT INTO commentator VALUES ('henry')")
    conn.execute(
        "INSERT INTO comment VALUES ('henry','GEN.1.1#p1',0,'GEN',1,1,1,?)",
        ("Observe first the effect produced: the visible world, framed from "
         "nothing by the almighty word, teaches us the eternal power of its "
         "Maker, whose glory the heavens declare from the very first day. " * 2,))
    conn.commit()
    conn.close()
    return out


def test_build_gold_is_deterministic_and_categorized(tmp_path: Path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root in (root_a, root_b):
        root.mkdir()
        _mk_full_fixture(root)

    pa = eval_gold.build_gold(root_a / "output", root_a)
    eval_gold.build_gold(root_b / "output", root_b)
    text_a = (root_a / "output" / eval_gold.GOLD_FILENAME).read_text(encoding="utf-8")
    text_b = (root_b / "output" / eval_gold.GOLD_FILENAME).read_text(encoding="utf-8")
    assert text_a == text_b, "same inputs must yield a byte-identical gold set"

    cats = {q["category"] for q in pa["queries"]}
    assert cats == {"prooftext", "crossref", "commentary-anchor", "hardneg"}

    by_id = {q["id"]: q for q in pa["queries"]}
    # Prooftext: HC 60 -> GEN 1:1 chunks (both translations), self excluded.
    pt = by_id["pt_heidelberg_60"]
    assert set(pt["relevant"]) == {"s_bsb_gen1a", "s_kjv_gen1a"}
    assert pt["exclude"] == ["c_hc60"]
    assert pt["query"] == "How are thou righteous before God?"
    # Crossref: GEN 1:1 -> PSA 23 chunk; source's own chunks excluded.
    xr = by_id["xr_GEN.1.1"]
    assert xr["relevant"] == ["s_bsb_psa23"]
    assert set(xr["exclude"]) == {"s_bsb_gen1a", "s_kjv_gen1a"}
    # Commentary anchor: paragraph -> GEN 1:1 chunks, its own chunk excluded.
    ca = by_id["ca_henry_GEN.1.1#p1"]
    assert set(ca["relevant"]) == {"s_bsb_gen1a", "s_kjv_gen1a"}
    assert ca["exclude"] == ["m_h1"]
    # Hardneg: written as a DRAFT tracked file; no self-exclusion.
    hn = [q for q in pa["queries"] if q["category"] == "hardneg"]
    assert hn and hn[0]["exclude"] == []
    assert (root_a / "data" / "eval" / "hard_negatives_v1.json").exists()
