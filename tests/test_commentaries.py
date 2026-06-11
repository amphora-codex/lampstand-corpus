"""Commentary parser unit tests + built-DB spot checks.

Unit tests run on small inline ThML fixtures (no network, no snapshots). The
spot-check class queries the built commentaries.sqlite and skips automatically
when it isn't present (fresh code-only checkout). Run
``python -m lampstand_corpus.cli build-commentaries`` first to exercise them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lampstand_corpus.commentaries import (
    CALVIN_NT_NOT_WRITTEN,
    CALVIN_VOLUMES_DEFERRED,
    CALVIN_VOLUMES_IN_SCOPE,
    COMMENTARY_SOURCES,
    CommentarySource,
    _parse_scripcom,
    parse_commentary,
)
from lampstand_corpus.schema import Provenance, ResourceType
from lampstand_corpus.validate_commentaries import (
    CALVIN_EXPECTED_BOOKS,
    validate_commentary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "output" / "commentaries.sqlite"

_PROV = Provenance(
    source="ccel:test", version="t", license="pd",
    retrieved="2026-06-10", url="u", checksum="cafe",
)


def _src(cid: str, volume: str) -> CommentarySource:
    """A single-volume test source mirroring a real commentator's id/shortcode."""
    base = COMMENTARY_SOURCES[cid]
    return CommentarySource(
        id=cid, name=base.name, shortcode=base.shortcode, author=base.author,
        work=base.work, volumes=(volume,), author_slug=base.author_slug,
        version="t", license="pd",
    )


def _parse(cid: str, volume: str, body: str):
    content = (
        "<?xml version='1.0'?><ThML><ThML.body>" + body + "</ThML.body></ThML>"
    )
    src = _src(cid, volume)
    return parse_commentary(src, {volume: _PROV}, {volume: content})


# --- scripCom mapping --------------------------------------------------------
def test_parse_scripcom_verse():
    ref, chap_level, reason = _parse_scripcom("|Rom|9|23|0|0", "Bible:Rom.9.23")
    assert reason is None and not chap_level
    assert (ref.book, ref.chapter, ref.verse_start) == ("ROM", 9, 23)
    assert ref.verse_end is None


def test_parse_scripcom_range():
    ref, chap_level, _ = _parse_scripcom("|Gen|1|1|1|2", "Bible:Gen.1.1-Gen.1.2")
    assert (ref.book, ref.chapter, ref.verse_start, ref.verse_end) == ("GEN", 1, 1, 2)
    assert not chap_level


def test_parse_scripcom_whole_chapter_is_chapter_level():
    ref, chap_level, _ = _parse_scripcom("|Ps|119|0|0|0", "Bible:Ps.119")
    assert chap_level and ref.verse_start == 0 and ref.book == "PSA"


def test_parse_scripcom_off_canon_book_flagged():
    ref, _cl, reason = _parse_scripcom("|Tob|1|1|0|0", "Bible:Tob.1.1")
    assert ref is None and "outside the 66-book canon" in reason


def test_parse_scripcom_malformed_flagged():
    ref, _cl, reason = _parse_scripcom("|Rom|9|", "garbage")
    assert ref is None and "unparseable" in reason


# --- paragraph-level chunking ------------------------------------------------
def test_paragraph_chunks_anchor_to_verse():
    body = (
        "<div2 title='Romans 9'>"
        "<p><scripCom type='Commentary' passage='Ro 9:23' parsed='|Rom|9|23|0|0' "
        "osisRef='Bible:Rom.9.23'/></p>"
        "<p>First paragraph on verse 23.</p>"
        "<p>Second paragraph on verse 23.</p>"
        "<p><scripCom type='Commentary' passage='Ro 9:24' parsed='|Rom|9|24|0|0' "
        "osisRef='Bible:Rom.9.24'/></p>"
        "<p>Paragraph on verse 24.</p>"
        "</div2>"
    )
    pc = _parse("henry", "mhc6", body)
    assert [c.key for c in pc.chunks] == [
        "ROM.9.23#p1", "ROM.9.23#p2", "ROM.9.24#p1",
    ]
    c0 = pc.chunks[0]
    assert c0.resource_type == ResourceType.COMMENTARY
    assert (c0.ref.book, c0.ref.chapter, c0.ref.verse_start) == ("ROM", 9, 23)
    assert c0.meta["shortcode"] == "MH"
    assert c0.text == "First paragraph on verse 23."
    assert pc.coverage == {"ROM": {9}}


def test_chapter_intro_prose_attaches_to_chapter_anchor():
    # Henry places the chapter-introduction lede BEFORE the whole-chapter anchor;
    # it must attach to the chapter (verse_start=0), never be dropped or collide.
    body = (
        "<div2 title='Chapter I'>"
        "<p>Intro lede paragraph one.</p>"
        "<p>Intro lede paragraph two.</p>"
        "<p><scripCom passage='Ge 1' parsed='|Gen|1|0|0|0' osisRef='Bible:Gen.1'/></p>"
        "<p>Chapter-level note after the anchor.</p>"
        "<p><scripCom passage='Ge 1:1' parsed='|Gen|1|1|0|0' osisRef='Bible:Gen.1.1'/></p>"
        "<p>Comment on verse 1.</p>"
        "</div2>"
    )
    pc = _parse("henry", "mhc1", body)
    keys = [c.key for c in pc.chunks]
    # two lede paras + one post-anchor note all anchor to GEN 1:0, distinct keys
    assert keys == ["GEN.1.0#p1", "GEN.1.0#p2", "GEN.1.0#p3", "GEN.1.1#p1"]
    intro = [c for c in pc.chunks if c.ref.verse_start == 0]
    assert all(c.meta["chapter_level"] for c in intro)
    assert intro[0].text == "Intro lede paragraph one."
    assert pc.chunks[-1].ref.verse_start == 1


def test_offcanon_anchor_dropped_with_flag():
    body = (
        "<div2 title='X'>"
        "<p><scripCom passage='Tob 1:1' parsed='|Tob|1|1|0|0' osisRef='Bible:Tob.1.1'/></p>"
        "<p>This comments on an apocryphal book and must not be ingested.</p>"
        "</div2>"
    )
    pc = _parse("jfb", "jfb", body)
    # No valid anchor was ever established, so the stranded prose is flagged.
    assert pc.chunks == []
    assert any("outside the 66-book canon" in f for f in pc.flags)
    assert any("no scripCom verse anchor" in f for f in pc.flags)


# --- scope guards ------------------------------------------------------------
def test_calvin_scope_is_genesis_psalms_nt_only():
    # In-scope volumes never include the deferred OT (Isaiah/Jeremiah/etc).
    assert "calcom01" in CALVIN_VOLUMES_IN_SCOPE     # Genesis
    assert "calcom08" in CALVIN_VOLUMES_IN_SCOPE     # Psalms
    assert "calcom45" in CALVIN_VOLUMES_IN_SCOPE     # Catholic Epistles (NT)
    assert "calcom13" not in CALVIN_VOLUMES_IN_SCOPE  # Isaiah (deferred)
    assert "calcom07" not in CALVIN_VOLUMES_IN_SCOPE  # Joshua (deferred)
    # No volume appears in both scope and deferred sets.
    deferred_keys = " ".join(CALVIN_VOLUMES_DEFERRED)
    assert "calcom13" in deferred_keys or "13" in deferred_keys


def test_calvin_expected_books_exclude_unwritten_nt():
    for b in CALVIN_NT_NOT_WRITTEN:           # 2JN, 3JN, REV
        assert b not in CALVIN_EXPECTED_BOOKS
    assert "GEN" in CALVIN_EXPECTED_BOOKS
    assert "PSA" in CALVIN_EXPECTED_BOOKS
    assert "ROM" in CALVIN_EXPECTED_BOOKS
    # Calvin DID comment on Jude (calcom45), so it is expected.
    assert "JUD" in CALVIN_EXPECTED_BOOKS
    # The deferred OT books are NOT expected of Calvin in v1.
    assert "ISA" not in CALVIN_EXPECTED_BOOKS


def test_ccel_sources_are_the_three_whole_bible_commentators():
    # COMMENTARY_SOURCES is the CCEL-only dict the snapshot loop iterates. Spurgeon
    # is ingested from Internet Archive OCR (a different path) and lives in the
    # combined registry, NOT here; Gill is deferred to v1.1.
    assert "spurgeon" not in COMMENTARY_SOURCES
    assert "gill" not in COMMENTARY_SOURCES
    assert set(COMMENTARY_SOURCES) == {"henry", "jfb", "calvin"}


def test_spurgeon_now_in_combined_registry():
    from lampstand_corpus.commentaries import all_commentary_sources

    combined = all_commentary_sources()
    assert "spurgeon" in combined
    assert "gill" not in combined  # still deferred to v1.1
    assert combined["spurgeon"].shortcode == "CHS"


# --- validation --------------------------------------------------------------
def test_validation_flags_long_block():
    long_text = "x " * 7000  # ~14k chars, over LONG_BLOCK_CHARS
    body = (
        "<div2 title='Psalm 119'>"
        "<p><scripCom passage='Ps 119' parsed='|Ps|119|0|0|0' osisRef='Bible:Ps.119'/></p>"
        f"<p>{long_text}</p>"
        "</div2>"
    )
    pc = _parse("calvin", "calcom11", body)
    rep = validate_commentary(pc)
    assert rep.long_blocks, "expected the long Psalm 119 block to be flagged"
    assert rep.error_total == 0  # a long block is an anomaly flag, not an error


def test_validation_reports_missing_books_for_whole_bible_commentator():
    body = (
        "<div2 title='John 1'>"
        "<p><scripCom passage='Jn 1:1' parsed='|John|1|1|0|0' osisRef='Bible:John.1.1'/></p>"
        "<p>In the beginning was the Word.</p>"
        "</div2>"
    )
    pc = _parse("henry", "mhc5", body)
    rep = validate_commentary(pc)
    # Henry is a whole-Bible commentator; one John verse leaves 65 books missing.
    assert "GEN" in rep.missing_books and "REV" in rep.missing_books
    assert "JHN" not in rep.missing_books
    assert rep.error_total == 0


# --- built-DB spot checks ----------------------------------------------------
@pytest.mark.skipif(not DB.exists(),
                    reason="commentaries.sqlite not built; run build-commentaries")
class TestCommentariesDB:
    @pytest.fixture(scope="class")
    def conn(self):
        c = sqlite3.connect(DB)
        yield c
        c.close()

    def test_commentators_present(self, conn):
        # Henry, JFB, Calvin (CCEL) + Spurgeon (Internet Archive OCR). Spurgeon is
        # present only when its snapshots were fetched before the build; tolerate
        # both so a CCEL-only build still passes.
        ids = {r[0] for r in conn.execute("SELECT id FROM commentator")}
        assert {"henry", "jfb", "calvin"} <= ids
        assert ids <= {"henry", "jfb", "calvin", "spurgeon"}

    def test_every_comment_ref_is_in_canon(self, conn):
        from lampstand_corpus import books
        rows = conn.execute(
            "SELECT DISTINCT book FROM comment"
        ).fetchall()
        assert rows
        for (b,) in rows:
            assert b in books.CANON, b

    def test_henry_covers_whole_bible(self, conn):
        n = conn.execute(
            "SELECT COUNT(DISTINCT book) FROM comment WHERE commentator='henry'"
        ).fetchone()[0]
        assert n == 66

    def test_calvin_excludes_deferred_ot(self, conn):
        books_ = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT book FROM comment WHERE commentator='calvin'"
            )
        }
        # In scope: Genesis, Psalms, NT. Out: Isaiah, Jeremiah, the deferred OT.
        assert "GEN" in books_ and "PSA" in books_ and "ROM" in books_
        assert "ISA" not in books_ and "JER" not in books_ and "EXO" not in books_

    def test_calvin_unwritten_nt_books_absent(self, conn):
        books_ = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT book FROM comment WHERE commentator='calvin'"
            )
        }
        for b in CALVIN_NT_NOT_WRITTEN:
            assert b not in books_, b

    def test_verse_end_ge_verse_start(self, conn):
        bad = conn.execute(
            "SELECT COUNT(*) FROM comment WHERE verse_end < verse_start"
        ).fetchone()[0]
        assert bad == 0
