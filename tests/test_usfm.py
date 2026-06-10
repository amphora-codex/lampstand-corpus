"""USFM parser unit tests: marker stripping, verse ranges, words-of-Christ spans.

These run on small inline USFM fixtures (no network, no snapshots) so they are
fast and hermetic. End-to-end spot checks against the real snapshots live in
test_corpus_spotcheck.py and skip when snapshots aren't present.
"""

from __future__ import annotations

from lampstand_corpus.usfm import parse_usfm


def test_basic_verse_and_chapter():
    pb = parse_usfm("\\id GEN\n\\c 1\n\\p \\v 1 In the beginning God created.\n")
    assert pb.book == "GEN"
    assert len(pb.verses) == 1
    v = pb.verses[0]
    assert (v.chapter, v.verse_start, v.verse_end) == (1, 1, 1)
    assert v.text == "In the beginning God created."
    assert not v.has_red_letter


def test_multiple_verses_one_paragraph_line():
    pb = parse_usfm("\\id JHN\n\\c 1\n\\p \\v 1 First. \\v 2 Second. \\v 3 Third.\n")
    assert [v.verse_start for v in pb.verses] == [1, 2, 3]
    assert pb.verses[1].text == "Second."


def test_rejects_non_canon_id():
    import pytest
    with pytest.raises(ValueError):
        parse_usfm("\\id TOB\n\\c 1\n\\v 1 Tobit text.\n")
    with pytest.raises(ValueError):
        parse_usfm("\\c 1\n\\v 1 no id.\n")


def test_strips_word_strong_markup():
    raw = ('\\id JHN\n\\c 1\n\\p \\v 1 \\w In|strong="G1722"\\w* the '
           '\\w beginning|strong="G0746"\\w* was the Word.\n')
    pb = parse_usfm(raw)
    assert pb.verses[0].text == "In the beginning was the Word."


def test_strips_footnotes_and_added_words():
    raw = ('\\id JHN\n\\c 1\n\\p \\v 6 There was a man\\f + \\fr 1:6 \\ft note\\f* '
           'sent \\add was\\add* from God.\n')
    pb = parse_usfm(raw)
    # footnote removed; \add word kept as plain text (real USFM always spaces
    # before \add — verified against the snapshots).
    assert pb.verses[0].text == "There was a man sent was from God."


def test_keeps_ref_display_text():
    raw = ('\\id JHN\n\\c 1\n\\p \\v 23 as in \\ref Isaiah 40:3|ISA 40:3\\ref* '
           'the prophet said.\n')
    pb = parse_usfm(raw)
    assert "Isaiah 40:3" in pb.verses[0].text
    assert "|" not in pb.verses[0].text


def test_verse_bridge():
    pb = parse_usfm("\\id ROM\n\\c 14\n\\p \\v 24-26 a bridged verse.\n")
    v = pb.verses[0]
    assert v.verse_start == 24 and v.verse_end == 26
    assert v.is_bridge


def test_wj_span_offsets_align():
    raw = ('\\id JHN\n\\c 1\n\\p \\v 38 He asked, \\wj "What do you want?"\\wj* '
           'they replied.\n')
    pb = parse_usfm(raw)
    v = pb.verses[0]
    assert v.has_red_letter
    assert len(v.wj_spans) == 1
    sp = v.wj_spans[0]
    assert v.text[sp.start:sp.end] == '"What do you want?"'


def test_wj_nested_plus_w():
    raw = ('\\id JHN\n\\c 3\n\\p \\v 16 \\wj \\+w For|strong="G1063"\\+w* '
           '\\+w God|strong="G2316"\\+w* loved\\wj* the world.\n')
    pb = parse_usfm(raw)
    v = pb.verses[0]
    assert v.text == "For God loved the world."
    assert len(v.wj_spans) == 1
    assert v.text[v.wj_spans[0].start:v.wj_spans[0].end] == "For God loved"


def test_pilcrow_removed():
    pb = parse_usfm("\\id JHN\n\\c 1\n\\p \\v 6 ¶ There was a man.\n")
    assert "¶" not in pb.verses[0].text
    assert pb.verses[0].text == "There was a man."


def test_headings_dropped():
    raw = ("\\id JHN\n\\c 1\n\\s1 The Beginning\n\\r (cross ref)\n"
           "\\p \\v 1 In the beginning.\n")
    pb = parse_usfm(raw)
    assert len(pb.verses) == 1
    assert pb.verses[0].text == "In the beginning."
