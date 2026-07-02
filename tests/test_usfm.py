"""USFM parser unit tests: marker stripping, verse ranges, words-of-Christ spans.

These run on small inline USFM fixtures (no network, no snapshots) so they are
fast and hermetic. End-to-end spot checks against the real snapshots live in
test_corpus_spotcheck.py and skip when snapshots aren't present.
"""

from __future__ import annotations

from lampstand_corpus.usfm import extract_bsb_omission_notes, parse_usfm


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


def test_omitted_verse_empty_text_no_source_note():
    # A normal (non-footnote) empty verse parses to empty text and a NULL note.
    pb = parse_usfm("\\id MAT\n\\c 18\n\\p \\v 11\n\\v 12 What do you think?\n")
    v11 = next(v for v in pb.verses if v.verse_start == 11)
    assert v11.text == ""
    assert v11.source_note is None


def test_omitted_verse_recovers_source_note():
    # ASV-style: the omitted verse carries an "ancient authorities insert" footnote;
    # we recover its plain text (caller, \fr ref, and \ft/\fqa tags stripped).
    raw = ("\\id MAT\n\\c 18\n"
           "\\v 11 \\f + \\fr 18:11 \\ft Many authorities insert v. 11. "
           "\\fqa For the son of man came to save that which was lost. "
           "\\ft See Luk 19:10.\\f*\n")
    pb = parse_usfm(raw)
    v = pb.verses[0]
    assert v.text == ""
    assert v.source_note is not None
    assert "Many authorities insert" in v.source_note
    assert "save that which was lost" in v.source_note
    assert "\\f" not in v.source_note and "18:11" not in v.source_note


def test_source_note_only_on_empty_verses():
    # A verse with real text never gets a source_note even if it carries a footnote.
    raw = ("\\id JHN\n\\c 1\n\\p \\v 6 There was a man"
           "\\f + \\fr 1:6 \\ft a note\\f* sent from God.\n")
    pb = parse_usfm(raw)
    v = pb.verses[0]
    assert v.text == "There was a man sent from God."
    assert v.source_note is None


def test_extract_bsb_omission_note_basic():
    # BSB Matt 18:11 lives inside 18:10's footnote as \fv 11\fv*...; recover it,
    # keyed by (chapter, verse), with \ref display kept and markup stripped.
    raw = (
        "\\id MAT\n\\c 18\n\\p \\v 10 See that you do not look down on these."
        "\\f + \\fr 18:10 \\ft BYZ and TR include \\fqa \\fv 11\\fv*For the Son of "
        "Man came to save the lost\\ft ; see \\ref Luke 19:10|LUK 19:10\\ref*.\\f*\n"
        "\\p \\v 12 What do you think?\n"
    )
    notes = extract_bsb_omission_notes(raw)
    assert notes[(18, 11)] == "For the Son of Man came to save the lost"


def test_extract_bsb_omission_note_multiple_fv_in_one_footnote():
    # A single footnote can name several omitted verses (Acts 24:7-8). Each \fv
    # marker is mapped independently to its own segment.
    raw = (
        "\\id ACT\n\\c 24\n\\p \\v 6 who even tried to desecrate the temple."
        "\\f + \\fr 24:6 \\ft TR includes \\fqa and we would have judged him "
        "according to our law. \\fv 7\\fv*But Lysias the commander came with great "
        "force and took him out of our hands, \\fv 8\\fv*ordering his accusers to "
        "come before you.\\f*\n"
    )
    notes = extract_bsb_omission_notes(raw)
    assert notes[(24, 7)].startswith("But Lysias the commander came with great force")
    assert notes[(24, 8)] == "ordering his accusers to come before you."


def test_extract_bsb_omission_note_absent_when_no_fv():
    # A footnote with no \fv marker yields nothing — never fabricated.
    raw = (
        "\\id JHN\n\\c 1\n\\p \\v 6 There was a man"
        "\\f + \\fr 1:6 \\ft just a translation note\\f* sent from God.\n"
    )
    assert extract_bsb_omission_notes(raw) == {}


# --- Psalm superscription handling (\d) --------------------------------------
def test_bsb_numbered_superscription_keeps_verse_one_body():
    # BSB numbers the superscription as verse 1 (\d \v 1 ...) and the body follows
    # on a \q line with no \v. Verse 1 must SURVIVE with the body line as its text
    # and the superscription captured separately — not dropped, not merged.
    raw = (
        "\\id PSA\n\\c 3\n\\s1 Deliver Me\n"
        "\\d \\v 1 A Psalm of David, when he fled from Absalom.\n"
        "\\q1 O LORD, how my foes have increased!\n"
        "\\q2 How many rise up against me!\n"
        "\\q1 \\v 2 Many say of me.\n"
    )
    pb = parse_usfm(raw)
    v1 = next(v for v in pb.verses if v.verse_start == 1)
    assert "O LORD, how my foes have increased" in v1.text
    assert "Psalm of David" not in v1.text          # superscription not in body
    assert v1.superscription == "A Psalm of David, when he fled from Absalom."
    v2 = next(v for v in pb.verses if v.verse_start == 2)
    assert v2.text == "Many say of me."
    assert v2.superscription is None


def test_kjv_unnumbered_superscription_does_not_steal_verse_one():
    # KJV/ASV/WEB leave the superscription unnumbered: \d on its own line, then
    # \v 1 on the NEXT line carrying the real body. The body must NOT be routed to
    # the superscription; verse 1 keeps its text and the \d is captured as metadata.
    raw = (
        "\\id PSA\n\\c 16\n"
        "\\d Michtam of David.\n"
        "\\q1\n"
        "\\v 1 Preserve me, O God: for in thee do I put my trust.\n"
        "\\v 2 O my soul, thou hast said unto the LORD.\n"
    )
    pb = parse_usfm(raw)
    v1 = next(v for v in pb.verses if v.verse_start == 1)
    assert v1.text == "Preserve me, O God: for in thee do I put my trust."
    assert v1.superscription == "Michtam of David."
    v2 = next(v for v in pb.verses if v.verse_start == 2)
    assert v2.text == "O my soul, thou hast said unto the LORD."


def test_no_superscription_psalm_unaffected():
    # A psalm without \d numbers normally; verse 1 has no superscription.
    raw = "\\id PSA\n\\c 1\n\\q1 \\v 1 Blessed is the man.\n\\q1 \\v 2 But his delight.\n"
    pb = parse_usfm(raw)
    v1 = next(v for v in pb.verses if v.verse_start == 1)
    assert v1.text == "Blessed is the man."
    assert v1.superscription is None


# --- Rank 8a: section headings + paragraph starts -------------------------------
def test_heading_attaches_to_next_verse():
    raw = (
        "\\id GEN\n\\c 1\n\\s1 The Creation\n\\p\n"
        "\\v 1 In the beginning God created the heavens and the earth.\n"
        "\\v 2 Now the earth was formless and void.\n"
        "\\s1 The First Day\n\\p\n"
        "\\v 3 And God said, Let there be light.\n"
    )
    pb = parse_usfm(raw)
    assert pb.headings == [(1, 1, "The Creation"), (1, 3, "The First Day")]


def test_heading_text_is_cleaned_of_inline_markup():
    raw = ("\\id PSA\n\\c 119\n\\s \\tl א ALEPH.\\tl* \n\\q1\n"
           "\\v 1 Blessed are the undefiled in the way.\n")
    pb = parse_usfm(raw)
    assert pb.headings == [(119, 1, "א ALEPH.")]


def test_para_start_flags_paragraph_opening_verses():
    raw = (
        "\\id GEN\n\\c 1\n\\p\n"
        "\\v 1 In the beginning.\n"
        "\\v 2 Now the earth was formless.\n"
        "\\p\n"
        "\\v 3 And God said.\n"
    )
    pb = parse_usfm(raw)
    flags = {v.verse_start: v.para_start for v in pb.verses}
    assert flags == {1: True, 2: False, 3: True}


def test_mid_verse_paragraph_break_does_not_flag_next_verse():
    # \p carrying continuation text is a mid-verse break: verse 2 does NOT open
    # a paragraph.
    raw = (
        "\\id GEN\n\\c 1\n\\p\n"
        "\\v 1 First part of the verse\n"
        "\\p and its continuation after a break.\n"
        "\\v 2 The next verse.\n"
    )
    pb = parse_usfm(raw)
    v1 = next(v for v in pb.verses if v.verse_start == 1)
    v2 = next(v for v in pb.verses if v.verse_start == 2)
    assert "continuation" in v1.text
    assert v1.para_start is True and v2.para_start is False


def test_heading_verse_text_unchanged():
    # Ingesting \s must not leak heading words into verse body text.
    raw = ("\\id GEN\n\\c 1\n\\s1 The Creation\n\\p\n\\v 1 In the beginning.\n")
    pb = parse_usfm(raw)
    assert pb.verses[0].text == "In the beginning."
