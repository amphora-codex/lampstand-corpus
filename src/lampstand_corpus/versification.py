"""Canonical reference spine + per-translation versification maps.

Architect decision "Option A": **standard English / KJV numbering is the canonical
reference spine.** Every confession proof-text, commentary anchor, and (future)
cross-reference is recorded against that spine. When the app resolves a canonical
reference to a *translation's* stored verse text, a per-translation versification
map is applied so the reference lands on the right line even when the translation
numbers verses differently.

This module owns those maps. It NEVER renumbers a translation's stored text — the
stored text keeps its native numbering; the map only translates a canonical
reference into the coordinates to query.

The known case — BSB Hebrew Psalm superscriptions
-------------------------------------------------
The Berean Standard Bible follows the Hebrew/Masoretic convention of treating a
Psalm's superscription ("For the choirmaster. A Psalm of David…") as **verse 1**.
116 psalms carry such a superscription on the BSB ``\\d \\v 1`` line (the list is
derived from the BSB USFM itself — see :data:`BSB_PSALM_SUPERSCRIPTION_V1`, NOT
guessed). Two psalms (1, 107) carry an *unnumbered* superscription, and 32 carry
none; those are unaffected.

What we verified empirically against the built ``bibles.sqlite`` (KJV is the
canonical spine here):

* For every superscribed psalm, BSB's verse 1 = the superscription, and BSB's
  **body verse N aligns 1:1 with KJV's body verse N** — i.e. the body offset is
  **0**, not +1. (113/116 confirmed by direct verse-content overlap; the other 3
  are short-verse content ties, not shifts.) So a canonical ``Ps N:V`` for ``V>=2``
  resolves to BSB ``Ps N:V`` directly.
* The ONLY canonical reference that needs care is ``Ps N:1`` for a superscribed
  psalm. In KJV/English numbering ``Ps N:1`` is the first *body* line; in BSB the
  same content sits in **verse 1 alongside the superscription** (BSB folds the
  superscription and the first poetic line into one verse-1 row). So canonical
  ``Ps N:1`` resolves to BSB ``Ps N:1`` — the verse that also carries the
  superscription.

IMPORTANT — divergence from the originally-stated "+1 offset" model
-------------------------------------------------------------------
The architect's framing described BSB as "body starts at v.2", implying a +1 shift
of every Psalm reference. The data does NOT support a blanket +1: BSB body verse
numbers match KJV (offset 0). A +1 map would mis-resolve every superscribed-psalm
reference (it would send ``Ps 3:1`` to BSB ``3:2`` = KJV's *verse 2* content). We
therefore implement the **content-verified offset-0 map** and FLAG this divergence
for the architect rather than silently coding the stated-but-wrong +1.

The map is identity-on-the-body, with verse 1 of a superscribed psalm being the
superscription-folded verse. The reason ``Ps N:1`` refs *appeared* to fail against
BSB earlier was a separate ingestion bug: the BSB ``\\d \\v 1`` line was dropped
wholesale by the USFM parser, so BSB had no verse-1 row at all. That is fixed in
``usfm.py`` (verse 1's body line is preserved; the superscription text is dropped,
consistent with how KJV/ASV/WEB treat ``\\d``). With that fix BSB carries a verse 1
for every superscribed psalm and the canonical map is a clean identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import VerseRef

# --- BSB Psalm superscription set (derived from the BSB USFM, not guessed) ----
# Every psalm whose BSB superscription is numbered as verse 1 (the ``\d \v 1``
# line in sources/bsb/bsb_usfm.zip → PSA.usfm). Regenerate with
# scripts-style probe: split on ``\c N`` and test each body for ``\d ... \v 1``.
BSB_PSALM_SUPERSCRIPTION_V1: frozenset[int] = frozenset({
    3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 98, 100, 101, 102, 103, 108,
    109, 110, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 138, 139, 140, 141, 142, 143, 144, 145,
})

# Psalms whose BSB superscription is present but UNNUMBERED (a \d line without a
# \v 1 on it) — these number like KJV and need no map entry. Kept for the record.
BSB_PSALM_SUPERSCRIPTION_UNNUMBERED: frozenset[int] = frozenset({1, 107})


# --- Non-Psalm versification differences (discovered + flagged) ---------------
# The task asks us to flag any non-Psalm versification differences found while
# building the map. Surveying max-verse-per-chapter across the four translations,
# the ONLY non-Psalm divergence is the Romans doxology relocation in the WEB:
#
#   * KJV / ASV / BSB place the doxology at Romans 16:25-27 (Rom 14 ends at v.23,
#     Rom 16 ends at v.27).
#   * WEB places the doxology at Romans 14:24-26 instead (Rom 14 ends at v.26,
#     Rom 16 ends at v.25) — a well-known textual-tradition difference.
#
# v1 bundles BSB only, and the canonical spine is KJV numbering (doxology at
# 16:25-27), which BSB shares — so this needs no active map entry for v1. It is
# recorded here so that if WEB is ever bundled, a canonical Rom 16:25-27 reference
# can be mapped to WEB's 14:24-26. NOT applied silently; surfaced for the architect.
#
# (book, canonical (chapter, verse_start, verse_end)) -> {translation: (ch, vs, ve)}
NON_PSALM_VERSIFICATION_NOTES: dict[str, str] = {
    "rom_doxology": (
        "Romans doxology: canonical/KJV/ASV/BSB at Rom 16:25-27; WEB relocates it "
        "to Rom 14:24-26. Only relevant if WEB is bundled. Flagged, not auto-mapped."
    ),
}


@dataclass(frozen=True)
class ResolvedRef:
    """The result of resolving a canonical reference against one translation."""

    book: str
    chapter: int
    verse: int
    note: str | None = None  # human-facing note (e.g. superscription folding)


def resolve(translation: str, ref: VerseRef) -> ResolvedRef:
    """Resolve a *canonical* (KJV/standard-spine) reference to a translation.

    Returns the verse number to query in ``translation``'s stored text. The stored
    text is never renumbered — this only maps the reference. ``ref.verse_start`` is
    used (callers handle ranges by resolving both ends).

    The only non-identity case is BSB Psalms: for a superscribed psalm, canonical
    verse 1 resolves to BSB verse 1 (the verse that also carries the
    superscription); body verses (>=2) resolve unchanged (offset 0). Every other
    translation/book is the identity map for v1 corpus.
    """
    book, chapter, verse = ref.book, ref.chapter, ref.verse_start
    if translation == "bsb" and book == "PSA" and \
            chapter in BSB_PSALM_SUPERSCRIPTION_V1:
        if verse == 1:
            return ResolvedRef(
                book, chapter, 1,
                note="BSB verse 1 carries the Hebrew superscription folded with "
                     "the first body line; canonical (KJV) verse 1 is that line.",
            )
        # Body verses align 1:1 with KJV (offset 0) — identity.
        return ResolvedRef(book, chapter, verse)
    return ResolvedRef(book, chapter, verse)


def is_superscription_verse(translation: str, book: str, chapter: int,
                            verse: int) -> bool:
    """True iff (translation, book, chapter, verse) is a BSB Psalm verse-1 that
    folds in the Hebrew superscription. Lets the app render the superscription
    distinctly and lets the validator reason about verse-1 specially."""
    return (
        translation == "bsb" and book == "PSA" and verse == 1
        and chapter in BSB_PSALM_SUPERSCRIPTION_V1
    )
