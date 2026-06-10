"""Smoke tests for the normalized schema. Real coverage (verse-reference math,
normalizer round-trips, validation rules) lands with P1+."""

from lampstand_corpus.schema import NormalizedChunk, Provenance, ResourceType, VerseRef


def _provenance() -> Provenance:
    return Provenance(
        source="bsb",
        version="2024",
        license="CC0 (public domain)",
        retrieved="2026-06-10",
        url="https://bereanbible.com",
        checksum="0" * 64,
    )


def test_verseref_normalized_fills_end():
    ref = VerseRef(book="JHN", chapter=3, verse_start=16)
    assert ref.normalized().verse_end == 16


def test_verseref_range_preserved():
    ref = VerseRef(book="JHN", chapter=3, verse_start=1, verse_end=21)
    assert ref.normalized().verse_end == 21


def test_chunk_requires_provenance():
    chunk = NormalizedChunk(
        id="bsb-JHN-3-16",
        resource_type=ResourceType.SCRIPTURE,
        ref=VerseRef(book="JHN", chapter=3, verse_start=16),
        text="For God so loved the world...",
        provenance=_provenance(),
    )
    assert chunk.provenance.checksum and chunk.resource_type is ResourceType.SCRIPTURE
