"""Normalized records + provenance. Every chunk carries full provenance so the
output is auditable and reproducible (docs/normalized-schema.md)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ResourceType(str, Enum):
    SCRIPTURE = "scripture"
    COMMENTARY = "commentary"
    CONFESSION = "confession"
    LEXICON = "lexicon"
    CROSSREF = "crossref"


class Provenance(BaseModel):
    """Where a chunk came from — attached to every normalized record."""

    source: str          # e.g. "bsb", "ccel:henry", "openscriptures:strongs"
    version: str
    license: str
    retrieved: str       # ISO date (YYYY-MM-DD)
    url: str
    checksum: str        # SHA-256 of the source snapshot


class VerseRef(BaseModel):
    """Canonical verse reference — the spine everything anchors to."""

    book: str            # stable book id
    chapter: int
    verse_start: int
    verse_end: int | None = None

    def normalized(self) -> "VerseRef":
        return self if self.verse_end else VerseRef(
            book=self.book, chapter=self.chapter,
            verse_start=self.verse_start, verse_end=self.verse_start,
        )


class NormalizedChunk(BaseModel):
    """One unit of corpus content in the common format, pre-SQLite."""

    id: str
    resource_type: ResourceType
    ref: VerseRef | None = None      # None for lexicon entries keyed by lemma/Strong's
    key: str | None = None           # lexicon: Strong's number / lemma; confession: section id
    text: str
    meta: dict = Field(default_factory=dict)
    provenance: Provenance


# TODO(P1+): per-source normalizers populate these models; build/ writes them to
# the per-type SQLite databases; validate/ checks ref integrity + emits the report.
