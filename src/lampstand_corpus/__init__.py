"""lampstand-corpus — corpus ingestion, normalization, embedding, and validation.

Pipeline stages (see docs/normalized-schema.md):
    sources   → fetch + checksum versioned snapshots
    schema    → normalized record + provenance models
    normalize → per-source converters to the normalized format
    build     → SQLite databases + embedding/BM25 indices
    validate  → validation report + human-review flags
"""

__version__ = "0.1.0"
