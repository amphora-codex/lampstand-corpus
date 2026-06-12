"""Smoke-test retrieval helpers — used only by the P6 validation report.

Loads the vectors back out of ``embeddings.sqlite``, encodes a query with the BGE
query instruction, and returns dense top-k. This mirrors how the app's Tier-1
retriever scores (cosine == dot product on L2-normalized vectors), so the report
exercises the real artifact rather than an in-memory copy.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import EMBED_DIM, QUERY_INSTRUCTION


@dataclass
class Neighbor:
    chunk_id: str
    score: float
    resource_type: str
    source: str
    anchor: str
    book: str | None
    chapter: int | None
    verse_start: int | None
    verse_end: int | None
    text: str


def load_matrix(db: Path) -> tuple[np.ndarray, list[dict]]:
    """Load all vectors + chunk metadata from ``embeddings.sqlite``.

    Returns ``(matrix (n, dim) float32, meta list aligned by row)``.
    """
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT e.chunk_id, e.vector, c.resource_type, c.source, c.anchor, "
            "c.book, c.chapter, c.verse_start, c.verse_end, c.text "
            "FROM embedding e JOIN chunk c ON c.id = e.chunk_id "
            "ORDER BY e.chunk_id"
        ).fetchall()
    finally:
        conn.close()
    n = len(rows)
    mat = np.empty((n, EMBED_DIM), dtype=np.float32)
    meta: list[dict] = []
    for i, r in enumerate(rows):
        mat[i] = np.frombuffer(r[1], dtype="<f4")
        meta.append({
            "chunk_id": r[0], "resource_type": r[2], "source": r[3],
            "anchor": r[4], "book": r[5], "chapter": r[6],
            "verse_start": r[7], "verse_end": r[8], "text": r[9],
        })
    return mat, meta


def encode_query(model, query: str) -> np.ndarray:
    """Encode a query with the BGE retrieval instruction prefix."""
    from .encode import encode_texts

    vec = encode_texts(model, [QUERY_INSTRUCTION + query])
    return vec[0]


def topk(
    matrix: np.ndarray, meta: list[dict], qvec: np.ndarray, k: int = 5
) -> list[Neighbor]:
    """Dense top-k by cosine (dot product on normalized vectors)."""
    scores = matrix @ qvec
    idx = np.argsort(-scores)[:k]
    out: list[Neighbor] = []
    for rank_i in idx:
        m = meta[int(rank_i)]
        out.append(Neighbor(
            chunk_id=m["chunk_id"], score=float(scores[int(rank_i)]),
            resource_type=m["resource_type"], source=m["source"],
            anchor=m["anchor"], book=m["book"], chapter=m["chapter"],
            verse_start=m["verse_start"], verse_end=m["verse_end"],
            text=m["text"],
        ))
    return out
