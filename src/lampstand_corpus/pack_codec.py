"""Pack-diet codecs: varint-delta posting blobs + int8 vector quantization.

The v1 packs stored one SQL row per BM25 posting, repeating a 28-char TEXT
chunk id in the row, its (term_id, chunk_id) PK index, AND idx_posting_chunk —
~289 MB of id strings per copy, ~3 copies, dwarfing the ~tens of MB of actual
information (df/tf/gaps). The v2 packs store one BLOB per term:

    postings BLOB = [ gap uvarint, tf uvarint ] * doc_freq

where ids are the STABLE INTEGER chunk ids (1-based, assigned by ascending
string chunk id — see ``assign_int_ids``), listed ascending, delta-encoded
(first gap = the first id itself, ids >= 1 so every gap >= 1). Varints are
unsigned LEB128 (7 bits per byte, high bit = continuation) — trivially
implementable in Swift for the app-side reader.

Vector quantization is symmetric scalar int8 with a per-vector scale:

    scale = max(|v|) / 127          (1.0 for an all-zero vector)
    q_i   = clip(rint(v_i / scale), -127, 127)   as int8
    v_i   ≈ q_i * scale

Scoring against a float32 query is ``dot(q, v_int8) * scale`` — bit-identical
to scoring the dequantized float32 vector. ``np.rint`` (round-half-to-even) is
deterministic, so the same float32 vectors always produce the same bytes.

Everything here is pure and dependency-light so both the packaging layer and
the eval harness share one implementation.
"""

from __future__ import annotations

import numpy as np


# --- unsigned LEB128 varints ---------------------------------------------------
def write_uvarint(buf: bytearray, n: int) -> None:
    """Append ``n`` (>= 0) to ``buf`` as an unsigned LEB128 varint."""
    if n < 0:
        raise ValueError(f"uvarint cannot encode negative value {n}")
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def read_uvarint(blob: bytes, pos: int) -> tuple[int, int]:
    """Read one uvarint from ``blob`` at ``pos``; returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        if pos >= len(blob):
            raise ValueError("truncated uvarint")
        b = blob[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


# --- zigzag (signed values inside uvarint streams) --------------------------------
def zigzag_encode(n: int) -> int:
    """Map a signed int to unsigned for uvarint storage (0,-1,1,-2 → 0,1,2,3)."""
    return (n << 1) if n >= 0 else ((-n) << 1) - 1


def zigzag_decode(u: int) -> int:
    """Inverse of :func:`zigzag_encode`."""
    return (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)


# --- posting blobs ---------------------------------------------------------------
def encode_postings(postings: list[tuple[int, int]]) -> bytes:
    """Encode ``[(int_chunk_id, term_freq), ...]`` (ids ascending, >= 1).

    Layout: for each posting, ``gap`` uvarint (id minus previous id; the first
    gap is the first id itself) then ``tf`` uvarint. Raises on unsorted or
    non-positive ids so a build bug can never produce an undecodable blob.
    """
    buf = bytearray()
    prev = 0
    for cid, tf in postings:
        gap = cid - prev
        if gap <= 0:
            raise ValueError(
                f"posting ids must be strictly ascending and >= 1 (id {cid} "
                f"after {prev})")
        if tf <= 0:
            raise ValueError(f"term_freq must be >= 1 (got {tf} for id {cid})")
        write_uvarint(buf, gap)
        write_uvarint(buf, tf)
        prev = cid
    return bytes(buf)


def decode_postings(blob: bytes) -> list[tuple[int, int]]:
    """Decode a posting blob back to ``[(int_chunk_id, term_freq), ...]``."""
    out: list[tuple[int, int]] = []
    pos = 0
    cid = 0
    n = len(blob)
    while pos < n:
        gap, pos = read_uvarint(blob, pos)
        tf, pos = read_uvarint(blob, pos)
        cid += gap
        out.append((cid, tf))
    return out


# --- stable integer chunk ids ------------------------------------------------------
def assign_int_ids(string_ids: list[str]) -> dict[str, int]:
    """Stable 1-based integer ids by ASCENDING string chunk id.

    The string ids are content-addressed (sha over resource/source/anchor/text
    checksum), so for a given chunk set the mapping is fully deterministic:
    same inputs → same mapping. Int ids are a PER-CORPUS-VERSION artifact —
    adding or removing any chunk renumbers — so cross-version identity stays
    with ``string_id``; the int id exists only to make postings/vector keys
    cheap inside one pack set.
    """
    return {sid: i for i, sid in enumerate(sorted(string_ids), start=1)}


# --- int8 vector quantization ---------------------------------------------------------
def quantize_int8(vec: np.ndarray) -> tuple[bytes, float]:
    """Symmetric scalar int8 quantization with a per-vector scale."""
    v = np.asarray(vec, dtype=np.float32)
    m = float(np.max(np.abs(v))) if v.size else 0.0
    scale = (m / 127.0) if m > 0 else 1.0
    q = np.clip(np.rint(v / scale), -127, 127).astype(np.int8)
    return q.tobytes(), scale


def dequantize_int8(blob: bytes, scale: float) -> np.ndarray:
    """Reconstruct the float32 vector from an int8 blob + its scale."""
    return np.frombuffer(blob, dtype=np.int8).astype(np.float32) * np.float32(scale)


def quantize_roundtrip_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row-wise int8 quantize→dequantize a matrix (the eval's quality probe).

    Returns the float32 matrix the app would effectively score against when
    the vectors pack is int8 — mathematically identical to scoring the stored
    int8 bytes with the per-vector scale.
    """
    m = np.max(np.abs(matrix), axis=1, keepdims=True)
    scale = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(matrix / scale), -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale
