"""Pack-diet codec tests — varint/delta postings + int8 quantization round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from lampstand_corpus import pack_codec as pc


# --- uvarint -----------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, 127, 128, 129, 16383, 16384, 2**21 - 1, 2**32])
def test_uvarint_round_trip(n):
    buf = bytearray()
    pc.write_uvarint(buf, n)
    got, pos = pc.read_uvarint(bytes(buf), 0)
    assert got == n and pos == len(buf)


def test_uvarint_boundary_widths():
    for n, width in ((0, 1), (127, 1), (128, 2), (16383, 2), (16384, 3)):
        buf = bytearray()
        pc.write_uvarint(buf, n)
        assert len(buf) == width, f"{n} should encode in {width} byte(s)"


def test_uvarint_rejects_negative_and_truncation():
    with pytest.raises(ValueError):
        pc.write_uvarint(bytearray(), -1)
    buf = bytearray()
    pc.write_uvarint(buf, 300)  # 2-byte varint
    with pytest.raises(ValueError):
        pc.read_uvarint(bytes(buf[:1]), 0)


# --- posting blobs ---------------------------------------------------------------------
def test_postings_round_trip():
    postings = [(1, 3), (2, 1), (500, 7), (175_442, 1)]
    blob = pc.encode_postings(postings)
    assert pc.decode_postings(blob) == postings


def test_postings_empty_and_single():
    assert pc.decode_postings(pc.encode_postings([])) == []
    assert pc.decode_postings(pc.encode_postings([(42, 9)])) == [(42, 9)]


def test_postings_encoding_is_deterministic_and_compact():
    postings = [(i, 1) for i in range(1, 1001)]  # dense gaps of 1
    a = pc.encode_postings(postings)
    b = pc.encode_postings(postings)
    assert a == b
    # gap=1 and tf=1 are single bytes -> exactly 2 bytes per posting.
    assert len(a) == 2000


def test_postings_reject_unsorted_or_bad_values():
    with pytest.raises(ValueError):
        pc.encode_postings([(5, 1), (5, 1)])   # duplicate id
    with pytest.raises(ValueError):
        pc.encode_postings([(5, 1), (3, 1)])   # descending
    with pytest.raises(ValueError):
        pc.encode_postings([(0, 1)])           # ids are 1-based
    with pytest.raises(ValueError):
        pc.encode_postings([(1, 0)])           # tf must be >= 1


# --- stable int ids ----------------------------------------------------------------------
def test_assign_int_ids_sorted_one_based_and_stable():
    ids = ["scr_b", "con_a", "lex_z"]
    mapping = pc.assign_int_ids(ids)
    assert mapping == {"con_a": 1, "lex_z": 2, "scr_b": 3}
    # Input order must not matter.
    assert pc.assign_int_ids(list(reversed(ids))) == mapping


# --- int8 quantization -------------------------------------------------------------------
def test_quantize_round_trip_error_bound():
    rng = np.random.default_rng(7)
    v = rng.standard_normal(384).astype(np.float32)
    v /= np.linalg.norm(v)
    blob, scale = pc.quantize_int8(v)
    back = pc.dequantize_int8(blob, scale)
    assert len(blob) == 384
    # Max reconstruction error is half a quantization step.
    assert float(np.max(np.abs(back - v))) <= scale / 2 + 1e-7
    # Unit vectors survive with high cosine.
    cos = float(np.dot(back, v) / (np.linalg.norm(back) * np.linalg.norm(v)))
    assert cos > 0.999


def test_quantize_zero_vector_is_safe():
    blob, scale = pc.quantize_int8(np.zeros(8, dtype=np.float32))
    assert scale == 1.0
    assert np.array_equal(pc.dequantize_int8(blob, scale), np.zeros(8, np.float32))


def test_quantize_is_deterministic():
    rng = np.random.default_rng(11)
    v = rng.standard_normal(384).astype(np.float32)
    assert pc.quantize_int8(v) == pc.quantize_int8(v.copy())


def test_quantize_matrix_matches_per_vector_path():
    rng = np.random.default_rng(13)
    mat = rng.standard_normal((5, 384)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    round_tripped = pc.quantize_roundtrip_matrix(mat)
    for i in range(mat.shape[0]):
        blob, scale = pc.quantize_int8(mat[i])
        expected = pc.dequantize_int8(blob, scale)
        assert np.array_equal(round_tripped[i], expected)
