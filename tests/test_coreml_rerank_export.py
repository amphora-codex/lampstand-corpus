"""Reranker Core ML export — pure-logic tests (no torch, no coremltools, no net).

Covers the tie-aware inversion counter, the relative-error parity semantics, the
deterministic parity-pair set, and the manifest pack/acknowledgement builders.
The actual trace/convert/parity-gate run is exercised by the CLI export (which
needs the [coreml]+[rerank] extras and a cached model), not here.
"""

from __future__ import annotations

from pathlib import Path

from lampstand_corpus import coreml_rerank_export as rx


def test_count_inversions_ignores_tied_pairs():
    # a = ground truth (PyTorch). Index 2 is the clear top; indices 0,1 are a
    # near-tie. b flips the tied (0,1) pair AND drops index 2 below index 0.
    a = [-11.3390, -11.3395, 5.0]   # order: 2 > 0 > 1
    b = [-11.4060, -11.4140, -20.0]  # order: 0 > 1 > 2
    #   (0,1): a has 0>1, b has 0>1 -> NOT flipped ... need a genuine tied flip
    inv, tied = rx._count_inversions(a, b)
    assert tied == 0
    # index 2 fell from top (a) to bottom (b): (0,2) and (1,2) are real inversions.
    assert inv == 2


def test_count_inversions_tied_flip_is_not_gated():
    # a: indices 0 and 1 tie within TIE_EPS (gap 0.0005). b flips their order.
    # That flip is a tied_reorder, not a gated inversion.
    a = [-11.3390, -11.3395]   # a[0] > a[1] by 0.0005 (<= TIE_EPS)
    b = [-11.4140, -11.4060]   # b[0] < b[1] -> flipped
    inv, tied = rx._count_inversions(a, b)
    assert inv == 0 and tied == 1


def test_count_inversions_zero_when_order_preserved():
    a = [3.0, 1.0, -5.0]
    b = [2.9, 0.8, -5.2]
    inv, tied = rx._count_inversions(a, b)
    assert inv == 0 and tied == 0


def test_parity_pairs_are_distinct_and_sized():
    pairs = rx._parity_pairs()
    assert len(pairs) == rx.PARITY_SAMPLE_N
    # No manufactured duplicates (which would create degenerate ties).
    assert len(set(pairs)) == len(pairs)


def test_parity_result_passes_on_small_rel_err_and_no_real_inversions():
    res = rx.RerankParityResult(
        n=30, max_abs_err=0.098, mean_abs_err=0.06, max_rel_err=0.008,
        inversions=0, tied_reorders=2)
    # Large ABS error but tiny RELATIVE error + zero real inversions -> PASS.
    assert res.passed


def test_parity_result_fails_on_real_inversion():
    res = rx.RerankParityResult(
        n=30, max_abs_err=0.01, mean_abs_err=0.005, max_rel_err=0.001,
        inversions=1, tied_reorders=0)
    assert not res.passed


def test_parity_result_fails_on_high_rel_err():
    res = rx.RerankParityResult(
        n=30, max_abs_err=1.0, mean_abs_err=0.5, max_rel_err=0.10,
        inversions=0, tied_reorders=0)
    assert not res.passed


def test_model_combined_sha256_is_deterministic(tmp_path: Path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text('{"a": 1}', encoding="utf-8")
    (snap / "vocab.txt").write_text("hello\nworld\n", encoding="utf-8")
    h1 = rx.model_combined_sha256(snap)
    h2 = rx.model_combined_sha256(snap)
    assert h1 == h2 and len(h1) == 64
    # Any content change moves the hash.
    (snap / "vocab.txt").write_text("hello\nworld\n!\n", encoding="utf-8")
    assert rx.model_combined_sha256(snap) != h1


def test_build_reranker_pack_shape(tmp_path: Path):
    from lampstand_corpus.package import build_reranker_pack

    ml = tmp_path / "Reranker.mlpackage"
    ml.mkdir()
    vocab = tmp_path / "reranker_vocab.txt"
    vocab.write_text("a\nb\n", encoding="utf-8")
    pack = build_reranker_pack(
        mlpackage_path=ml, mlpackage_tree_sha256="deadbeef", mlpackage_bytes=1000,
        vocab_path=vocab, vocab_sha256="cafef00d",
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_revision="rev", model_combined_sha256="abc",
        license_id="apache-2.0", precision="float16", seq_len="RangeDim(1,512)",
        max_pair_tokens=192)
    assert pack["delivery"] == "app-binary"
    assert pack["max_pair_tokens"] == 192
    names = [f["name"] for f in pack["files"]]
    assert names == ["Reranker.mlpackage", "reranker_vocab.txt"]
    assert pack["files"][0]["role"] == "reranker-model"
    assert pack["total_bytes"] == 1000 + vocab.stat().st_size


def test_update_manifest_reranker_and_ack(tmp_path: Path):
    import json
    from collections import OrderedDict

    from lampstand_corpus.package import (
        update_manifest_acknowledgement,
        update_manifest_reranker,
    )

    mf = tmp_path / "corpus_manifest.json"
    mf.write_text(json.dumps({
        "packs": {"models": {"x": 1}},
        "acknowledgements": [{"id": "bsb", "name": "Berean"}],
    }), encoding="utf-8")

    update_manifest_reranker(mf, OrderedDict([("description", "d"), ("files", [])]))
    update_manifest_acknowledgement(mf, OrderedDict([
        ("id", "reranker-model"), ("name", "cross-encoder/ms-marco-MiniLM-L-6-v2")]))
    # Idempotent replace by id.
    update_manifest_acknowledgement(mf, OrderedDict([
        ("id", "reranker-model"), ("name", "updated")]))

    m = json.loads(mf.read_text(encoding="utf-8"))
    assert m["packs"]["reranker"]["description"] == "d"
    assert m["packs"]["models"] == {"x": 1}  # untouched
    acks = {a["id"]: a for a in m["acknowledgements"]}
    assert acks["reranker-model"]["name"] == "updated"  # replaced, not duplicated
    assert len([a for a in m["acknowledgements"] if a["id"] == "reranker-model"]) == 1
    assert "bsb" in acks
