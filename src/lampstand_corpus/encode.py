"""BGE-small encoder + model provenance (the heavy P6 step).

Isolated from ``embeddings.py`` so the chunk-extraction / tokenizer logic (and its
tests) import without pulling in torch. ``sentence-transformers`` + ``torch`` are
the ``[embeddings]`` extra — only this module imports them, lazily.

Determinism (CLAUDE.md pipeline rule 6): the committed artifact is encoded on
**CPU** with fixed seeds and torch's deterministic flags set. MPS may be used for
trial runs but the validation report must reflect the CPU build. We verify
bit-for-bit reproducibility by encoding twice; if that ever fails we fall back to
asserting cosine-identity within tolerance and FLAG the deviation rather than
silently accepting it.

Incremental re-encode (P6, corpus-update path)
----------------------------------------------
A full corpus re-encode is ~4h on CPU, but most corpus updates touch a small
fraction of chunks (the Strong's CC0/PD swap, for instance, re-texts only the
~14k Strong's lexicon chunks). :func:`encode_chunks_incremental` reuses existing
vectors out of a prior ``embeddings.sqlite`` wherever a chunk is *unchanged*, and
re-encodes only the changed/new chunks. Unchanged is decided by the chunk **id**,
which is content-addressed over ``(resource_type, source, anchor, text_checksum)``
— so a reused vector is provably the embedding of byte-identical text under the
same model revision (Determinism policy: architect-decided Option A —
cosine-tolerance acceptance, locked by model revision + input checksums + artifact
SHA). Removed chunks are simply absent from the new chunk list, so their vectors
never carry forward. Reuse is gated on the model revision recorded in the prior
DB matching the pinned :data:`MODEL_REVISION`; a mismatch forces a full re-encode.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embeddings import EMBED_DIM, MODEL_NAME, MODEL_REVISION, Chunk

# Local gitignored model cache (kept inside the repo so provenance is
# self-contained; weights are never committed — see .gitignore models/).
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = REPO_ROOT / "models"

# Encoding batch size. Affects only wall-time, not the resulting vectors (BGE
# encodes each text independently; batching does not change per-text output on a
# fixed backend).
BATCH_SIZE = 64


def _set_deterministic_cpu() -> None:
    """Pin every knob that could perturb the committed artifact."""
    import torch

    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Single CPU thread removes nondeterministic float reduction order from
    # intra-op parallelism — the price of bit-for-bit reproducibility.
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def _snapshot_dir() -> Path:
    d = (MODEL_CACHE / f"models--{MODEL_NAME.replace('/', '--')}"
         / "snapshots" / MODEL_REVISION)
    if not d.exists():
        raise FileNotFoundError(
            f"Model snapshot not found at {d}. Download it first with the pinned "
            f"revision (the embeddings CLI does this automatically)."
        )
    return d


def file_hashes() -> dict[str, str]:
    """SHA-256 of every weight/config file in the pinned snapshot (provenance).

    Hashes are computed over the real files (HF stores them as symlinks into
    ``blobs/``); the keys are snapshot-relative paths so the manifest is portable.
    """
    snap = _snapshot_dir()
    out: dict[str, str] = {}
    for p in sorted(snap.rglob("*")):
        if p.is_dir():
            continue
        real = p.resolve()
        h = hashlib.sha256(real.read_bytes()).hexdigest()
        out[str(p.relative_to(snap))] = h
    return out


def model_provenance() -> dict:
    """Model identity block recorded in the embeddings manifest."""
    hashes = file_hashes()
    # A single combined hash over the sorted per-file hashes — one number that
    # changes if any input weight changes.
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(hashes.items())).encode()
    ).hexdigest()
    return {
        "name": MODEL_NAME,
        "revision": MODEL_REVISION,
        "dim": EMBED_DIM,
        "combined_sha256": combined,
        "files": hashes,
    }


def load_model(device: str = "cpu"):
    """Load BGE-small from the pinned local snapshot (offline, no live fetch)."""
    from sentence_transformers import SentenceTransformer

    if device == "cpu":
        _set_deterministic_cpu()
    snap = _snapshot_dir()
    model = SentenceTransformer(str(snap), device=device)
    model.eval()
    return model


def encode_texts(model, texts: list[str], *, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Encode passages to L2-normalized float32 vectors (BGE convention).

    Returns an ``(n, EMBED_DIM)`` float32 array. Normalization makes cosine
    similarity a plain dot product downstream.
    """
    import torch

    with torch.no_grad():
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return np.ascontiguousarray(vecs.astype(np.float32))


def encode_chunks(
    chunks: list[Chunk], *, device: str = "cpu", batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """Encode chunk INDEX texts (header + body) in order.

    Returns ``(len(chunks), EMBED_DIM)`` float32."""
    model = load_model(device=device)
    texts = [c.index_text for c in chunks]
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    return encode_texts(model, texts, batch_size=batch_size)


# --- Incremental re-encode ---------------------------------------------------
@dataclass
class IncrementalResult:
    """Outcome of an incremental encode: vectors (aligned to ``chunks``) + stats."""

    vectors: np.ndarray
    n_total: int = 0
    n_reused: int = 0
    n_encoded: int = 0
    n_dropped: int = 0          # chunk ids in the prior DB not in the new set
    prior_model_revision: str = ""
    full_reencode: bool = False  # True if reuse was impossible (no prior / mismatch)
    notes: list[str] = field(default_factory=list)


def _load_prior_vectors(db: Path) -> tuple[dict[str, bytes], str]:
    """Read ``{chunk_id: vector_blob}`` + the model revision from a prior DB.

    Returns ``({}, "")`` when the DB is absent or unreadable. The blobs are kept as
    raw little-endian float32 bytes so a reused vector is byte-identical to what was
    written before (no decode/re-encode round-trip that could perturb the artifact).
    """
    if not db.exists():
        return {}, ""
    conn = sqlite3.connect(db)
    try:
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        except sqlite3.Error:
            meta = {}
        revision = meta.get("model_revision", "")
        vectors: dict[str, bytes] = {}
        for cid, dim, blob in conn.execute(
                "SELECT chunk_id, dim, vector FROM embedding"):
            if dim == EMBED_DIM and isinstance(blob, (bytes, bytearray)):
                vectors[cid] = bytes(blob)
        return vectors, revision
    except sqlite3.Error:
        return {}, ""
    finally:
        conn.close()


def encode_chunks_incremental(
    chunks: list[Chunk],
    prior_db: Path,
    *,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
) -> IncrementalResult:
    """Encode ``chunks`` reusing unchanged vectors from ``prior_db``.

    A chunk is *unchanged* iff its content-addressed ``id`` is present in the prior
    embeddings DB (the id folds in the text checksum, so an id match guarantees the
    text is byte-identical). Such vectors are copied verbatim; only the changed/new
    chunks are sent to the encoder. Returns vectors aligned to ``chunks`` (row i is
    chunks[i]'s vector) plus reuse/encode/drop counts for the report.

    Reuse is disabled (full re-encode) when the prior DB is missing or its recorded
    model revision differs from the pinned :data:`MODEL_REVISION` — a different
    model means the old vectors live in a different space and must not be mixed in.
    """
    n = len(chunks)
    out = np.empty((n, EMBED_DIM), dtype=np.float32)
    res = IncrementalResult(vectors=out, n_total=n)

    prior, prior_rev = _load_prior_vectors(prior_db)
    res.prior_model_revision = prior_rev

    reuse_ok = bool(prior) and prior_rev == MODEL_REVISION
    if prior and not reuse_ok:
        res.full_reencode = True
        res.notes.append(
            f"prior embeddings model revision {prior_rev!r} != pinned "
            f"{MODEL_REVISION!r}; reuse disabled, full re-encode"
        )
    elif not prior:
        res.full_reencode = True
        res.notes.append("no prior embeddings.sqlite to reuse; full encode")

    new_ids = {c.id for c in chunks}
    if reuse_ok:
        res.n_dropped = sum(1 for cid in prior if cid not in new_ids)

    # Partition: reuse where the id is in the prior DB; encode the rest.
    to_encode_idx: list[int] = []
    for i, c in enumerate(chunks):
        blob = prior.get(c.id) if reuse_ok else None
        if blob is not None:
            out[i] = np.frombuffer(blob, dtype="<f4")
            res.n_reused += 1
        else:
            to_encode_idx.append(i)

    res.n_encoded = len(to_encode_idx)
    if to_encode_idx:
        model = load_model(device=device)
        texts = [chunks[i].index_text for i in to_encode_idx]
        encoded = encode_texts(model, texts, batch_size=batch_size)
        for row, i in enumerate(to_encode_idx):
            out[i] = encoded[row]
    return res
