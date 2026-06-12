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
"""

from __future__ import annotations

import hashlib
import os
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
    """Encode chunk texts in order; returns ``(len(chunks), EMBED_DIM)`` float32."""
    model = load_model(device=device)
    texts = [c.text for c in chunks]
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    return encode_texts(model, texts, batch_size=batch_size)
