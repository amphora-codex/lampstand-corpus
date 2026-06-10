"""Source snapshots — fetch upstream files once, checksum them, and record
provenance so builds never depend on live URLs (spec/CLAUDE.md pipeline rule 1).

Only canonical, public-domain sources (see README). A snapshot is the unit of
reproducibility: the SHA-256 recorded here is what every derived chunk cites."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

SOURCES_DIR = Path(__file__).resolve().parents[2] / "sources"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, *, timeout: int = 60) -> str:
    """Download `url` to `dest` (idempotent) and return its SHA-256.

    Re-running with an unchanged upstream yields the same checksum — the basis
    for reproducible builds. Callers record (url, retrieved date, checksum) in
    the sources manifest.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return sha256_of(dest)


# TODO(P1): per-source snapshot definitions (BSB/KJV/ASV/WEB USFM, CCEL pages,
# OpenScriptures repos, TSK) + a manifest writer recording provenance.
# Decision flagged for the architect: whether to COMMIT the raw snapshots into
# sources/ (CLAUDE.md pipeline rule 1 — large public-domain text in a public
# repo; consider git-lfs) or keep a manifest of URLs+checksums and fetch on
# build. Default leans toward committing snapshots for true reproducibility.
