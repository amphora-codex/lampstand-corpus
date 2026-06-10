"""Source snapshots — fetch upstream files once, checksum them, and record
provenance so builds never depend on live URLs (spec/CLAUDE.md pipeline rule 1).

Only canonical, public-domain sources (see README). A snapshot is the unit of
reproducibility: the SHA-256 recorded here is what every derived chunk cites."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BibleSource:
    """A canonical Bible snapshot definition (P1)."""

    id: str            # 'bsb','kjv','asv','web'
    name: str
    url: str           # canonical upstream download (USFM zip)
    filename: str      # snapshot file written under sources/<id>/
    version: str       # upstream edition identifier
    license: str

    @property
    def dest(self) -> Path:
        return SOURCES_DIR / self.id / self.filename


# Decision (flagged for the architect): we COMMIT the raw USFM zip snapshots into
# sources/ via git-lfs (.gitattributes already tracks sources/**/*.zip). CLAUDE.md
# pipeline rule 1 wants reproducible builds that never depend on live URLs; the
# committed snapshot's SHA-256 is the unit of reproducibility. All four are
# public-domain / CC0, so committing the bytes is permitted.
BIBLE_SOURCES: dict[str, BibleSource] = {
    "bsb": BibleSource(
        id="bsb",
        name="Berean Standard Bible",
        url="https://bereanbible.com/bsb_usfm.zip",
        filename="bsb_usfm.zip",
        version="bereanbible.com USFM (snapshot 2026-06-10)",
        license="CC0 / public domain (bereanbible.com)",
    ),
    "kjv": BibleSource(
        id="kjv",
        name="King James Version",
        url="https://ebible.org/Scriptures/eng-kjv2006_usfm.zip",
        filename="eng-kjv2006_usfm.zip",
        version="eng-kjv2006 (eBible.org USFM)",
        license="Public domain (KJV; UK Crown patent printing restriction only)",
    ),
    "asv": BibleSource(
        id="asv",
        name="American Standard Version",
        url="https://ebible.org/Scriptures/eng-asv_usfm.zip",
        filename="eng-asv_usfm.zip",
        version="eng-asv 1901 (eBible.org USFM)",
        license="Public domain",
    ),
    "web": BibleSource(
        id="web",
        name="World English Bible",
        url="https://ebible.org/Scriptures/eng-web_usfm.zip",
        filename="eng-web_usfm.zip",
        version="eng-web 2020 stable (eBible.org USFM)",
        license="Public domain",
    ),
}


def snapshot_bibles(retrieved: str, *, force: bool = False) -> dict:
    """Download (if absent) each Bible source and write sources/manifest.json.

    Idempotent: existing snapshots are checksummed in place unless ``force``.
    Returns the manifest dict. ``retrieved`` is the ISO fetch date recorded as
    provenance for every snapshot in a single run.
    """
    manifest: dict = {"retrieved": retrieved, "sources": {}}
    for src in BIBLE_SOURCES.values():
        if force or not src.dest.exists():
            checksum = fetch(src.url, src.dest)
        else:
            checksum = sha256_of(src.dest)
        manifest["sources"][src.id] = {
            "name": src.name,
            "url": src.url,
            "file": str(src.dest.relative_to(SOURCES_DIR.parent)),
            "version": src.version,
            "license": src.license,
            "retrieved": retrieved,
            "sha256": checksum,
        }
    manifest_path = SOURCES_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


# TODO(P2+): CCEL pages, OpenScriptures repos, TSK snapshot definitions land in
# later phases. P1 covers the four Bibles only.
