"""Reproducibility: the same snapshots must produce a bit-for-bit identical DB.

Skipped when snapshots aren't present. Builds twice into temp paths and compares
SHA-256. No timestamps or unfixed ordering may leak into the output.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "sources" / "manifest.json"
COMMENTARIES_MANIFEST = (
    REPO_ROOT / "sources" / "commentaries" / "manifest.json"
)
CONFESSIONS_MANIFEST = (
    REPO_ROOT / "sources" / "confessions" / "manifest.json"
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@pytest.mark.skipif(
    not MANIFEST.exists(), reason="snapshots not present; run `cli snapshot` first"
)
def test_build_is_deterministic(tmp_path):
    from lampstand_corpus.build import write_bibles
    from lampstand_corpus.cli import normalize_all

    parsed, provenance = normalize_all()
    a = tmp_path / "a.sqlite"
    b = tmp_path / "b.sqlite"
    write_bibles(parsed, provenance, a)
    write_bibles(parsed, provenance, b)
    assert _sha(a) == _sha(b)


@pytest.mark.skipif(
    not COMMENTARIES_MANIFEST.exists(),
    reason="commentary snapshots not present; run `cli snapshot-commentaries` first",
)
def test_commentaries_build_is_deterministic(tmp_path):
    from lampstand_corpus.build_commentaries import write_commentaries
    from lampstand_corpus.cli import normalize_commentaries

    parsed = normalize_commentaries()
    a = tmp_path / "a.sqlite"
    b = tmp_path / "b.sqlite"
    write_commentaries(parsed, a)
    write_commentaries(parsed, b)
    assert _sha(a) == _sha(b)


@pytest.mark.skipif(
    not CONFESSIONS_MANIFEST.exists(),
    reason="confession snapshots not present; run `cli snapshot-confessions` first",
)
def test_confessions_build_is_deterministic(tmp_path):
    from lampstand_corpus.build_confessions import write_confessions
    from lampstand_corpus.cli import normalize_confessions

    parsed = normalize_confessions()
    a = tmp_path / "a.sqlite"
    b = tmp_path / "b.sqlite"
    write_confessions(parsed, a)
    write_confessions(parsed, b)
    assert _sha(a) == _sha(b)
