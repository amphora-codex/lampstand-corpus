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


def snapshot_confessions(retrieved: str, *, force: bool = False) -> dict:
    """Download (if absent) each confession source and write its manifest.

    Idempotent like :func:`snapshot_bibles`. Sources are canonical public-domain
    confession texts (CCEL ThML, the westminster-json MIT repo, the CC0
    ParticularBaptists 1689 repo, and the Wikisource 1840 Belgic), committed under
    ``sources/confessions/<id>/`` via git-lfs. The underlying confessions are all
    PD; the upstream repo license is recorded separately. Returns the manifest
    dict and writes ``sources/confessions/manifest.json``.
    """
    # Imported here to avoid a circular import (confessions imports SOURCES_DIR).
    from .confessions import CONFESSION_SOURCES, CONFESSIONS_DIR

    manifest: dict = {"retrieved": retrieved, "sources": {}}
    for src in CONFESSION_SOURCES.values():
        if force or not src.dest.exists():
            checksum = fetch(src.url, src.dest)
        else:
            checksum = sha256_of(src.dest)
        entry: dict = {
            "name": src.name,
            "shortcode": src.shortcode,
            "url": src.url,
            "file": str(src.dest.relative_to(SOURCES_DIR.parent)),
            "version": src.version,
            "license": src.license,
            "retrieved": retrieved,
            "sha256": checksum,
        }
        if src.repo_license:
            entry["repo_license"] = src.repo_license
        # Auxiliary file (e.g. the 1689 per-paragraph proof-text markdown).
        aux = src.aux_dest
        if aux is not None:
            if force or not aux.exists():
                aux_checksum = fetch(src.aux_url, aux)
            else:
                aux_checksum = sha256_of(aux)
            entry["aux_file"] = str(aux.relative_to(SOURCES_DIR.parent))
            entry["aux_url"] = src.aux_url
            entry["aux_sha256"] = aux_checksum
        # Amendment source (e.g. the CCEL 1788 American-revision WCF, used to
        # populate the verbatim revised wording of the six amended loci).
        amend = src.amend_dest
        if amend is not None:
            if force or not amend.exists():
                amend_checksum = fetch(src.amend_url, amend)
            else:
                amend_checksum = sha256_of(amend)
            entry["amend_file"] = str(amend.relative_to(SOURCES_DIR.parent))
            entry["amend_url"] = src.amend_url
            entry["amend_sha256"] = amend_checksum
            if src.amend_license:
                entry["amend_license"] = src.amend_license
            entry["amend_note"] = (
                "1788 American-revision verbatim text source for the six amended "
                "loci; original numbering recovered from parenthetical titles; "
                "later [PCUS]/[UPCUSA] denominational brackets removed for the 1788 "
                "base reading (flagged)"
            )
        # Validation-only cross-check snapshot (e.g. Wikisource Burges-1646 for the
        # WCF prose diff) — recorded for provenance, not used as primary text.
        xref = src.xref_dest
        if xref is not None:
            if force or not xref.exists():
                xref_checksum = fetch(src.xref_url, xref)
            else:
                xref_checksum = sha256_of(xref)
            entry["xref_file"] = str(xref.relative_to(SOURCES_DIR.parent))
            entry["xref_url"] = src.xref_url
            entry["xref_sha256"] = xref_checksum
            entry["xref_note"] = "validation cross-check only; not a primary source"
        manifest["sources"][src.id] = entry
    manifest_path = CONFESSIONS_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def snapshot_commentaries(retrieved: str, *, force: bool = False) -> dict:
    """Download (if absent) each commentary VOLUME and write its manifest.

    Commentaries are multi-volume on CCEL (Henry mhc1-6, Calvin's many calcomNN
    files, JFB one file), so the manifest is keyed ``<commentator>/<volume>``.
    Idempotent like the others; sources are canonical CCEL ThML (public domain),
    committed under ``sources/commentaries/<id>/`` via git-lfs. Spurgeon is NOT
    listed (the CCEL Treasury of David is a page-image scan with no machine text —
    flagged-and-skipped for v1; see commentaries.py), nor is Gill (deferred v1.1).
    """
    from .commentaries import COMMENTARIES_DIR, COMMENTARY_SOURCES

    manifest: dict = {"retrieved": retrieved, "sources": {}}
    for src in COMMENTARY_SOURCES.values():
        vols: dict = {}
        for volume in src.volumes:
            dest = src.dest(volume)
            if force or not dest.exists():
                checksum = fetch(src.url(volume), dest, timeout=180)
            else:
                checksum = sha256_of(dest)
            vols[volume] = {
                "url": src.url(volume),
                "file": str(dest.relative_to(SOURCES_DIR.parent)),
                "sha256": checksum,
            }
        manifest["sources"][src.id] = {
            "name": src.name,
            "shortcode": src.shortcode,
            "author": src.author,
            "work": src.work,
            "version": src.version,
            "license": src.license,
            "retrieved": retrieved,
            "volumes": vols,
        }
    manifest_path = COMMENTARIES_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def snapshot_spurgeon(retrieved: str, *, force: bool = False) -> dict:
    """Download (if absent) each Treasury-of-David volume's djvu OCR.

    Six volumes come from the Internet Archive ``*spurgoog`` Google scans (public
    domain). The seventh (tod5, Psalms 104-118) has NO ``*spurgoog`` item, so it is
    gap-filled from an alternate PD scan (``treasuryofdavidc0005spur``, vol. 5,
    1882) — surfaced in the manifest with its exact identifier. Snapshots are
    committed under ``sources/commentaries/spurgeon/`` via git-lfs. Returns the
    manifest dict and writes ``sources/commentaries/spurgeon/manifest.json``.
    """
    from .spurgeon import (
        MISSING_VOLUME,
        REJECTED_DUPLICATES,
        SPURGEON_DIR,
        SPURGEON_SOURCE,
        SPURGEON_VOLUMES,
        TOD5_SOURCE_NOTE,
    )

    vols: dict = {}
    for v in SPURGEON_VOLUMES:
        if force or not v.dest.exists():
            checksum = fetch(v.url, v.dest, timeout=300)
        else:
            checksum = sha256_of(v.dest)
        vols[v.stem] = {
            "identifier": v.identifier,
            "url": v.url,
            "file": str(v.dest.relative_to(SOURCES_DIR.parent)),
            "psalm_first": v.psalm_first,
            "psalm_last": v.psalm_last,
            "sha256": checksum,
        }
    lo, hi, why = MISSING_VOLUME
    manifest: dict = {
        "retrieved": retrieved,
        "sources": {
            SPURGEON_SOURCE.id: {
                "name": SPURGEON_SOURCE.name,
                "shortcode": SPURGEON_SOURCE.shortcode,
                "author": SPURGEON_SOURCE.author,
                "work": SPURGEON_SOURCE.work,
                "version": SPURGEON_SOURCE.version,
                "license": SPURGEON_SOURCE.license,
                "retrieved": retrieved,
                "volumes": vols,
                "missing_volume": {"psalm_first": lo, "psalm_last": hi, "note": why},
                "tod5_gap_fill": TOD5_SOURCE_NOTE,
                "rejected_duplicates": REJECTED_DUPLICATES,
            }
        },
    }
    manifest_path = SPURGEON_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def snapshot_lexicons(retrieved: str, *, force: bool = False) -> dict:
    """Download (if absent) each lexicon dictionary source + write its manifest.

    Strong's Greek/Hebrew ``.js`` dictionaries and the BDB XML (with its companion
    ``LexicalIndex.xml`` aux file, which carries the Strong's<->BDB linkage). All
    canonical OpenScriptures sources; the underlying dictionary text is PD, the
    OpenScriptures editions are CC-BY-SA (Strong's) / CC-BY-4.0 (BDB) — both
    licenses recorded. Snapshots commit under ``sources/lexicons/<id>/`` via
    git-lfs. Idempotent like the other snapshot functions.
    """
    from .lexicons import (
        BDB_INDEX_FILENAME,
        BDB_INDEX_URL,
        LEXICON_SOURCES,
        LEXICONS_DIR,
    )

    manifest: dict = {"retrieved": retrieved, "sources": {}}
    for src in LEXICON_SOURCES.values():
        if force or not src.dest.exists():
            checksum = fetch(src.url, src.dest, timeout=180)
        else:
            checksum = sha256_of(src.dest)
        entry: dict = {
            "name": src.name,
            "language": src.language,
            "lexicon": src.lexicon,
            "url": src.url,
            "file": str(src.dest.relative_to(SOURCES_DIR.parent)),
            "version": src.version,
            "license": src.license,
            "text_license": src.text_license,
            "retrieved": retrieved,
            "sha256": checksum,
        }
        if src.id == "bdb":
            aux_dest = LEXICONS_DIR / src.id / BDB_INDEX_FILENAME
            if force or not aux_dest.exists():
                aux_checksum = fetch(BDB_INDEX_URL, aux_dest, timeout=180)
            else:
                aux_checksum = sha256_of(aux_dest)
            entry["aux_file"] = str(aux_dest.relative_to(SOURCES_DIR.parent))
            entry["aux_url"] = BDB_INDEX_URL
            entry["aux_sha256"] = aux_checksum
            entry["aux_note"] = (
                "LexicalIndex.xml — Strong's<->BDB entry-id linkage; required to "
                "key BDB entries by Strong's number"
            )
        manifest["sources"][src.id] = entry
    manifest_path = LEXICONS_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def snapshot_tagged_text(retrieved: str, *, force: bool = False) -> dict:
    """Download (if absent) the Strong's-tagged original-language text (P4b).

    Ingested: OSHB (morphhb) Hebrew, 39 ``wlc/*.xml`` files, CC-BY-4.0.
    Snapshot-only + FLAGGED: MorphGNT/SBLGNT Greek (no Strong's tagging; SBLGNT
    EULA on the text). Both recorded in ``sources/lexicons/tagged_manifest.json``.
    """
    from .lexicons import (
        FLAGGED_TEXT_SOURCES,
        LEXICONS_DIR,
        TAGGED_TEXT_SOURCES,
    )

    manifest: dict = {"retrieved": retrieved, "ingested": {}, "flagged": {}}
    for src in TAGGED_TEXT_SOURCES.values():
        files: dict = {}
        for stem in src.book_stems:
            dest = src.dest(stem)
            if force or not dest.exists():
                checksum = fetch(src.url(stem), dest, timeout=180)
            else:
                checksum = sha256_of(dest)
            files[stem] = {
                "url": src.url(stem),
                "file": str(dest.relative_to(SOURCES_DIR.parent)),
                "sha256": checksum,
            }
        manifest["ingested"][src.id] = {
            "name": src.name,
            "language": src.language,
            "version": src.version,
            "license": src.license,
            "attribution": src.attribution,
            "retrieved": retrieved,
            "files": files,
        }
    for src in FLAGGED_TEXT_SOURCES.values():
        files = {}
        for fname in src.files:
            dest = src.dest(fname)
            if force or not dest.exists():
                checksum = fetch(src.url(fname), dest, timeout=180)
            else:
                checksum = sha256_of(dest)
            files[fname] = {
                "url": src.url(fname),
                "file": str(dest.relative_to(SOURCES_DIR.parent)),
                "sha256": checksum,
            }
        manifest["flagged"][src.id] = {
            "name": src.name,
            "version": src.version,
            "license": src.license,
            "flag": src.flag,
            "retrieved": retrieved,
            "files": files,
        }
    manifest_path = LEXICONS_DIR / "tagged_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def snapshot_stepbible(retrieved: str, *, force: bool = False) -> dict:
    """Download (if absent) the STEPBible TAGNT + TBESG Greek sources (P4b).

    Source: github.com/STEPBible/STEPBible-Data, CC-BY 4.0. The required
    attribution ("STEP Bible, www.STEPBible.org") and the license are recorded in
    ``sources/lexicons/stepbible/manifest.json``. Two TAGNT data files (Mat-Jhn,
    Act-Rev — split only for GitHub size limits) cover the 27 NT books; one TBESG
    file is the Greek lexicon. Snapshots commit under
    ``sources/lexicons/stepbible/<id>/`` via git-lfs. Idempotent.
    """
    from .lexicons import (
        STEPBIBLE_ATTRIBUTION,
        STEPBIBLE_DIR,
        STEPBIBLE_LICENSE,
        STEPBIBLE_SOURCES,
        STEPBIBLE_VERSION,
    )

    manifest: dict = {
        "retrieved": retrieved,
        "license": STEPBIBLE_LICENSE,
        "attribution": STEPBIBLE_ATTRIBUTION,
        "version": STEPBIBLE_VERSION,
        "sources": {},
    }
    for src in STEPBIBLE_SOURCES.values():
        files: dict = {}
        for fname in src.files:
            dest = src.dest(fname)
            if force or not dest.exists():
                checksum = fetch(src.url(fname), dest, timeout=300)
            else:
                checksum = sha256_of(dest)
            files[fname] = {
                "url": src.url(fname),
                "file": str(dest.relative_to(SOURCES_DIR.parent)),
                "sha256": checksum,
            }
        manifest["sources"][src.id] = {
            "name": src.name,
            "files": files,
        }
    manifest_path = STEPBIBLE_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


# TODO(P5+): TSK cross-ref snapshot definitions land in a later phase.
