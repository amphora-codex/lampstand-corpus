"""Corpus pipeline CLI: snapshot -> normalize -> build -> validate.

Usage (P1 Bibles):
    python -m lampstand_corpus.cli snapshot   # download Bible sources + manifest
    python -m lampstand_corpus.cli build      # parse, build bibles.sqlite
    python -m lampstand_corpus.cli validate   # parse + write reports/ (no DB)
    python -m lampstand_corpus.cli all        # snapshot(if needed) + build + validate

Usage (P2 confessions & catechisms):
    python -m lampstand_corpus.cli snapshot-confessions  # download CCEL ThML sources
    python -m lampstand_corpus.cli build-confessions     # build confessions.sqlite
    python -m lampstand_corpus.cli validate-confessions  # write report (no DB)

Reads committed snapshots from sources/, writes output/*.sqlite (gitignored) and
reports/*.txt.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

from . import books
from .build import write_bibles
from .build_confessions import write_confessions
from .confessions import CONFESSION_SOURCES, CONFESSIONS_DIR, parse_confession
from .schema import Provenance
from .sources import (
    BIBLE_SOURCES,
    SOURCES_DIR,
    snapshot_bibles,
    snapshot_confessions,
)
from .usfm import ParsedBook, parse_usfm
from .validate import render_report, validate_all
from .validate_confessions import (
    render_confession_report,
    validate_all_confessions,
)

REPO_ROOT = SOURCES_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"
REPORTS_DIR = REPO_ROOT / "reports"
RETRIEVED = "2026-06-10"

# Matches the \id book code at the very start of a USFM member file.
_ID_RE = re.compile(rb"^\\id\s+(\S+)", re.MULTILINE)


def _iter_usfm_members(zip_path: Path):
    """Yield (book_id, text) for each canonical-book USFM member in a zip.

    Non-canon members (Apocrypha, glossary, front/intro matter, html/css) are
    skipped by inspecting each member's \\id code against the 66-book canon.
    """
    with zipfile.ZipFile(zip_path) as zf:
        for info in sorted(zf.namelist()):
            if not info.lower().endswith((".usfm", ".sfm")):
                continue
            raw = zf.read(info)
            m = _ID_RE.search(raw)
            if not m:
                continue
            book_id = m.group(1).decode("ascii", "replace").upper()
            if book_id not in books.CANON:
                continue
            yield book_id, raw.decode("utf-8-sig")


def normalize_all() -> tuple[dict[str, dict[str, ParsedBook]], dict[str, Provenance]]:
    """Parse every committed Bible snapshot into ParsedBooks + provenance."""
    manifest_path = SOURCES_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No sources/manifest.json — run `snapshot` first.", file=sys.stderr)
        sys.exit(2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    parsed: dict[str, dict[str, ParsedBook]] = {}
    provenance: dict[str, Provenance] = {}

    for tid, src in BIBLE_SOURCES.items():
        entry = manifest["sources"][tid]
        prov = Provenance(
            source=tid,
            version=entry["version"],
            license=entry["license"],
            retrieved=entry["retrieved"],
            url=entry["url"],
            checksum=entry["sha256"],
        )
        provenance[tid] = prov
        books_for_tid: dict[str, ParsedBook] = {}
        for _book_id, text in _iter_usfm_members(src.dest):
            pb = parse_usfm(text)
            books_for_tid[pb.book] = pb
        parsed[tid] = books_for_tid
        print(f"  {tid}: parsed {len(books_for_tid)} books "
              f"({sum(len(b.verses) for b in books_for_tid.values())} verses)")
    return parsed, provenance


def cmd_snapshot() -> None:
    print("Snapshotting Bible sources -> sources/ + manifest.json")
    manifest = snapshot_bibles(RETRIEVED)
    for tid, e in sorted(manifest["sources"].items()):
        print(f"  {tid}: {e['sha256']}  {e['file']}")


def cmd_build() -> None:
    print("Normalizing snapshots...")
    parsed, provenance = normalize_all()
    out = OUTPUT_DIR / "bibles.sqlite"
    print(f"Building {out} ...")
    write_bibles(parsed, provenance, out)
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
    _emit_report(parsed)


def cmd_validate() -> None:
    print("Normalizing snapshots (validate only)...")
    parsed, _ = normalize_all()
    _emit_report(parsed)


def _emit_report(parsed: dict[str, dict[str, ParsedBook]]) -> None:
    reports = validate_all(parsed)
    text = render_report(reports)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / "bible_validation_p1.txt"
    rp.write_text(text, encoding="utf-8")
    print(f"  wrote {rp}")
    for tid, r in sorted(reports.items()):
        print(f"  {tid}: books={r.n_books}/66 verses={r.n_verses} "
              f"errors={r.error_total} flags={r.flag_total}")


# --- P2 confessions ----------------------------------------------------------
def normalize_confessions() -> dict:
    """Parse every committed confession snapshot into ParsedConfessions."""
    manifest_path = CONFESSIONS_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No sources/confessions/manifest.json — run "
              "`snapshot-confessions` first.", file=sys.stderr)
        sys.exit(2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    parsed: dict = {}
    for cid, src in CONFESSION_SOURCES.items():
        entry = manifest["sources"][cid]
        prov = Provenance(
            source=f"ccel:{cid}",
            version=entry["version"],
            license=entry["license"],
            retrieved=entry["retrieved"],
            url=entry["url"],
            checksum=entry["sha256"],
        )
        content = src.dest.read_text(encoding="utf-8")
        pc = parse_confession(src, prov, content)
        parsed[cid] = pc
        print(f"  {cid}: {len(pc.chunks)} chunks, {len(pc.flags)} flag(s)")
    return parsed


def cmd_snapshot_confessions() -> None:
    print("Snapshotting confession sources -> sources/confessions/ + manifest.json")
    manifest = snapshot_confessions(RETRIEVED)
    for cid, e in sorted(manifest["sources"].items()):
        print(f"  {cid}: {e['sha256']}  {e['file']}")


def cmd_build_confessions() -> None:
    print("Normalizing confession snapshots...")
    parsed = normalize_confessions()
    out = OUTPUT_DIR / "confessions.sqlite"
    print(f"Building {out} ...")
    write_confessions(parsed, out)
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
    _emit_confession_report(parsed)


def cmd_validate_confessions() -> None:
    print("Normalizing confession snapshots (validate only)...")
    parsed = normalize_confessions()
    _emit_confession_report(parsed)


def _emit_confession_report(parsed: dict) -> None:
    reports = validate_all_confessions(parsed)
    text = render_confession_report(reports)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / "confessions_validation_p2.txt"
    rp.write_text(text, encoding="utf-8")
    print(f"  wrote {rp}")
    for did, r in sorted(reports.items()):
        print(f"  {did}: count_ok={r.count_ok} errors={r.error_total} "
              f"flags={r.flag_total}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "all"
    if cmd == "snapshot":
        cmd_snapshot()
    elif cmd == "build":
        cmd_build()
    elif cmd == "validate":
        cmd_validate()
    elif cmd == "all":
        if not (SOURCES_DIR / "manifest.json").exists():
            cmd_snapshot()
        cmd_build()
    elif cmd == "snapshot-confessions":
        cmd_snapshot_confessions()
    elif cmd == "build-confessions":
        cmd_build_confessions()
    elif cmd == "validate-confessions":
        cmd_validate_confessions()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
