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

Usage (P3 commentaries):
    python -m lampstand_corpus.cli snapshot-commentaries # download CCEL ThML volumes
    python -m lampstand_corpus.cli snapshot-spurgeon     # download Treasury OCR (IA)
    python -m lampstand_corpus.cli build-commentaries    # build commentaries.sqlite
    python -m lampstand_corpus.cli validate-commentaries # write report (no DB)

``build-commentaries`` / ``validate-commentaries`` automatically include Spurgeon
when its snapshots are present (run ``snapshot-spurgeon`` first); otherwise the
three CCEL commentators build without it.

Usage (P4 lexicons):
    python -m lampstand_corpus.cli snapshot-lexicons   # Strong's G/H + BDB sources
    python -m lampstand_corpus.cli snapshot-tagged      # OSHB Hebrew + SBLGNT (flag)
    python -m lampstand_corpus.cli snapshot-stepbible   # STEPBible TAGNT + TBESG (Greek)
    python -m lampstand_corpus.cli build-lexicons       # build lexicons.sqlite
    python -m lampstand_corpus.cli validate-lexicons    # write report (no DB)

``build-lexicons`` / ``validate-lexicons`` automatically include the OSHB
Strong's-tagged Hebrew text and the STEPBible Greek TAGNT/TBESG when their
snapshots are present (run ``snapshot-tagged`` + ``snapshot-stepbible`` first);
otherwise the dictionaries build without the missing tables.

Reads committed snapshots from sources/, writes output/*.sqlite (gitignored) and
reports/*.txt.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

from . import books
from .build import write_bibles
from .build_commentaries import write_commentaries
from .build_confessions import write_confessions
from .build_lexicons import write_lexicons
from .commentaries import (
    COMMENTARIES_DIR,
    COMMENTARY_SOURCES,
    parse_commentary,
)
from .confessions import CONFESSION_SOURCES, CONFESSIONS_DIR, parse_confession
from .lexicons import (
    FLAGGED_TEXT_SOURCES,
    LEXICON_SOURCES,
    LEXICONS_DIR,
    STEPBIBLE_DIR,
    STEPBIBLE_LICENSE,
    STEPBIBLE_SOURCES,
    STEPBIBLE_VERSION,
    TAGGED_TEXT_SOURCES,
    THAYERS_FLAG,
    ParsedLexicon,
    ParsedTaggedText,
    parse_bdb,
    parse_oshb,
    parse_strongs,
    parse_tagnt,
    parse_tbesg,
)
from .schema import Provenance
from .sources import (
    BIBLE_SOURCES,
    SOURCES_DIR,
    snapshot_bibles,
    snapshot_commentaries,
    snapshot_confessions,
    snapshot_lexicons,
    snapshot_spurgeon,
    snapshot_stepbible,
    snapshot_tagged_text,
)
from .spurgeon import (
    SPURGEON_DIR,
    SPURGEON_SOURCE,
    ParsedSpurgeon,
    parse_spurgeon,
)
from .usfm import ParsedBook, parse_usfm
from .validate import render_report, validate_all
from .validate_commentaries import (
    render_commentary_report,
    validate_all_commentaries,
)
from .validate_confessions import (
    crosscheck_against_bibles,
    crosscheck_wcf_prose,
    render_confession_report,
    validate_all_confessions,
)
from .validate_lexicons import (
    render_lexicon_report,
    validate_lexicon,
    validate_orphans,
    validate_tagged,
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
    crosscheck = crosscheck_against_bibles(parsed, OUTPUT_DIR / "bibles.sqlite")
    burges = (CONFESSIONS_DIR / "wcf" / "wcf-burges-1646-wikisource.parse.json")
    wcf_prose = (
        crosscheck_wcf_prose(parsed["wcf"], burges) if "wcf" in parsed else None
    )
    text = render_confession_report(
        reports, bible_crosscheck=crosscheck, wcf_prose_crosscheck=wcf_prose
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / "confessions_validation_p2.txt"
    rp.write_text(text, encoding="utf-8")
    print(f"  wrote {rp}")
    for did, r in sorted(reports.items()):
        print(f"  {did}: count_ok={r.count_ok} errors={r.error_total} "
              f"flags={r.flag_total}")


# --- P3 commentaries ---------------------------------------------------------
def normalize_commentaries() -> dict:
    """Parse every committed commentary snapshot into ParsedCommentaries."""
    manifest_path = COMMENTARIES_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No sources/commentaries/manifest.json — run "
              "`snapshot-commentaries` first.", file=sys.stderr)
        sys.exit(2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    parsed: dict = {}
    for cid, src in COMMENTARY_SOURCES.items():
        entry = manifest["sources"][cid]
        prov_by_volume: dict[str, Provenance] = {}
        content_by_volume: dict[str, str] = {}
        for volume in src.volumes:
            ventry = entry["volumes"][volume]
            prov_by_volume[volume] = Provenance(
                source=f"ccel:{cid}:{volume}",
                version=src.version,
                license=src.license,
                retrieved=entry["retrieved"],
                url=ventry["url"],
                checksum=ventry["sha256"],
            )
            content_by_volume[volume] = src.dest(volume).read_text(encoding="utf-8")
        pc = parse_commentary(src, prov_by_volume, content_by_volume)
        parsed[cid] = pc
        print(f"  {cid}: {len(pc.chunks)} chunks across {len(pc.coverage)} books, "
              f"{len(pc.flags)} flag(s)")

    sp = normalize_spurgeon()
    if sp is not None:
        parsed[sp.id] = sp
        print(f"  {sp.id}: {len(sp.chunks)} chunks across "
              f"{len(sp.psalms_seen)} psalms, {len(sp.flags)} flag(s)")
    return parsed


def normalize_spurgeon() -> ParsedSpurgeon | None:
    """Parse the committed Treasury-of-David OCR snapshots, if present.

    Returns None (with a note) when the Spurgeon snapshots haven't been fetched —
    the CCEL commentators still build without it.
    """
    manifest_path = SPURGEON_DIR / "manifest.json"
    if not manifest_path.exists():
        print("  spurgeon: no snapshot manifest — run `snapshot-spurgeon` to "
              "include the Treasury of David.")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["sources"][SPURGEON_SOURCE.id]
    prov_by_volume: dict[str, Provenance] = {}
    content_by_volume: dict[str, str] = {}
    for stem in SPURGEON_SOURCE.volumes:
        ventry = entry["volumes"][stem]
        prov_by_volume[stem] = Provenance(
            source=f"ia:spurgeon:{stem}",
            version=SPURGEON_SOURCE.version,
            license=SPURGEON_SOURCE.license,
            retrieved=entry["retrieved"],
            url=ventry["url"],
            checksum=ventry["sha256"],
        )
        content_by_volume[stem] = SPURGEON_SOURCE.dest(stem).read_text(
            encoding="utf-8"
        )
    return parse_spurgeon(SPURGEON_SOURCE, prov_by_volume, content_by_volume)


def cmd_snapshot_commentaries() -> None:
    print("Snapshotting commentary sources -> sources/commentaries/ + manifest.json")
    manifest = snapshot_commentaries(RETRIEVED)
    for cid, e in sorted(manifest["sources"].items()):
        nvol = len(e["volumes"])
        print(f"  {cid}: {nvol} volume(s)")
        for _v, ve in sorted(e["volumes"].items()):
            print(f"     {ve['sha256']}  {ve['file']}")


def cmd_snapshot_spurgeon() -> None:
    print("Snapshotting Treasury-of-David OCR -> sources/commentaries/spurgeon/")
    manifest = snapshot_spurgeon(RETRIEVED)
    entry = manifest["sources"]["spurgeon"]
    for stem, ve in sorted(entry["volumes"].items()):
        print(f"  {stem} ({ve['identifier']}) Ps {ve['psalm_first']}-"
              f"{ve['psalm_last']}: {ve['sha256']}")
    mv = entry["missing_volume"]
    if mv["psalm_first"] is None:
        print("  Psalms 104-118 gap-filled from treasuryofdavidc0005spur "
              "(vol. 5, 1882; flagged for the architect spot-check)")
    else:
        print(f"  MISSING from the *spurgoog set: Psalms {mv['psalm_first']}-"
              f"{mv['psalm_last']} (flagged for the architect)")


def cmd_build_commentaries() -> None:
    print("Normalizing commentary snapshots...")
    parsed = normalize_commentaries()
    out = OUTPUT_DIR / "commentaries.sqlite"
    print(f"Building {out} ...")
    write_commentaries(parsed, out)
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
    _emit_commentary_report(parsed)


def cmd_validate_commentaries() -> None:
    print("Normalizing commentary snapshots (validate only)...")
    parsed = normalize_commentaries()
    _emit_commentary_report(parsed)


def _emit_commentary_report(parsed: dict) -> None:
    reports = validate_all_commentaries(parsed)
    spurgeon = parsed.get(SPURGEON_SOURCE.id)
    text = render_commentary_report(reports, spurgeon=spurgeon)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / "commentaries_validation_p3.txt"
    rp.write_text(text, encoding="utf-8")
    print(f"  wrote {rp}")
    for cid, r in sorted(reports.items()):
        print(f"  {cid}: chunks={r.n_chunks} books={r.n_books}/{r.expected_books} "
              f"errors={r.error_total} flags={r.flag_total}")


# --- P4 lexicons -------------------------------------------------------------
def normalize_lexicons() -> dict[str, ParsedLexicon]:
    """Parse every committed lexicon dictionary snapshot into ParsedLexicons."""
    manifest_path = LEXICONS_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No sources/lexicons/manifest.json — run `snapshot-lexicons` "
              "first.", file=sys.stderr)
        sys.exit(2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    parsed: dict[str, ParsedLexicon] = {}
    for lid, src in LEXICON_SOURCES.items():
        entry = manifest["sources"][lid]
        prov = Provenance(
            source=f"openscriptures:{lid}",
            version=entry["version"],
            license=entry["license"],
            retrieved=entry["retrieved"],
            url=entry["url"],
            checksum=entry["sha256"],
        )
        content = src.dest.read_text(encoding="utf-8")
        if src.lexicon == "strongs":
            pl = parse_strongs(src, prov, content)
        else:  # bdb — needs the LexicalIndex aux file for Strong's linkage
            aux_path = LEXICONS_DIR / src.id / "LexicalIndex.xml"
            index_xml = aux_path.read_text(encoding="utf-8")
            pl = parse_bdb(src, prov, content, index_xml)
        parsed[lid] = pl
        print(f"  {lid}: {len(pl.entries)} entries, {len(pl.flags)} flag(s)")

    # TBESG (STEPBible Greek lexicon) — added when its snapshot is present.
    tbesg = normalize_tbesg()
    if tbesg is not None:
        parsed["tbesg"] = tbesg
        print(f"  tbesg: {len(tbesg.entries)} entries, {len(tbesg.flags)} flag(s)")
    return parsed


def _stepbible_manifest() -> dict | None:
    manifest_path = STEPBIBLE_DIR / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _stepbible_provenance(source_id: str, src) -> Provenance:
    """Build a single combined provenance for a STEPBible source (TAGNT/TBESG)."""
    manifest = _stepbible_manifest()
    assert manifest is not None
    files = manifest["sources"][source_id]["files"]
    file_sums = sorted(f["sha256"] for f in files.values())
    combined = hashlib.sha256("".join(file_sums).encode()).hexdigest()
    # url points at the repo subdir (per-file urls are in the manifest).
    first_url = next(iter(files.values()))["url"]
    return Provenance(
        source=f"stepbible:{source_id}",
        version=STEPBIBLE_VERSION,
        license=STEPBIBLE_LICENSE,
        retrieved=manifest["retrieved"],
        url=first_url,
        checksum=combined,
    )


def normalize_tbesg() -> ParsedLexicon | None:
    """Parse the committed TBESG Greek-lexicon snapshot, if present."""
    manifest = _stepbible_manifest()
    if manifest is None or "tbesg" not in manifest["sources"]:
        print("  tbesg: no STEPBible snapshot — run `snapshot-stepbible` to "
              "include the TBESG Greek lexicon.")
        return None
    src = STEPBIBLE_SOURCES["tbesg"]
    prov = _stepbible_provenance("tbesg", src)
    fname = src.files[0]
    content = src.dest(fname).read_text(encoding="utf-8")
    return parse_tbesg(prov, content)


def normalize_tagged() -> dict[str, ParsedTaggedText]:
    """Parse the committed OSHB Strong's-tagged Hebrew snapshots, if present.

    Returns an empty dict (with a note) when the tagged snapshots haven't been
    fetched — the dictionaries still build without the tagged-word table. SBLGNT
    is snapshot-only + flagged and is intentionally NOT normalized.
    """
    manifest_path = LEXICONS_DIR / "tagged_manifest.json"
    if not manifest_path.exists():
        print("  tagged-text: no snapshot manifest — run `snapshot-tagged` to "
              "include the OSHB Strong's-tagged Hebrew text.")
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    parsed: dict[str, ParsedTaggedText] = {}
    for sid, src in TAGGED_TEXT_SOURCES.items():
        entry = manifest["ingested"][sid]
        # One provenance per source; checksum is the sorted concatenation of the
        # per-file checksums so it's stable + reproducible.
        file_sums = sorted(f["sha256"] for f in entry["files"].values())
        combined = hashlib.sha256("".join(file_sums).encode()).hexdigest()
        prov = Provenance(
            source=f"openscriptures:{sid}",
            version=entry["version"],
            license=entry["license"],
            retrieved=entry["retrieved"],
            url=src.base_url,
            checksum=combined,
        )
        content_by_book = {
            stem: src.dest(stem).read_text(encoding="utf-8")
            for stem in src.book_stems
        }
        pt = parse_oshb(src, prov, content_by_book)
        parsed[sid] = pt
        print(f"  {sid}: {len(pt.words)} tagged words across "
              f"{len(pt.books_seen)} books, {len(pt.flags)} flag(s)")

    # TAGNT (STEPBible Greek NT tagging) — added when its snapshot is present.
    tagnt = normalize_tagnt()
    if tagnt is not None:
        parsed["tagnt"] = tagnt
        print(f"  tagnt: {len(tagnt.words)} tagged words across "
              f"{len(tagnt.books_seen)} books, {len(tagnt.flags)} flag(s)")
    return parsed


def normalize_tagnt() -> ParsedTaggedText | None:
    """Parse the committed TAGNT Strong's-tagged Greek-NT snapshot, if present."""
    manifest = _stepbible_manifest()
    if manifest is None or "tagnt" not in manifest["sources"]:
        print("  tagnt: no STEPBible snapshot — run `snapshot-stepbible` to "
              "include the TAGNT Greek-NT tagging.")
        return None
    src = STEPBIBLE_SOURCES["tagnt"]
    prov = _stepbible_provenance("tagnt", src)
    content_by_file = {
        fname: src.dest(fname).read_text(encoding="utf-8") for fname in src.files
    }
    return parse_tagnt(prov, content_by_file)


def cmd_snapshot_lexicons() -> None:
    print("Snapshotting lexicon sources -> sources/lexicons/ + manifest.json")
    manifest = snapshot_lexicons(RETRIEVED)
    for lid, e in sorted(manifest["sources"].items()):
        print(f"  {lid}: {e['sha256']}  {e['file']}")
        if "aux_file" in e:
            print(f"     aux {e['aux_sha256']}  {e['aux_file']}")


def cmd_snapshot_tagged() -> None:
    print("Snapshotting tagged-text sources -> sources/lexicons/ "
          "+ tagged_manifest.json")
    manifest = snapshot_tagged_text(RETRIEVED)
    for sid, e in sorted(manifest["ingested"].items()):
        print(f"  INGESTED {sid}: {len(e['files'])} file(s) [{e['license']}]")
    for sid, e in sorted(manifest["flagged"].items()):
        print(f"  FLAGGED  {sid}: {len(e['files'])} file(s) "
              f"snapshotted for provenance only")
        print(f"     {e['flag']}")


def cmd_snapshot_stepbible() -> None:
    print("Snapshotting STEPBible TAGNT + TBESG -> sources/lexicons/stepbible/")
    manifest = snapshot_stepbible(RETRIEVED)
    print(f"  license: {manifest['license']}  attribution: "
          f"{manifest['attribution']}")
    for sid, e in sorted(manifest["sources"].items()):
        print(f"  {sid}: {len(e['files'])} file(s) [{e['name']}]")
        for _f, fe in sorted(e["files"].items()):
            print(f"     {fe['sha256']}  {fe['file']}")


def cmd_build_lexicons() -> None:
    print("Normalizing lexicon snapshots...")
    lexicons = normalize_lexicons()
    tagged = normalize_tagged()
    out = OUTPUT_DIR / "lexicons.sqlite"
    print(f"Building {out} ...")
    write_lexicons(lexicons, out, tagged=tagged)
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
    _emit_lexicon_report(lexicons, tagged)


def cmd_validate_lexicons() -> None:
    print("Normalizing lexicon snapshots (validate only)...")
    lexicons = normalize_lexicons()
    tagged = normalize_tagged()
    _emit_lexicon_report(lexicons, tagged)


def _emit_lexicon_report(
    lexicons: dict[str, ParsedLexicon], tagged: dict[str, ParsedTaggedText]
) -> None:
    lex_reports = {lid: validate_lexicon(pl) for lid, pl in lexicons.items()}
    tagged_reports = {sid: validate_tagged(pt) for sid, pt in tagged.items()}
    orphans = validate_orphans(lexicons, tagged)
    text = render_lexicon_report(
        lex_reports,
        tagged_reports=tagged_reports,
        orphans=orphans,
        thayers_flag=THAYERS_FLAG,
        sblgnt_flag=FLAGGED_TEXT_SOURCES["sblgnt"].flag,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / "lexicons_validation_p4.txt"
    rp.write_text(text, encoding="utf-8")
    print(f"  wrote {rp}")
    for lid, r in sorted(lex_reports.items()):
        print(f"  {lid}: entries={r.n_entries} linked={r.n_linked} "
              f"errors={r.error_total} flags={r.flag_total}")
    for sid, r in sorted(tagged_reports.items()):
        print(f"  {sid}: words={r.n_words} books={r.n_books} "
              f"errors={r.error_total} flags={r.flag_total}")
    print(f"  orphan Strong's: hebrew-tagged={len(orphans.from_tagged)} "
          f"greek-tagged={len(orphans.from_greek)} bdb={len(orphans.from_bdb)}")


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
    elif cmd == "snapshot-commentaries":
        cmd_snapshot_commentaries()
    elif cmd == "snapshot-spurgeon":
        cmd_snapshot_spurgeon()
    elif cmd == "build-commentaries":
        cmd_build_commentaries()
    elif cmd == "validate-commentaries":
        cmd_validate_commentaries()
    elif cmd == "snapshot-lexicons":
        cmd_snapshot_lexicons()
    elif cmd == "snapshot-tagged":
        cmd_snapshot_tagged()
    elif cmd == "snapshot-stepbible":
        cmd_snapshot_stepbible()
    elif cmd == "build-lexicons":
        cmd_build_lexicons()
    elif cmd == "validate-lexicons":
        cmd_validate_lexicons()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
