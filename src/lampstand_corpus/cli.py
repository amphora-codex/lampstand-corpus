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

Usage (P5 cross-references):
    python -m lampstand_corpus.cli snapshot-crossrefs  # OpenBible TSK (CC-BY) zip
    python -m lampstand_corpus.cli build-crossrefs     # build crossrefs.sqlite
    python -m lampstand_corpus.cli validate-crossrefs  # write report (no DB)

Usage (P6 embeddings + BM25):
    python -m lampstand_corpus.cli snapshot-model      # download BGE-small (pinned)
    python -m lampstand_corpus.cli build-embeddings    # build embeddings.sqlite + report
    python -m lampstand_corpus.cli validate-embeddings # rebuild report from chunks only

``build-embeddings`` chunks the *built* per-resource DBs, encodes them with
BGE-small on CPU (deterministic), writes embeddings.sqlite (gitignored), and emits
the P6 validation report (chunk counts, BM25 stats, retrieval smoke test, and the
bit-for-bit determinism outcome). Requires the ``[embeddings]`` extra
(sentence-transformers + torch) and a one-time ``snapshot-model``.

Reads committed snapshots from sources/, writes output/*.sqlite (gitignored) and
reports/*.txt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

from . import books
from .build import write_bibles
from .build_commentaries import write_commentaries
from .build_confessions import write_confessions
from .build_crossrefs import write_crossrefs
from .build_lexicons import write_lexicons
from .commentaries import (
    COMMENTARIES_DIR,
    COMMENTARY_SOURCES,
    parse_commentary,
)
from .confessions import CONFESSION_SOURCES, CONFESSIONS_DIR, parse_confession
from .crossrefs import (
    CROSSREFS_DIR,
    ParsedCrossRefs,
    parse_crossrefs,
)
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
    snapshot_crossrefs,
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
from .validate_crossrefs import (
    render_crossref_report,
    validate_crossrefs,
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


# --- P5 cross-references -----------------------------------------------------
def normalize_crossrefs() -> ParsedCrossRefs:
    """Parse the committed TSK cross-reference snapshot into normalized refs."""
    manifest_path = CROSSREFS_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No sources/crossrefs/manifest.json — run `snapshot-crossrefs` "
              "first.", file=sys.stderr)
        sys.exit(2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["sources"]["tsk"]
    prov = Provenance(
        source="openbible:tsk",
        version=f"OpenBible.info cross-references (retrieved {entry['retrieved']})",
        license=entry["license"],
        retrieved=entry["retrieved"],
        url=entry["url"],
        checksum=entry["sha256"],
    )
    txt_path = CROSSREFS_DIR / "cross_references.txt"
    content = txt_path.read_text(encoding="utf-8")
    parsed = parse_crossrefs(content, prov)
    print(f"  tsk: {len(parsed.refs)} cross-references "
          f"({len(parsed.unparsed)} unparsed line(s))")
    return parsed


def cmd_snapshot_crossrefs() -> None:
    print("Snapshotting TSK cross-references -> sources/crossrefs/ + manifest.json")
    manifest = snapshot_crossrefs(RETRIEVED)
    e = manifest["sources"]["tsk"]
    print(f"  license: {manifest['license']}")
    print(f"  attribution: {manifest['attribution']}")
    print(f"  zip {e['zip_sha256']}  {e['zip_file']}")
    print(f"  txt {e['sha256']}  {e['file']}")


def cmd_build_crossrefs() -> None:
    print("Normalizing cross-reference snapshot...")
    parsed = normalize_crossrefs()
    out = OUTPUT_DIR / "crossrefs.sqlite"
    print(f"Building {out} ...")
    write_crossrefs(parsed, out)
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")
    _emit_crossref_report(parsed)


def cmd_validate_crossrefs() -> None:
    print("Normalizing cross-reference snapshot (validate only)...")
    parsed = normalize_crossrefs()
    _emit_crossref_report(parsed)


def _emit_crossref_report(parsed: ParsedCrossRefs) -> None:
    rep = validate_crossrefs(parsed)
    text = render_crossref_report(rep)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / "crossrefs_validation_p5.txt"
    rp.write_text(text, encoding="utf-8")
    print(f"  wrote {rp}")
    print(f"  refs={rep.n_refs} sources={rep.n_distinct_sources} "
          f"ranges={rep.n_ranges} errors={rep.error_total} flags={rep.flag_total}")
    print(f"  non-resolving: source={rep.n_nonresolving_source} "
          f"target={rep.n_nonresolving_target}")


# --- P6 embeddings + BM25 ----------------------------------------------------
# Canonical retrieval smoke queries (spec: task validation). Each carries a
# predicate over the dense top-k that a human can also eyeball in the report.
_SMOKE_QUERIES: list[tuple[str, str, object]] = [
    (
        "justification by faith",
        "Romans 3-5 / Galatians (and Reformed confessions on justification)",
        lambda n: (
            (n.book in {"ROM", "GAL"})
            or (n.resource_type == "confession")
            or ("justif" in n.text.lower())
        ),
    ),
    (
        "the LORD is my shepherd",
        "Psalm 23",
        lambda n: (n.book == "PSA" and n.chapter == 23),
    ),
    (
        "in the beginning God created the heavens and the earth",
        "Genesis 1",
        lambda n: (n.book == "GEN" and n.chapter == 1),
    ),
]


def cmd_snapshot_model() -> None:
    print("Snapshotting embedding model -> models/ (gitignored)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub not installed — run `pip install -e \".[embeddings]\"`",
              file=sys.stderr)
        sys.exit(2)
    from .embeddings import MODEL_NAME, MODEL_REVISION
    from .encode import MODEL_CACHE, model_provenance

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))
    path = snapshot_download(MODEL_NAME, revision=MODEL_REVISION,
                             cache_dir=str(MODEL_CACHE))
    print(f"  model:    {MODEL_NAME}")
    print(f"  revision: {MODEL_REVISION}")
    print(f"  path:     {path}")
    prov = model_provenance()
    print(f"  combined sha256: {prov['combined_sha256']}")
    print(f"  files hashed:    {len(prov['files'])}")


def cmd_build_embeddings() -> None:
    import time

    import numpy as np

    from .embeddings import extract_all
    from .encode import MODEL_CACHE, encode_chunks, model_provenance

    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))

    print("Extracting chunks from built DBs...")
    ec = extract_all(OUTPUT_DIR)
    print(f"  {len(ec.chunks)} chunks "
          f"({', '.join(f'{k}={v}' for k, v in sorted(ec.by_type().items()))})")
    for rt, items in sorted(ec.skipped.items()):
        if items:
            print(f"  skipped {rt}: {len(items)} (flagged, not dropped)")

    prov = model_provenance()
    print(f"Encoding with {prov['name']} @ {prov['revision'][:12]} on CPU "
          f"(deterministic)...")
    t0 = time.perf_counter()
    vectors = encode_chunks(ec.chunks, device="cpu")
    wall = time.perf_counter() - t0
    print(f"  encoded {vectors.shape[0]} vectors dim={vectors.shape[1]} "
          f"in {wall:.1f}s")

    # Determinism: encode a deterministic sample twice on CPU; require bit-for-bit.
    det = _check_determinism(ec.chunks, vectors)

    out = OUTPUT_DIR / "embeddings.sqlite"
    print(f"Building {out} ...")
    from .build_embeddings import write_embeddings
    stats = write_embeddings(
        ec.chunks, vectors, out, model_provenance=prov, skipped=ec.skipped)
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")

    _emit_embedding_report(ec, prov, stats, det, wall, np)


def _check_determinism(chunks, vectors):
    """Re-encode a deterministic sample on CPU and compare bit-for-bit.

    Encoding the full corpus twice is slow; we re-encode a fixed deterministic
    sample (every Nth chunk, capped) and require an exact byte match. If the bytes
    ever differ we fall back to cosine-within-tolerance and FLAG it.
    """
    import numpy as np

    from .encode import encode_chunks
    from .validate_embeddings import DeterminismResult

    if not chunks:
        return DeterminismResult(method="bit-for-bit", identical=True,
                                 note="no chunks")
    # Fixed, deterministic sample (no RNG): stride through the corpus, cap at 400.
    n = len(chunks)
    stride = max(1, n // 400)
    sample_idx = list(range(0, n, stride))[:400]
    sample = [chunks[i] for i in sample_idx]
    first = vectors[sample_idx]
    second = encode_chunks(sample, device="cpu")
    if first.tobytes() == second.tobytes():
        return DeterminismResult(
            method="bit-for-bit", identical=True,
            note=f"verified on a {len(sample)}-chunk CPU re-encode sample")
    # Not bit-for-bit — record tolerance + flag.
    diff = float(np.max(np.abs(first - second)))
    cos = float(np.min(np.sum(first * second, axis=1)
                       / (np.linalg.norm(first, axis=1)
                          * np.linalg.norm(second, axis=1) + 1e-12)))
    return DeterminismResult(
        method="cosine-tolerance", identical=False,
        max_abs_diff=diff, min_cosine=cos,
        note=f"{len(sample)}-chunk CPU re-encode differed in bytes; "
             f"FLAGGED for architect (CLAUDE.md rule 6)")


def cmd_validate_embeddings() -> None:
    """Rebuild the chunk-level report sections without re-encoding.

    Reads the existing embeddings.sqlite for BM25 stats + provenance and re-runs
    the retrieval smoke test against the stored vectors. Does not re-encode the
    corpus or re-check determinism (that's part of build).
    """
    import numpy as np

    from .embeddings import extract_all
    from .encode import MODEL_CACHE

    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))
    out = OUTPUT_DIR / "embeddings.sqlite"
    if not out.exists():
        print("No output/embeddings.sqlite — run `build-embeddings` first.",
              file=sys.stderr)
        sys.exit(2)
    ec = extract_all(OUTPUT_DIR)
    conn = __import__("sqlite3").connect(out)
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    bstats = dict(conn.execute("SELECT key, value FROM bm25_stats").fetchall())
    n_postings = conn.execute("SELECT count(*) FROM bm25_posting").fetchone()[0]
    conn.close()
    prov = {
        "name": meta.get("model_name", ""),
        "revision": meta.get("model_revision", ""),
        "combined_sha256": meta.get("model_combined_sha256", ""),
    }
    stats = {
        "vocab_size": int(bstats.get("vocab_size", 0)),
        "n_postings": n_postings,
        "avgdl": bstats.get("avgdl", 0.0),
    }
    _emit_embedding_report(ec, prov, stats, None, 0.0, np)


def _run_smoke(out_path, np):
    """Encode the smoke queries + return SmokeQuery results against the artifact."""
    from .encode import load_model
    from .retrieve import encode_query, load_matrix, topk
    from .validate_embeddings import SmokeHit, SmokeQuery

    matrix, meta = load_matrix(out_path)
    model = load_model(device="cpu")
    results: list[SmokeQuery] = []
    for query, expect, predicate in _SMOKE_QUERIES:
        qvec = encode_query(model, query)
        neighbors = topk(matrix, meta, qvec, k=5)
        sq = SmokeQuery(query=query, expect=expect)
        for i, nb in enumerate(neighbors, start=1):
            preview = nb.text[:70].replace("\n", " ")
            sq.hits.append(SmokeHit(
                rank=i, score=nb.score, resource_type=nb.resource_type,
                source=nb.source, anchor=nb.anchor, preview=preview))
        sq.passed = any(predicate(nb) for nb in neighbors)
        results.append(sq)
    return results


def _emit_embedding_report(ec, prov, stats, det, wall, np):
    from .validate_embeddings import render_embedding_report, summarize_extract

    rep = summarize_extract(ec)
    rep.model_name = prov["name"]
    rep.model_revision = prov["revision"]
    rep.model_combined_sha256 = prov["combined_sha256"]
    rep.vocab_size = stats["vocab_size"]
    rep.n_postings = stats["n_postings"]
    rep.avgdl = float(stats["avgdl"])
    rep.determinism = det
    rep.wall_seconds = wall

    out = OUTPUT_DIR / "embeddings.sqlite"
    print("Running retrieval smoke test...")
    rep.smoke = _run_smoke(out, np)
    for sq in rep.smoke:
        print(f"  [{'PASS' if sq.passed else 'REVIEW'}] {sq.query!r}")

    text = render_embedding_report(rep)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / "embeddings_validation_p6.txt"
    rp.write_text(text, encoding="utf-8")
    print(f"  wrote {rp}")
    print(f"  chunks={rep.n_chunks} dim={rep.embedding_dim} "
          f"vocab={rep.vocab_size} postings={rep.n_postings} "
          f"errors={rep.error_total} flags={rep.flag_total}")


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
    elif cmd == "snapshot-crossrefs":
        cmd_snapshot_crossrefs()
    elif cmd == "build-crossrefs":
        cmd_build_crossrefs()
    elif cmd == "validate-crossrefs":
        cmd_validate_crossrefs()
    elif cmd == "snapshot-model":
        cmd_snapshot_model()
    elif cmd == "build-embeddings":
        cmd_build_embeddings()
    elif cmd == "validate-embeddings":
        cmd_validate_embeddings()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
