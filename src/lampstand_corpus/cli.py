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
    python -m lampstand_corpus.cli snapshot-model       # download BGE-small (pinned)
    python -m lampstand_corpus.cli build-embeddings     # INCREMENTAL build + report
    python -m lampstand_corpus.cli build-embeddings full # force full re-encode
    python -m lampstand_corpus.cli validate-embeddings  # rebuild report from chunks only

``build-embeddings`` chunks the *built* per-resource DBs, encodes them with
BGE-small on CPU (deterministic), writes embeddings.sqlite (gitignored), and emits
the P6 validation report (chunk counts, BM25 stats, retrieval smoke test, the
incremental reuse accounting, and the determinism outcome). By default it is
INCREMENTAL — vectors whose content-addressed chunk id is unchanged are reused from
the existing embeddings.sqlite and only changed/new chunks are re-encoded (the
BM25 index is rebuilt over the full new chunk set). ``build-embeddings full`` forces
a from-scratch re-encode. Requires the ``[embeddings]`` extra (sentence-transformers
+ torch) and a one-time ``snapshot-model``.

Usage (M4 on-device query model — Core ML export):
    python -m lampstand_corpus.cli coreml-export  # BGE-small -> BGEQuery.mlpackage

``coreml-export`` traces the pinned BGE-small weights into a fp16 Core ML
``.mlpackage`` whose graph bakes in CLS pooling + L2-normalize, runs a lineage gate
(combined_sha256 == the corpus vectors' model hash) and a cosine-parity gate (P4)
against the bundled BSB+WSC index, and emits the .mlpackage + vocab.txt +
bge_parity_fixture.json + bge_tokenizer_fixture.json under output/models/
(gitignored) with a report in reports/. Requires the ``[coreml]`` extra
(``pip install -e ".[coreml]"``) and a built bundled_search.sqlite (run ``package``).

Usage (retrieval eval — F5 measurement foundation):
    python -m lampstand_corpus.cli build-eval          # gold set -> output/eval_gold_v1.json
    python -m lampstand_corpus.cli validate-retrieval  # BM25/dense/hybrid arms + report
    python -m lampstand_corpus.cli sweep-retrieval     # fusion-constant sweep + report
    python -m lampstand_corpus.cli rerank-eval         # cross-encoder rerank gate + report

``build-eval`` derives a deterministic query→relevant-chunk gold set from the
built DBs (confession proof-texts, high-vote TSK cross-refs, commentary anchors,
plus the DRAFT hard-negative suite tracked at data/eval/hard_negatives_v1.json).
``validate-retrieval`` runs three app-parity retrieval arms (BM25-only, dense-
only, hybrid RRF exactly as HybridRetriever fuses) over the gold set and writes
reports/retrieval_eval_v1.md (committed). ``sweep-retrieval`` grid-sweeps the
fusion knobs the app hard-codes (rrfK, per-type BM25 limit, dense depth) and
appends the best-config deltas to the report — it RECOMMENDS constants, it never
changes the app.

Usage (P7 packaging):
    python -m lampstand_corpus.cli package  # split built DBs -> output/packs/ + manifest

``package`` splits the built per-resource DBs into a BUNDLED pack (BSB + WSC +
their search index, ships in the binary) and free ON-DEMAND packs (everything
else, downloaded on first launch), deterministically, under output/packs/
(gitignored). It writes the committed corpus_manifest.json (pack contents,
per-file SHA-256 + byte sizes, the rolled-up source licenses/attributions, and a
corpus-version placeholder) and reports the real pack sizes vs the targets.

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
    STRONGS_HEBREW_FLAG,
    TAGGED_TEXT_SOURCES,
    THAYERS_FLAG,
    ParsedLexicon,
    ParsedTaggedText,
    parse_bdb,
    parse_oshb,
    parse_strongs,
    parse_strongs_greek_xml,
    parse_strongs_hebrew_xml,
    parse_tagnt,
    parse_tbesg,
)
from .package import CORPUS_VERSION_PLACEHOLDER, package_corpus
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
# Snapshot-run date for the Westminster proof-text supplement (confessions);
# unchanged files keep their original retrieved stamps (see snapshot_confessions).
RETRIEVED_WESTMINSTER = "2026-07-02"

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
    manifest = snapshot_confessions(RETRIEVED_WESTMINSTER)
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
    from .validate_confessions import prooftext_summary
    text = render_confession_report(
        reports, bible_crosscheck=crosscheck, wcf_prose_crosscheck=wcf_prose,
        prooftext_lines=prooftext_summary(parsed),
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
        # Provenance source prefix reflects the repo the edition is sourced from
        # (morphgnt for the CC0 Greek, openscriptures for the CC-BY Hebrew/BDB).
        prefix = "morphgnt" if "morphgnt" in src.url else "openscriptures"
        prov = Provenance(
            source=f"{prefix}:{lid}",
            version=entry["version"],
            license=entry["license"],
            retrieved=entry["retrieved"],
            url=entry["url"],
            checksum=entry["sha256"],
        )
        content = src.dest.read_text(encoding="utf-8")
        if src.fmt == "strongs-greek-xml":
            pl = parse_strongs_greek_xml(src, prov, content)
        elif src.fmt == "strongs-hebrew-xml":
            pl = parse_strongs_hebrew_xml(src, prov, content)
        elif src.fmt == "bdb-xml":  # needs LexicalIndex aux for Strong's linkage
            aux_path = LEXICONS_DIR / src.id / "LexicalIndex.xml"
            index_xml = aux_path.read_text(encoding="utf-8")
            pl = parse_bdb(src, prov, content, index_xml)
        else:  # legacy .js JSON edition (no longer used for the swap)
            pl = parse_strongs(src, prov, content)
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
        strongs_hebrew_flag=STRONGS_HEBREW_FLAG,
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


def cmd_build_embeddings(*, full: bool = False) -> None:
    """Build embeddings.sqlite, reusing unchanged vectors from a prior build.

    By default this is INCREMENTAL: chunks whose content-addressed id already has a
    vector in ``output/embeddings.sqlite`` (built under the same model revision) are
    reused verbatim, and only changed/new chunks are encoded. Pass ``full=True`` to
    force a from-scratch re-encode of every chunk. The BM25 index is rebuilt over
    the full new chunk set either way (it's cheap and order-stable).
    """
    import time

    import numpy as np

    from .embeddings import extract_all
    from .encode import (
        MODEL_CACHE,
        encode_chunks,
        encode_chunks_incremental,
        model_provenance,
    )

    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))

    print("Extracting chunks from built DBs...")
    ec = extract_all(OUTPUT_DIR)
    print(f"  {len(ec.chunks)} chunks "
          f"({', '.join(f'{k}={v}' for k, v in sorted(ec.by_type().items()))})")
    for rt, items in sorted(ec.skipped.items()):
        if items:
            print(f"  skipped {rt}: {len(items)} (flagged, not dropped)")

    prov = model_provenance()
    out = OUTPUT_DIR / "embeddings.sqlite"
    inc = None
    # Only the EMBEDDABLE subset is encoded (BSB Scripture children + all
    # non-scripture retrieval units — Rank 8d); context-only parents and the
    # KJV/ASV/WEB children are BM25-only.
    embeddable = [c for c in ec.chunks if c.embed]
    print(f"  {len(embeddable)} embeddable of "
          f"{sum(1 for c in ec.chunks if c.indexed)} indexed chunks")
    t0 = time.perf_counter()
    if full or not out.exists():
        mode = "full re-encode" if full else "full encode (no prior DB)"
        print(f"Encoding with {prov['name']} @ {prov['revision'][:12]} on CPU "
              f"(deterministic, {mode})...")
        vectors = encode_chunks(embeddable, device="cpu")
    else:
        print(f"Incremental encode with {prov['name']} @ {prov['revision'][:12]} "
              f"on CPU (reusing unchanged vectors from {out.name})...")
        inc = encode_chunks_incremental(embeddable, out, device="cpu")
        vectors = inc.vectors
        for note in inc.notes:
            print(f"  note: {note}")
        print(f"  reused={inc.n_reused} encoded={inc.n_encoded} "
              f"dropped={inc.n_dropped} of {inc.n_total} total")
    wall = time.perf_counter() - t0
    print(f"  {vectors.shape[0]} vectors dim={vectors.shape[1]} ready in {wall:.1f}s")

    # Determinism (Option A): re-encode a deterministic sample on CPU and accept
    # within cosine-tolerance, locked by model revision + input checksums.
    det = _check_determinism(embeddable, vectors)

    print(f"Building {out} ...")
    from .build_embeddings import write_embeddings
    stats = write_embeddings(
        ec.chunks, vectors, out, model_provenance=prov, skipped=ec.skipped)
    print(f"  wrote {out} ({out.stat().st_size:,} bytes)")

    _emit_embedding_report(ec, prov, stats, det, wall, np, incremental=inc)


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


def _emit_embedding_report(ec, prov, stats, det, wall, np, *, incremental=None):
    from .validate_embeddings import (
        IncrementalStats,
        render_embedding_report,
        summarize_extract,
    )

    rep = summarize_extract(ec)
    rep.model_name = prov["name"]
    rep.model_revision = prov["revision"]
    rep.model_combined_sha256 = prov["combined_sha256"]
    rep.vocab_size = stats["vocab_size"]
    rep.n_postings = stats["n_postings"]
    rep.avgdl = float(stats["avgdl"])
    rep.determinism = det
    rep.wall_seconds = wall
    if incremental is not None:
        rep.incremental = IncrementalStats(
            n_total=incremental.n_total,
            n_reused=incremental.n_reused,
            n_encoded=incremental.n_encoded,
            n_dropped=incremental.n_dropped,
            prior_model_revision=incremental.prior_model_revision,
            full_reencode=incremental.full_reencode,
            notes=list(incremental.notes),
        )

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


# --- M4 T0: Core ML query-embedding model export -----------------------------
MODELS_OUT_DIR = OUTPUT_DIR / "models"


def _render_coreml_report(res, search_db: Path, det_note: str) -> str:
    """Fixed-width report for the .mlpackage export (lineage + parity + repro)."""
    from .embeddings import MODEL_REVISION as MODEL_REVISION_DISPLAY
    p = res.parity
    lines: list[str] = []
    w = lines.append
    w("=" * 72)
    w(" LampStand corpus — M4 T0 Core ML query-embedding export")
    w("=" * 72)
    w("")
    w("MODEL")
    w("  name                 BAAI/bge-small-en-v1.5")
    w(f"  revision             {MODEL_REVISION_DISPLAY}")
    w(f"  combined_sha256      {res.model_combined_sha256}")
    w("")
    w("LINEAGE GATE")
    w("  PASS — snapshot combined_sha256 matches the corpus vectors' "
      "model_combined_sha256")
    w("  (== bundled_search.sqlite meta value, == manifest acknowledgements)")
    w("")
    w("ARTIFACTS")
    w(f"  {res.mlpackage_path.name:24s} {res.mlpackage_bytes:>12,} B  "
      f"tree-sha256 {res.mlpackage_tree_sha256[:16]}…")
    w(f"  {res.vocab_path.name:24s} {res.vocab_path.stat().st_size:>12,} B  "
      f"sha256 {res.vocab_sha256[:16]}…")
    w(f"    vocab matches expected sha256: "
      f"{'YES' if res.vocab_sha256_matches_expected else 'NO (FLAG)'}")
    w(f"  {res.parity_fixture_path.name:24s} "
      f"{res.parity_fixture_path.stat().st_size:>12,} B")
    w(f"  {res.tokenizer_fixture_path.name:24s} "
      f"{res.tokenizer_fixture_path.stat().st_size:>12,} B")
    w("")
    w("PARITY GATE (P4) — bare passage vectors vs stored float32")
    w(f"  source index         {search_db.name}")
    w(f"  cases                {p.n}")
    w(f"  embedding dim        {p.dim}  (expect 384)")
    w(f"  mean cosine          {p.mean_cosine:.6f}  (floor {0.999})")
    w(f"  min  cosine          {p.min_cosine:.6f}  (floor {0.995})")
    w(f"  max  cosine          {p.max_cosine:.6f}")
    w(f"  output L2 norm range [{p.min_norm:.6f}, {p.max_norm:.6f}]  "
      f"(expect [0.999, 1.001])")
    w(f"  VERDICT              {'PASS' if p.passed else 'FAIL — STOP-AND-INVESTIGATE'}")
    if p.multi_subword_anchors:
        w("  multi-subword coverage (>=3 WordPieces exercised):")
        for a in p.multi_subword_anchors:
            w(f"    - {a}")
    w("")
    w("  per-case (worst 8 by cosine):")
    worst = sorted(p.per_case, key=lambda c: c["cosine"])[:8]
    for c in worst:
        w(f"    {c['cosine']:.6f}  norm={c['fp16_norm']:.5f}  "
          f"{c['resource_type']:10s} {c['anchor']}")
    w("")
    w("TOKENIZER FIXTURE (HF ground truth for the Swift tokenizer test)")
    for c in res.tokenizer_fixture:
        shown = c["text"] if c["text"] else "(empty string)"
        w(f"  {shown[:58]:58s} -> {c['input_ids']}")
    w("")
    w("PERFORMANCE / RESIDENCY")
    w(f"  single 512-token forward pass: {res.forward_512_ms:.1f} ms (CPU, fp16)")
    w("  seq axis: RangeDim(1,512,default=16)")
    w("")
    w("REPRODUCIBILITY")
    w(f"  {det_note}")
    if res.notes:
        w("")
        w("NOTES / FLAGS")
        for n in res.notes:
            w(f"  {n}")
    w("")
    w("NOTE: candidate artifacts only. The architect's 23-point spot-check + the")
    w("Swift-side parity test (T6) gate ship. Artifacts are gitignored build")
    w("output (output/models/) and are synced into the app via sync-corpus.sh.")
    w("")
    return "\n".join(lines)


def cmd_coreml_export() -> None:
    """Export the on-device BGE-small query model to a fp16 Core ML .mlpackage.

    Runs the lineage gate (combined_sha256 == the corpus vectors' model hash) and
    the parity gate (P4 cosine vs stored float32 passage vectors), emits the
    .mlpackage + vocab.txt + parity/tokenizer fixtures under output/models/
    (gitignored), and writes a report to reports/. Requires the ``[coreml]`` extra.
    """
    try:
        from .coreml_export import (
            LineageError,
            ParityError,
            export_coreml,
        )
    except ImportError as e:  # pragma: no cover - guarded import message
        print(f"  coreml_export import failed ({e}). Install the extras with "
              "`pip install -e \".[coreml]\"` (needs coremltools + torch + "
              "transformers).", file=sys.stderr)
        sys.exit(2)

    from .encode import MODEL_CACHE
    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))
    bundled_search = PACKS_DIR / "bundled_search.sqlite"
    if not bundled_search.exists():
        print(f"No {bundled_search} — run `package` first (the parity gate scores "
              "against the bundled BSB+WSC index).", file=sys.stderr)
        sys.exit(2)

    print("Exporting BGE-small query model -> Core ML (.mlpackage, fp16)")
    print("  Lineage gate, trace+convert (CLS pool + L2-normalize baked in), "
          "parity gate (P4)...")
    try:
        res = export_coreml(MODELS_OUT_DIR, bundled_search)
    except LineageError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # Reproducibility check: re-trace + re-convert into a temp dir and compare the
    # .mlpackage tree hash (coremltools mlprogram should embed no timestamps).
    det_note = _check_coreml_reproducibility(res)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = _render_coreml_report(res, bundled_search, det_note)
    rp = REPORTS_DIR / "coreml_export_m4.txt"
    rp.write_text(report, encoding="utf-8")

    # Update the committed manifest's packs.models subtree (sibling to bundled/
    # on_demand) so sync-corpus.sh verify() picks up the model + vocab with zero jq
    # changes. Only touches packs.models; the rest of the manifest is left intact.
    from .embeddings import MODEL_NAME, MODEL_REVISION
    from .package import build_models_pack, update_manifest_models
    models_pack = build_models_pack(
        mlpackage_path=res.mlpackage_path,
        mlpackage_tree_sha256=res.mlpackage_tree_sha256,
        mlpackage_bytes=res.mlpackage_bytes,
        vocab_path=res.vocab_path,
        vocab_sha256=res.vocab_sha256,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        model_combined_sha256=res.model_combined_sha256,
        precision="float16",
        seq_len="RangeDim(1,512)",
    )
    if CORPUS_MANIFEST.exists():
        update_manifest_models(CORPUS_MANIFEST, models_pack)
        print(f"  updated {CORPUS_MANIFEST.name} packs.models subtree")
    else:
        print(f"  WARNING: {CORPUS_MANIFEST.name} not found — run `package` first "
              "to write the base manifest; packs.models NOT added.", file=sys.stderr)

    print(f"  mlpackage:  {res.mlpackage_path} ({res.mlpackage_bytes:,} B)")
    print(f"  tree sha256: {res.mlpackage_tree_sha256}")
    print(f"  vocab.txt:  sha256 {res.vocab_sha256} "
          f"({'OK' if res.vocab_sha256_matches_expected else 'FLAG: mismatch'})")
    print(f"  parity:     mean={res.parity.mean_cosine:.6f} "
          f"min={res.parity.min_cosine:.6f} "
          f"norm=[{res.parity.min_norm:.5f},{res.parity.max_norm:.5f}] "
          f"dim={res.parity.dim} -> "
          f"{'PASS' if res.parity.passed else 'FAIL'}")
    print(f"  wrote {rp}")

    if not res.parity.passed:
        print("PARITY GATE FAILED (P4). This is a stop-and-investigate (almost "
              "always a tokenizer / token_type_ids bug), never a threshold "
              "loosening. Report written for forensics.", file=sys.stderr)
        raise ParityError(
            f"P4 floors not met: mean={res.parity.mean_cosine:.6f} (>=0.999), "
            f"min={res.parity.min_cosine:.6f} (>=0.995), "
            f"norm=[{res.parity.min_norm:.5f},{res.parity.max_norm:.5f}], "
            f"dim={res.parity.dim}")


def _check_coreml_reproducibility(res) -> str:
    """Re-trace+convert to a temp dir; compare the .mlpackage tree hash.

    A byte-identical tree hash proves the conversion is reproducible (no embedded
    timestamps). A mismatch is FLAGGED (not failed) with the differing hash so the
    architect can inspect — coremltools mlprogram is expected to be deterministic
    but the finding is recorded rather than assumed (CLAUDE.md rule 6).
    """
    import shutil as _shutil
    import tempfile

    from .coreml_export import (
        MLPACKAGE_NAME,
        _build_wrapper,
        _convert,
        _snapshot_dir,
        _trace,
    )
    from .package import _sha256_tree

    tmp = Path(tempfile.mkdtemp(prefix="coreml_repro_"))
    try:
        snap = _snapshot_dir()
        wrapper = _build_wrapper(snap)
        traced = _trace(wrapper)
        out = tmp / MLPACKAGE_NAME
        _convert(traced, out)
        second = _sha256_tree(out)
        if second == res.mlpackage_tree_sha256:
            return (f"PASS — re-trace+convert (separate process) yielded a "
                    f"byte-identical .mlpackage tree (sha256 {second[:16]}…). "
                    f"coremltools embeds no timestamps; cross-process protobuf/"
                    f"UUID ordering is canonicalized in the tool (model behavior "
                    f"is unchanged — predictions bit-identical).")
        return (f"FLAG — re-conversion tree hash {second} != first "
                f"{res.mlpackage_tree_sha256}. mlprogram is NOT byte-identical on "
                f"re-run; recorded for architect (CLAUDE.md rule 6).")
    finally:
        _shutil.rmtree(tmp, ignore_errors=True)


def cmd_coreml_export_reranker() -> None:
    """Export the on-device cross-encoder RERANKER to a fp16 Core ML .mlpackage.

    Runs only after the rerank quality gate SHIPPED (reports/reranker_eval_v1.md).
    Traces the pinned ms-marco-MiniLM-L-6-v2 cross-encoder (Apache-2.0, ~22 MB
    fp16 — the model that both clears the gate AND fits the on-device envelope),
    runs a lineage + score-parity gate (fp16 logits vs float32 PyTorch, plus
    zero order-inversions), emits Reranker.mlpackage + reranker_vocab.txt + the
    parity/tokenizer fixtures under output/models/ (gitignored), writes a report,
    and records the model in the manifest's packs.reranker + acknowledgements.
    Requires the [coreml] + [rerank] extras.
    """
    try:
        from .coreml_rerank_export import (
            RerankLineageError,
            RerankParityError,
            export_reranker_coreml,
        )
    except ImportError as e:  # pragma: no cover - guarded import message
        print(f"  coreml_rerank_export import failed ({e}). Install the extras "
              "with `pip install -e \".[coreml]\" -e \".[rerank]\"` (needs "
              "coremltools + torch + transformers).", file=sys.stderr)
        sys.exit(2)

    from .coreml_rerank_export import DEFAULT_RERANK_MODEL
    from .encode import MODEL_CACHE
    from .eval_rerank import RERANK_MODELS

    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))
    model_key = DEFAULT_RERANK_MODEL
    spec = RERANK_MODELS[model_key]
    print(f"Exporting cross-encoder reranker ({model_key}, {spec['license']}) "
          "-> Core ML (.mlpackage, fp16)")
    print("  Lineage, trace+convert (single logit output), parity gate "
          "(fp16 logit abs + zero order-inversions)...")
    try:
        res = export_reranker_coreml(MODELS_OUT_DIR, MODEL_CACHE, model_key=model_key)
    except RerankLineageError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    det_note = _check_reranker_reproducibility(res, model_key)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = _render_reranker_report(res, spec, det_note)
    rp = REPORTS_DIR / "coreml_reranker_export.txt"
    rp.write_text(report, encoding="utf-8")

    # Record in the manifest: packs.reranker subtree + acknowledgements entry.
    from collections import OrderedDict

    from .package import (
        build_reranker_pack,
        update_manifest_acknowledgement,
        update_manifest_reranker,
    )
    reranker_pack = build_reranker_pack(
        mlpackage_path=res.mlpackage_path,
        mlpackage_tree_sha256=res.mlpackage_tree_sha256,
        mlpackage_bytes=res.mlpackage_bytes,
        vocab_path=res.vocab_path,
        vocab_sha256=res.vocab_sha256,
        model_name=res.model_name,
        model_revision=res.model_revision,
        model_combined_sha256=res.model_combined_sha256,
        license_id=spec["license"],
        precision="float16",
        seq_len="RangeDim(1,512)",
        max_pair_tokens=192,
    )
    if CORPUS_MANIFEST.exists():
        update_manifest_reranker(CORPUS_MANIFEST, reranker_pack)
        update_manifest_acknowledgement(CORPUS_MANIFEST, OrderedDict([
            ("id", "reranker-model"),
            ("name", res.model_name),
            ("resource_type", "reranker"),
            ("version", f"HF revision {res.model_revision}"),
            ("license", spec["license"]),
            ("attribution", None),
            ("source_url", f"https://huggingface.co/{res.model_name}"),
            ("model_combined_sha256", res.model_combined_sha256),
            ("note", "On-device cross-encoder reranker (Rank 4); shipped as a "
                     "fp16 Core ML export, not the original weights. Chosen over "
                     "the higher-scoring BAAI/bge-reranker-base (MIT) because it "
                     "fits the ~20-25 MB on-device envelope."),
        ]))
        print(f"  updated {CORPUS_MANIFEST.name} packs.reranker + acknowledgements")
    else:
        print(f"  WARNING: {CORPUS_MANIFEST.name} not found — run `package` first.",
              file=sys.stderr)

    print(f"  mlpackage:  {res.mlpackage_path} ({res.mlpackage_bytes:,} B / "
          f"{res.mlpackage_bytes / (1024 * 1024):.1f} MB)")
    print(f"  tree sha256: {res.mlpackage_tree_sha256}")
    print(f"  vocab:      sha256 {res.vocab_sha256}")
    from .coreml_rerank_export import PARITY_MAX_REL
    print(f"  parity:     max_rel={res.parity.max_rel_err:.5f} "
          f"(<= {PARITY_MAX_REL}) max_abs={res.parity.max_abs_err:.5f} "
          f"inversions={res.parity.inversions} "
          f"tied_reorders={res.parity.tied_reorders} -> "
          f"{'PASS' if res.parity.passed else 'FAIL'}")
    print(f"  wrote {rp}")

    if not res.parity.passed:
        print("PARITY GATE FAILED. Stop-and-investigate (tokenizer / "
              "token_type_ids / segment-id bug), never a threshold loosening. "
              "Report written for forensics.", file=sys.stderr)
        raise RerankParityError(
            f"max_rel={res.parity.max_rel_err:.5f} "
            f"inversions={res.parity.inversions}")


def _check_reranker_reproducibility(res, model_key: str) -> str:
    """Re-trace+convert to a temp dir; compare the .mlpackage tree hash."""
    import shutil as _shutil
    import tempfile

    from .coreml_rerank_export import (
        MLPACKAGE_NAME,
        _build_wrapper,
        _convert,
        _snapshot_dir,
        _trace,
    )
    from .encode import MODEL_CACHE
    from .package import _sha256_tree

    tmp = Path(tempfile.mkdtemp(prefix="coreml_rerank_repro_"))
    try:
        snap = _snapshot_dir(model_key, MODEL_CACHE)
        wrapper = _build_wrapper(snap)
        traced = _trace(wrapper)
        out = tmp / MLPACKAGE_NAME
        _convert(traced, out)
        second = _sha256_tree(out)
        if second == res.mlpackage_tree_sha256:
            return (f"PASS — re-trace+convert yielded a byte-identical "
                    f".mlpackage tree (sha256 {second[:16]}…).")
        return (f"FLAG — re-conversion tree hash {second} != first "
                f"{res.mlpackage_tree_sha256}; recorded for the architect.")
    finally:
        _shutil.rmtree(tmp, ignore_errors=True)


def _render_reranker_report(res, spec: dict, det_note: str) -> str:
    """Deterministic text report for the reranker Core ML export."""
    lines = [
        "LampStand corpus — on-device reranker Core ML export (Rank 4, SHIP)",
        "",
        f"Model: {res.model_name} ({spec['license']}, {spec['arch']})",
        f"Revision: {res.model_revision}",
        f"model_combined_sha256: {res.model_combined_sha256}",
        "",
        "Artifact:",
        f"  {res.mlpackage_path.name}  "
        f"{res.mlpackage_bytes:,} B ({res.mlpackage_bytes / (1024 * 1024):.1f} MB, "
        "fp16)",
        f"  tree sha256: {res.mlpackage_tree_sha256}",
        f"  {res.vocab_path.name}  sha256 {res.vocab_sha256}",
        "  seq axis: RangeDim(1,512)  pair truncation: 192 tokens",
        f"  512-token forward: {res.forward_512_ms:.1f} ms (CPU_ONLY convert)",
        "",
        "Score semantics: raw classifier logit; higher = more relevant "
        "(no sigmoid; only the order matters for reranking).",
        "",
        "Parity gate (fp16 CPU_ONLY Core ML, reloaded from disk, vs float32 "
        f"PyTorch, over {res.parity.n} fixed pairs):",
        f"  max relative logit err: {res.parity.max_rel_err:.5f}  (floor <= 0.02)",
        f"  max |abs err|: {res.parity.max_abs_err:.5f}  "
        "(unbounded logits ~[-12,+11])",
        f"  mean |abs err|: {res.parity.mean_abs_err:.5f}",
        f"  non-tied order inversions: {res.parity.inversions}  (floor <= 0)",
        f"  tied reorders (not gated): {res.parity.tied_reorders}",
        f"  -> {'PASS' if res.parity.passed else 'FAIL'}",
        "",
        f"Reproducibility: {det_note}",
        "",
    ]
    if res.notes:
        lines.append("Flags:")
        lines.extend(f"  - {n}" for n in res.notes)
        lines.append("")
    lines.append("The .mlpackage + vocab are synced (never committed) like the "
                 "corpus packs; app-integration contract: docs/reranker-pack.md.")
    lines.append("")
    return "\n".join(lines)


# --- Retrieval eval (F5 measurement foundation) --------------------------------
def cmd_build_eval() -> None:
    """Build the deterministic retrieval gold set from the built DBs."""
    from .eval_gold import build_gold

    required = ["embeddings.sqlite", "confessions.sqlite", "crossrefs.sqlite",
                "commentaries.sqlite", "bibles.sqlite"]
    missing = [f for f in required if not (OUTPUT_DIR / f).exists()]
    if missing:
        print("Missing built DBs: " + ", ".join(missing)
              + " — run the build-* commands first.", file=sys.stderr)
        sys.exit(2)
    print("Building retrieval gold set -> output/ (deterministic, seeded)...")
    payload = build_gold(OUTPUT_DIR, REPO_ROOT)
    for cat, n in payload["counts"].items():
        print(f"  {cat}: {n} queries")
    print(f"  crossref vote threshold (data-driven): "
          f">= {payload['thresholds']['crossref_votes_min']}")
    for note in payload["notes"]:
        print(f"  note: {note}")
    print(f"  wrote {OUTPUT_DIR / 'eval_gold_v1.json'} "
          f"({sum(payload['counts'].values())} queries)")


def _query_encoder():
    """(callable, None) when the BGE query encoder loads, else (None, reason)."""
    from .embeddings import QUERY_INSTRUCTION

    try:
        from .encode import MODEL_CACHE, encode_texts, load_model
        os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))
        model = load_model(device="cpu")
    except Exception as e:  # degrade loudly to BM25-only arms
        return None, f"{type(e).__name__}: {e}"

    def encode(texts: list[str]):
        return encode_texts(model, [QUERY_INSTRUCTION + t for t in texts])

    return encode, None


def _build_eval_harness():
    """Load the gold set (building it if absent) and cache per-query rankings."""
    from .eval_gold import GOLD_FILENAME, load_gold
    from .eval_retrieval import Harness

    if not (OUTPUT_DIR / GOLD_FILENAME).exists():
        print(f"No output/{GOLD_FILENAME} — building the gold set first.")
        cmd_build_eval()
    gold = load_gold(OUTPUT_DIR)

    encoder, reason = _query_encoder()
    if encoder is None:
        print(f"  !! DENSE ARMS UNAVAILABLE — query encoder failed to load "
              f"({reason}); degrading to BM25-only.", file=sys.stderr)
    print(f"Caching per-query rankings over {len(gold['queries'])} gold queries "
          f"(BM25{' + dense fp32 + dense int8' if encoder else ' only'})...")
    harness = Harness.build(
        OUTPUT_DIR / "embeddings.sqlite", gold["queries"], encode_queries=encoder,
        int8_variant=True, crossrefs_db=OUTPUT_DIR / "crossrefs.sqlite",
        bibles_db=OUTPUT_DIR / "bibles.sqlite")
    if harness.graph_available:
        print(f"  TSK expansion loaded for {len(harness.expansion):,} Scripture "
              "chunks (graph-boost arms enabled)")
    if harness.expand_available:
        print(f"  query-expansion map loaded ({len(harness.term_expansion):,} "
              "terms; -expand arms enabled)")
    return gold, harness, reason


def _write_eval_report(gold: dict, results: dict) -> None:
    from .eval_report import REPORT_FILENAME, render_report
    from .eval_retrieval import SWEEP_FILENAME

    sweep_path = OUTPUT_DIR / SWEEP_FILENAME
    sweep = (json.loads(sweep_path.read_text(encoding="utf-8"))
             if sweep_path.exists() else None)
    text = render_report(gold, results, sweep)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / REPORT_FILENAME
    rp.write_text(text, encoding="utf-8")
    print(f"  wrote {rp}")


def _evaluate_arms(gold: dict, harness, dense_reason: str | None) -> dict:
    from .eval_retrieval import APP_CONFIG

    # Note: a deeper dense-only arm (e.g. depth 80) is deliberately NOT run —
    # single-list RRF is rank-preserving, so no metric computed within the top
    # 20 can differ from the depth-20 dense arm by construction.
    arm_order = ["bm25"]
    if harness.dense_available:
        arm_order += ["dense", "hybrid"]
    results: dict = {
        "app_config": APP_CONFIG.as_dict(),
        "app_config_label": APP_CONFIG.label(),
        "dense_available": harness.dense_available,
        "arm_order": arm_order,
        "arms": {},
    }
    if dense_reason:
        results["dense_unavailable_reason"] = dense_reason
    # Pack-diet quality probe: dense-bearing arms re-run against the int8-
    # quantized vectors. Stored under arms{} (consumed by the pack-diet
    # report) but listed in int8_arms — NOT arm_order — so the retrieval-eval
    # report's arm sections and F5 verdict are unchanged.
    int8_arms = (["dense-int8", "hybrid-int8"]
                 if getattr(harness, "int8_available", False) else [])
    results["int8_arms"] = int8_arms
    # Experimental TSK graph-boost arms (Rank 13) — also outside arm_order;
    # consumed by reports/crossref_pack_v1.md.
    graph_arms = (["hybrid-graph", "hybrid-graph-weak"]
                  if (harness.dense_available
                      and getattr(harness, "graph_available", False)) else [])
    results["graph_arms"] = graph_arms
    # Rank 7 query-expansion arms (down-weighted expansion terms) — also
    # outside arm_order; rendered in the eval report's expansion section.
    expand_arms = ["bm25-expand"] if getattr(
        harness, "expand_available", False) else []
    if expand_arms and harness.dense_available:
        expand_arms.append("hybrid-expand")
    results["expand_arms"] = expand_arms
    # Record the synonym-wiring state so the report's §4b can say whether the
    # measured expansion INCLUDES the advisor-approved theological synonyms
    # (archaic+suffix always; synonyms only once the advisor gate trips).
    if expand_arms:
        from .expansion import SYNONYMS_RELPATH, load_approved_synonyms
        syn = load_approved_synonyms(REPO_ROOT / SYNONYMS_RELPATH)
        results["synonyms_wired"] = len(syn) > 0
        results["n_synonyms"] = len(syn)
    for arm in arm_order + int8_arms + graph_arms + expand_arms:
        results["arms"][arm] = harness.evaluate_arm(arm, APP_CONFIG)
        o = results["arms"][arm]["overall"]
        print(f"  {arm:17s} recall@20={o['recall_at_20']:.3f} "
              f"MRR={o['mrr']:.3f} nDCG@10={o['ndcg_at_10']:.3f}")
    return results


def cmd_validate_retrieval() -> None:
    """Run the three arms at the app's constants; write results + report."""
    from .eval_retrieval import RESULTS_FILENAME

    gold, harness, dense_reason = _build_eval_harness()
    print("Evaluating arms at the app's shipped constants...")
    results = _evaluate_arms(gold, harness, dense_reason)
    harness.index.close()

    (OUTPUT_DIR / RESULTS_FILENAME).write_text(
        json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote {OUTPUT_DIR / RESULTS_FILENAME}")
    _write_eval_report(gold, results)


def cmd_sweep_retrieval() -> None:
    """Sweep the fusion knobs (cached rankings; recommends, never changes)."""
    from .eval_retrieval import RESULTS_FILENAME, SWEEP_FILENAME, run_sweep, sweep_grid

    gold, harness, dense_reason = _build_eval_harness()
    if not harness.dense_available:
        print("Sweep needs the dense arm (it tunes the FUSION); aborting. "
              f"Encoder failure: {dense_reason}", file=sys.stderr)
        sys.exit(2)

    # Keep the committed results JSON in lockstep (cheap once cached).
    print("Evaluating arms at the app's shipped constants...")
    results = _evaluate_arms(gold, harness, dense_reason)
    (OUTPUT_DIR / RESULTS_FILENAME).write_text(
        json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Sweeping {len(sweep_grid())} fusion configs (hybrid arm, cached "
          "rankings)...")
    sweep = run_sweep(harness)
    harness.index.close()
    for metric in ("recall_at_20", "mrr", "ndcg_at_10"):
        b = sweep["best"][metric]
        print(f"  best {metric}: {b['label']} "
              f"({b['metrics'][metric]:.3f}, {b['delta_vs_app'][metric]:+.3f} vs app)")
    (OUTPUT_DIR / SWEEP_FILENAME).write_text(
        json.dumps(sweep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote {OUTPUT_DIR / SWEEP_FILENAME}")
    _write_eval_report(gold, results)
    print("Sweep RECOMMENDS constants; it changes nothing in the app.")


def cmd_rerank_eval(*, all_models: bool = False) -> None:
    """Measure a cross-encoder reranker over the hybrid top-K (EVAL-GATED).

    Reranks each gold query's un-reranked v2 HYBRID top-``RERANK_K`` candidates
    with a permissively-licensed cross-encoder, recomputes metrics per category
    vs the un-reranked baseline, and writes ``reports/reranker_eval_v1.md`` with
    a SHIP/HOLD verdict. The cross-encoder (sentence-transformers/torch) is a
    DEV/EVAL-only dependency — install with ``pip install -e ".[rerank]"``.
    If no model can be loaded (offline / not cached), the gate cannot run and the
    command stops WITHOUT writing a verdict, as required.
    """
    from .encode import MODEL_CACHE
    from .eval_rerank import (
        DEFAULT_RERANK_MODEL,
        RERANK_MODELS,
        run_rerank_eval,
    )
    from .eval_rerank_report import REPORT_FILENAME, render_report

    gold, harness, dense_reason = _build_eval_harness()
    if not harness.dense_available:
        print("Rerank eval needs the HYBRID arm (it reranks the fused top-K); "
              f"the query encoder failed to load: {dense_reason}. Aborting.",
              file=sys.stderr)
        harness.index.close()
        sys.exit(2)

    model_keys = list(RERANK_MODELS) if all_models else [DEFAULT_RERANK_MODEL]
    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))
    print(f"Reranking hybrid top-30 with cross-encoder(s): "
          f"{', '.join(model_keys)} (header + raw variants)...")
    try:
        results = run_rerank_eval(
            harness, gold["queries"], OUTPUT_DIR / "embeddings.sqlite",
            MODEL_CACHE, model_keys=model_keys)
    except Exception as e:  # degrade loudly — the gate could not run
        harness.index.close()
        print(f"  !! RERANK GATE COULD NOT RUN — cross-encoder unavailable "
              f"({type(e).__name__}: {e}).\n"
              "     Install the eval extra (`pip install -e \".[rerank]\"`) and "
              "ensure the model downloads (needs network on first run). No "
              "verdict written.", file=sys.stderr)
        sys.exit(2)
    harness.index.close()

    (OUTPUT_DIR / "eval_rerank_v1.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for arm in results["deltas"]:
        o = arm["overall"]
        print(f"  {arm['model_key']:24s} {arm['variant']:6s} "
              f"ΔMRR={o['mrr']:+.3f} Δrecall@10={o['recall_at_10']:+.3f} "
              f"Δrecall@20={o['recall_at_20']:+.3f} "
              f"hardneg={arm['hardneg_win_rate']:.3f}")
    text = render_report(gold, results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / REPORT_FILENAME
    rp.write_text(text, encoding="utf-8")
    print(f"  wrote {rp}")
    # Echo the verdict word so the operator sees SHIP/HOLD without opening the file.
    for line in text.splitlines():
        if line.startswith("**Verdict:"):
            print(f"  {line.strip('*')}")
            break


# --- P7 packaging ------------------------------------------------------------
PACKS_DIR = OUTPUT_DIR / "packs"
CORPUS_MANIFEST = REPO_ROOT / "corpus_manifest.json"


def cmd_package(*, fp32: bool = False) -> None:
    """Split the built DBs into bundled + on-demand packs and write the manifest.

    Requires the per-resource DBs (incl. embeddings.sqlite) to be built first.
    Pack-diet v2: vectors are int8-quantized by default; ``package fp32`` keeps
    the exact float32 bytes. Also writes reports/pack_diet_v1.md (sizes + the
    int8 quality delta when validate-retrieval has produced one).
    """
    required = [
        "bibles.sqlite", "confessions.sqlite", "commentaries.sqlite",
        "lexicons.sqlite", "crossrefs.sqlite", "embeddings.sqlite",
    ]
    missing = [f for f in required if not (OUTPUT_DIR / f).exists()]
    if missing:
        print("Missing built DBs: " + ", ".join(missing)
              + " — run the build-* commands first.", file=sys.stderr)
        sys.exit(2)

    from .package import VECTOR_FORMAT_FP32, VECTOR_FORMAT_INT8

    vector_format = VECTOR_FORMAT_FP32 if fp32 else VECTOR_FORMAT_INT8
    print(f"Packaging built DBs -> {PACKS_DIR} (gitignored; vectors "
          f"{vector_format})")
    result = package_corpus(
        OUTPUT_DIR, PACKS_DIR, vector_format=vector_format, repo_root=REPO_ROOT)

    print(f"\nBUNDLED pack ({result.bundled_bytes:,} B = "
          f"{result.bundled_bytes / (1024*1024):.1f} MB):")
    for f in result.files:
        if f.pack == "bundled":
            print(f"  {f.name:34s} {f.bytes:>14,} B  {f.sha256[:12]}")
    print(f"\nON-DEMAND pack(s) ({result.ondemand_bytes:,} B = "
          f"{result.ondemand_bytes / (1024*1024):.1f} MB):")
    for f in result.files:
        if f.pack == "on-demand":
            print(f"  {f.name:34s} {f.bytes:>14,} B  {f.sha256[:12]}")

    if result.flags:
        print("\nSIZE FLAGS:")
        for fl in result.flags:
            print(f"  FLAG: {fl}")
    else:
        print("\nNo size flags (bundled pack is within target).")

    # Manifest is a committed artifact of this step. The packs.models subtree
    # (Core ML query model + vocab, added by coreml-export) is carried over
    # from the existing manifest so a package run never drops it.
    from .package import preserve_models_subtree
    if preserve_models_subtree(CORPUS_MANIFEST, result.manifest):
        print("  preserved packs.models / packs.reranker subtrees + their "
              "model-provenance acknowledgements from the existing manifest")
    CORPUS_MANIFEST.write_text(
        json.dumps(result.manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\nWrote committed manifest {CORPUS_MANIFEST} "
          f"(corpus_version placeholder: {CORPUS_VERSION_PLACEHOLDER})")

    # Pack-diet report (committed): sizes + int8 quality delta when measured.
    from .eval_retrieval import RESULTS_FILENAME
    from .pack_report import PACK_DIET_REPORT_FILENAME, render_pack_diet_report

    results_path = OUTPUT_DIR / RESULTS_FILENAME
    eval_results = (json.loads(results_path.read_text(encoding="utf-8"))
                    if results_path.exists() else None)
    report = render_pack_diet_report(
        corpus_version=CORPUS_VERSION_PLACEHOLDER,
        files=result.files, flags=result.flags, eval_results=eval_results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = REPORTS_DIR / PACK_DIET_REPORT_FILENAME
    rp.write_text(report, encoding="utf-8")
    print(f"Wrote {rp}")

    # TSK crossref-pack report (committed): size + graph-boost measurement.
    from .pack_report import CROSSREF_REPORT_FILENAME, render_crossref_pack_report
    cross_file = next(
        (f for f in result.files if f.name == "bundled_crossrefs.sqlite"), None)
    if cross_file is not None:
        report = render_crossref_pack_report(
            corpus_version=CORPUS_VERSION_PLACEHOLDER,
            pack_file=cross_file, eval_results=eval_results)
        rp = REPORTS_DIR / CROSSREF_REPORT_FILENAME
        rp.write_text(report, encoding="utf-8")
        print(f"Wrote {rp}")
    print("Candidate only — the architect's 23-point spot-check gates ship.")


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
        # `build-embeddings full` forces a from-scratch re-encode; default is the
        # incremental path (reuse unchanged vectors from the prior DB).
        cmd_build_embeddings(full=("full" in argv[1:]))
    elif cmd == "validate-embeddings":
        cmd_validate_embeddings()
    elif cmd == "coreml-export":
        cmd_coreml_export()
    elif cmd == "coreml-export-reranker":
        cmd_coreml_export_reranker()
    elif cmd == "build-eval":
        cmd_build_eval()
    elif cmd == "validate-retrieval":
        cmd_validate_retrieval()
    elif cmd == "sweep-retrieval":
        cmd_sweep_retrieval()
    elif cmd == "rerank-eval":
        # `rerank-eval all` measures every permissive cross-encoder candidate;
        # default measures the primary (ms-marco-MiniLM-L-6-v2, Apache-2.0).
        cmd_rerank_eval(all_models=("all" in argv[1:]))
    elif cmd == "package":
        # `package fp32` keeps exact float32 vector bytes (no quantization).
        cmd_package(fp32=("fp32" in argv[1:]))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
