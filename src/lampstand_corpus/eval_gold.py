"""Retrieval gold-set builder (F5 measurement foundation).

Builds a deterministic query → relevant-chunk gold set from three ZERO-ANNOTATION
label sources already in the built corpus, plus a curated (DRAFT) hard-negative
suite derived from WSC/WLC structure:

  prooftext         confession/catechism proof-texts: the query is the catechism
                    question (or a confession section's opening sentence, inline
                    citations stripped); relevant = the Scripture chunks covering
                    the section's proof-text verses.
  crossref          high-vote TSK edges: the query is the SOURCE verse text (BSB);
                    relevant = the Scripture chunks covering the TARGET verse(s)
                    of every above-threshold edge from that source verse. The
                    vote threshold is data-driven (smallest vote value whose pool
                    still has >= CROSSREF_POOL_MIN edges).
  commentary-anchor commentary paragraph → its Scripture anchor: the query is the
                    paragraph text (word-boundary-truncated); relevant = the
                    Scripture chunks overlapping the paragraph's verse anchor.
  hardneg           WSC/WLC doctrinally-adjacent question pairs (justification vs
                    sanctification etc.), generated deterministically by answer
                    token overlap. Query = question A; relevant = A's Q&A chunk;
                    hard negative = B's. Candidates live in a TRACKED file
                    (data/eval/hard_negatives_v1.json) marked DRAFT — pending
                    theological-advisor review.

Self-hit exclusion: for prooftext / crossref / commentary-anchor queries the
chunk(s) the query text was drawn from are recorded in ``exclude`` and removed
from the candidate pool at retrieval time — otherwise the query's own verbatim
chunk trivially occupies rank 1. Hardneg queries have NO exclusion (their
relevant chunk IS the chunk containing the question).

Verse → chunk anchoring uses the SAME chunk table the embeddings index was built
from (``embeddings.sqlite`` chunk rows), so a "relevant chunk" is exactly a
retrievable unit. A verse maps to the union of the covering pericope chunks
across all four translations (the app dedups translations at hydration; any one
of the four ids identifies the deduped row).

Determinism: fixed seed (:data:`EVAL_SEED`), candidates enumerated in canonical
sorted order before sampling, no timestamps. Same built DBs → byte-identical
gold JSON.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from . import books

# Arbitrary fixed seed — the value is meaningless; only its stability matters.
EVAL_SEED = 613

GOLD_VERSION = "v1"
GOLD_FILENAME = "eval_gold_v1.json"
HARDNEG_RELPATH = Path("data") / "eval" / "hard_negatives_v1.json"

# Sample sizes (stratified; see module docstring).
PROOFTEXT_TARGET = 150
CROSSREF_TARGET = 150
COMMENTARY_PER_SOURCE = 38          # x4 commentators = 152
HARDNEG_TARGET = 60

# Data-driven crossref vote threshold: the smallest vote value whose >=votes
# pool still holds at least this many edges.
CROSSREF_POOL_MIN = 1000

# Commentary paragraphs shorter than this carry too little signal to be queries.
COMMENTARY_MIN_CHARS = 200
# Queries are truncated at a word boundary at/below this many characters.
QUERY_MAX_CHARS = 400
# Confession opening-sentence queries are capped tighter (they are one clause).
SENTENCE_MAX_CHARS = 240

# Hard-negative pair generation: answer-token Jaccard floor + stopwords. The
# stopword set intentionally includes the catechism formula words so overlap is
# driven by doctrinal content ("justification…pardoneth…righteousness"), not by
# "is an act of God's free grace". The floor only prunes the tail — candidates
# are ranked by Jaccard descending and the top pairs are taken; the canonical
# WSC justification/sanctification pair scores ≈0.12 under this stopword set.
HARDNEG_MIN_JACCARD = 0.10
_HARDNEG_STOPWORDS = frozenset(
    "the of and to a in is that it all by for we our us his he him god god's "
    "christ what unto are as with which or be an do doth hath from upon whereby "
    "wherein us's their them they i my me thou thy thee".split()
)

_WORD_RE = re.compile(r"[a-z0-9']+")


# --- data model ---------------------------------------------------------------
@dataclass
class GoldQuery:
    """One eval query: text, relevant chunk ids, self-hit exclusions."""

    id: str
    category: str                 # prooftext | crossref | commentary-anchor | hardneg
    query: str
    relevant: list[str]           # chunk ids (any one satisfies the query)
    exclude: list[str] = field(default_factory=list)
    hard_negative: str | None = None   # chunk id (hardneg category only)
    label: str = ""               # human-readable provenance (anchor / ref)

    def as_dict(self) -> dict:
        d = {
            "id": self.id,
            "category": self.category,
            "query": self.query,
            "relevant": self.relevant,
            "exclude": self.exclude,
            "label": self.label,
        }
        if self.hard_negative is not None:
            d["hard_negative"] = self.hard_negative
        return d


# --- verse -> chunk anchoring ---------------------------------------------------
class ScriptureChunkIndex:
    """Maps canonical verse coordinates to the covering retrieval chunk ids.

    Built from the SAME ``chunk`` rows the embeddings index holds, so the ids
    are exactly the retrievable units. Windows never cross a chapter boundary
    (embeddings.py pericope invariant), so a (book, chapter) bucket lookup is
    complete.
    """

    def __init__(self, embeddings_db: Path) -> None:
        conn = sqlite3.connect(embeddings_db)
        try:
            rows = conn.execute(
                "SELECT id, book, chapter, verse_start, verse_end FROM chunk "
                "WHERE resource_type='scripture' ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        self._by_chapter: dict[tuple[str, int], list[tuple[int, int, str]]] = {}
        for cid, book, ch, vs, ve in rows:
            self._by_chapter.setdefault((book, ch), []).append((vs, ve, cid))

    def covering(self, book: str, chapter: int, verse: int) -> set[str]:
        """Chunk ids (all translations) whose window contains the verse."""
        return {
            cid
            for (vs, ve, cid) in self._by_chapter.get((book, chapter), [])
            if vs <= verse <= ve
        }

    def overlapping(
        self,
        start: tuple[str, int, int],
        end: tuple[str, int, int],
    ) -> set[str]:
        """Chunk ids overlapping the inclusive verse range ``start``..``end``.

        Handles chapter- and book-crossing ranges by walking the canonical
        spine (books.ORDER / VERSE_COUNTS). Malformed (reversed) ranges yield
        the empty set.
        """
        sb, sc, sv = start
        eb, ec, ev = end
        if sb not in books.ORDER_INDEX or eb not in books.ORDER_INDEX:
            return set()
        skey = (books.ORDER_INDEX[sb], sc, sv)
        ekey = (books.ORDER_INDEX[eb], ec, ev)
        if skey > ekey:
            return set()
        out: set[str] = set()
        for bo in range(skey[0], ekey[0] + 1):
            book = books.ORDER[bo]
            counts = books.VERSE_COUNTS.get(book, [])
            ch_first = sc if bo == skey[0] else 1
            ch_last = ec if bo == ekey[0] else len(counts)
            for ch in range(ch_first, ch_last + 1):
                lo = sv if (bo, ch) == (skey[0], sc) else 1
                hi = ev if (bo, ch) == (ekey[0], ec) else (
                    counts[ch - 1] if ch <= len(counts) else 10_000)
                out |= {
                    cid
                    for (vs, ve, cid) in self._by_chapter.get((book, ch), [])
                    if vs <= hi and ve >= lo
                }
        return out


def _chunk_id_by_anchor(conn: sqlite3.Connection, resource_type: str, anchor: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM chunk WHERE resource_type=? AND anchor=?",
        (resource_type, anchor),
    ).fetchone()
    return row[0] if row else None


# --- query-text extraction ------------------------------------------------------
# WSC/WLC sections: "Q33. What is justification?\nA33. Justification is …"
_QA_RE = re.compile(r"^Q\d+\.\s*(?P<q>.*?)\s*\nA\d+\.\s*(?P<a>.*)\Z", re.DOTALL)
# Heidelberg sections: "Question 60. How are thou …? Answer. Only by …"
_HC_RE = re.compile(r"^Question\s+\d+\.\s*(?P<q>.*?)\s*Answer\.\s*(?P<a>.*)\Z", re.DOTALL)
_PAREN_RE = re.compile(r"\([^)]*\)")


def split_catechism(text: str) -> tuple[str, str] | None:
    """Split a WSC/WLC/Heidelberg section into (question, answer), else None."""
    m = _QA_RE.match(text) or _HC_RE.match(text)
    if not m:
        return None
    q = " ".join(m.group("q").split())
    a = " ".join(m.group("a").split())
    return (q, a) if q and a else None


def opening_sentence(text: str, max_chars: int = SENTENCE_MAX_CHARS) -> str:
    """A confession section's opening sentence, inline citations stripped.

    Parenthesised proof-text citations ("(Rom. 8:30; Rom. 3:24)") are removed so
    the query does not leak reference labels; the first '.'-terminated sentence
    is taken, word-boundary-capped at ``max_chars``.
    """
    clean = " ".join(_PAREN_RE.sub(" ", text).split())
    # First sentence boundary: a period followed by a space + uppercase, or EOF.
    m = re.search(r"\.\s+[A-Z0-9]", clean)
    sentence = clean[: m.start() + 1] if m else clean
    return truncate_words(sentence, max_chars)


def truncate_words(text: str, max_chars: int) -> str:
    """Whitespace-normalize and cut at a word boundary at/below ``max_chars``."""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    cut = clean.rfind(" ", 0, max_chars + 1)
    return clean[:cut] if cut > 0 else clean[:max_chars]


# --- source 1: proof-texts -------------------------------------------------------
def prooftext_queries(
    confessions_db: Path,
    emb_conn: sqlite3.Connection,
    scripture: ScriptureChunkIndex,
    *,
    target: int = PROOFTEXT_TARGET,
    seed: int = EVAL_SEED,
) -> tuple[list[GoldQuery], list[str]]:
    """Proof-text queries, stratified per document. Returns (queries, notes)."""
    conn = sqlite3.connect(confessions_db)
    try:
        shortcodes = dict(conn.execute("SELECT id, shortcode FROM document"))
        rows = conn.execute(
            "SELECT document, key, ord, text, proof_texts FROM section "
            "WHERE proof_texts IS NOT NULL ORDER BY document, ord"
        ).fetchall()
    finally:
        conn.close()

    notes: list[str] = []
    candidates: dict[str, list[GoldQuery]] = {}
    for document, key, _ord, text, proofs_json in rows:
        sc = shortcodes.get(document, document.upper())
        anchor = f"{sc} {key}"
        qa = split_catechism(text)
        query = qa[0] if qa else opening_sentence(text)
        if len(query) < 15:
            notes.append(f"prooftext {anchor}: query too short; skipped")
            continue
        relevant: set[str] = set()
        for p in json.loads(proofs_json):
            b, ch = p["book"], p["chapter"]
            vs = p["verse_start"]
            ve = p.get("verse_end", vs)
            relevant |= scripture.overlapping((b, ch, vs), (b, ch, ve))
        if not relevant:
            notes.append(f"prooftext {anchor}: no proof verse resolves to a chunk; skipped")
            continue
        own = _chunk_id_by_anchor(emb_conn, "confession", anchor)
        candidates.setdefault(document, []).append(GoldQuery(
            id=f"pt_{document}_{key}",
            category="prooftext",
            query=query,
            relevant=sorted(relevant - ({own} if own else set())),
            exclude=[own] if own else [],
            label=anchor,
        ))

    total = sum(len(v) for v in candidates.values())
    out: list[GoldQuery] = []
    for document in sorted(candidates):
        pool = candidates[document]
        quota = max(1, round(target * len(pool) / total)) if total else 0
        rng = Random(f"{seed}:prooftext:{document}")
        picked = pool if quota >= len(pool) else rng.sample(pool, quota)
        out.extend(sorted(picked, key=lambda q: q.id))
    return out, notes


# --- source 2: high-vote crossrefs -----------------------------------------------
def crossref_vote_threshold(conn: sqlite3.Connection, pool_min: int = CROSSREF_POOL_MIN) -> int:
    """Smallest vote value whose >=votes pool still has >= ``pool_min`` edges."""
    rows = conn.execute(
        "SELECT votes, count(*) FROM crossref "
        "WHERE src_resolves=1 AND tgt_resolves=1 AND votes > 0 "
        "GROUP BY votes ORDER BY votes DESC"
    ).fetchall()
    running = 0
    threshold = None
    for votes, n in rows:
        running += n
        threshold = votes
        if running >= pool_min:
            break
    return threshold if threshold is not None else 1


def crossref_queries(
    crossrefs_db: Path,
    bibles_db: Path,
    scripture: ScriptureChunkIndex,
    *,
    target: int = CROSSREF_TARGET,
    seed: int = EVAL_SEED,
) -> tuple[list[GoldQuery], int, list[str]]:
    """High-vote TSK queries. Returns (queries, vote_threshold, notes)."""
    conn = sqlite3.connect(crossrefs_db)
    bconn = sqlite3.connect(bibles_db)
    notes: list[str] = []
    try:
        threshold = crossref_vote_threshold(conn)
        edges = conn.execute(
            "SELECT src_book, src_chapter, src_verse, tgt_book, tgt_chapter, "
            "tgt_verse, tgt_end_book, tgt_end_chapter, tgt_end_verse "
            "FROM crossref WHERE src_resolves=1 AND tgt_resolves=1 AND votes>=? ",
            (threshold,),
        ).fetchall()

        # Group edges by source verse; each source verse becomes ONE query whose
        # relevant set is the union of all its above-threshold targets.
        by_source: dict[tuple[str, int, int], list[tuple]] = {}
        for e in edges:
            by_source.setdefault((e[0], e[1], e[2]), []).append(e[3:])

        candidates: list[GoldQuery] = []
        for (sb, sc, sv) in sorted(
            by_source, key=lambda s: (books.ORDER_INDEX.get(s[0], 99), s[1], s[2])
        ):
            row = bconn.execute(
                "SELECT text FROM verse WHERE translation='bsb' AND book=? "
                "AND chapter=? AND verse_start<=? AND verse_end>=?",
                (sb, sc, sv, sv),
            ).fetchone()
            if not row or not (row[0] and row[0].strip()):
                notes.append(f"crossref {sb} {sc}:{sv}: no BSB text (omitted?); skipped")
                continue
            own = scripture.covering(sb, sc, sv)
            relevant: set[str] = set()
            for (tb, tc, tv, teb, tec, tev) in by_source[(sb, sc, sv)]:
                relevant |= scripture.overlapping((tb, tc, tv), (teb, tec, tev))
            relevant -= own
            if not relevant:
                notes.append(
                    f"crossref {sb} {sc}:{sv}: all targets fall in the source's own "
                    f"chunk window; skipped")
                continue
            candidates.append(GoldQuery(
                id=f"xr_{sb}.{sc}.{sv}",
                category="crossref",
                query=truncate_words(row[0], QUERY_MAX_CHARS),
                relevant=sorted(relevant),
                exclude=sorted(own),
                label=f"{sb} {sc}:{sv} (votes>={threshold})",
            ))

        rng = Random(f"{seed}:crossref")
        picked = candidates if target >= len(candidates) else rng.sample(candidates, target)
        return sorted(picked, key=lambda q: q.id), threshold, notes
    finally:
        conn.close()
        bconn.close()


# --- source 3: commentary anchors ------------------------------------------------
def commentary_queries(
    commentaries_db: Path,
    emb_conn: sqlite3.Connection,
    scripture: ScriptureChunkIndex,
    *,
    per_source: int = COMMENTARY_PER_SOURCE,
    seed: int = EVAL_SEED,
) -> tuple[list[GoldQuery], list[str]]:
    """Commentary paragraph → Scripture-anchor queries, per-commentator quota."""
    conn = sqlite3.connect(commentaries_db)
    notes: list[str] = []
    out: list[GoldQuery] = []
    try:
        commentators = [r[0] for r in conn.execute(
            "SELECT id FROM commentator ORDER BY id")]
        for commentator in commentators:
            keys = [r[0] for r in conn.execute(
                "SELECT key FROM comment WHERE commentator=? AND verse_start>0 "
                "AND length(text)>=? ORDER BY ord",
                (commentator, COMMENTARY_MIN_CHARS),
            )]
            rng = Random(f"{seed}:commentary:{commentator}")
            # Over-sample 2x, then keep the first `per_source` that resolve —
            # deterministic backfill for paragraphs whose anchor has no chunk.
            n_draw = min(len(keys), per_source * 2)
            drawn = rng.sample(keys, n_draw) if n_draw < len(keys) else list(keys)
            kept = 0
            for key in drawn:
                if kept >= per_source:
                    break
                row = conn.execute(
                    "SELECT book, chapter, verse_start, verse_end, text "
                    "FROM comment WHERE commentator=? AND key=?",
                    (commentator, key),
                ).fetchone()
                book, ch, vs, ve, text = row
                relevant = scripture.overlapping((book, ch, vs), (book, ch, max(ve, vs)))
                if not relevant:
                    notes.append(
                        f"commentary {commentator}:{key}: anchor resolves to no chunk; skipped")
                    continue
                own = emb_conn.execute(
                    "SELECT id FROM chunk WHERE resource_type='commentary' "
                    "AND source=? AND key=?",
                    (commentator, key),
                ).fetchone()
                out.append(GoldQuery(
                    id=f"ca_{commentator}_{key}",
                    category="commentary-anchor",
                    query=truncate_words(text, QUERY_MAX_CHARS),
                    relevant=sorted(relevant),
                    exclude=[own[0]] if own else [],
                    label=f"{commentator}:{key}",
                ))
                kept += 1
            if kept < per_source:
                notes.append(
                    f"commentary {commentator}: only {kept}/{per_source} usable candidates")
    finally:
        conn.close()
    return sorted(out, key=lambda q: q.id), notes


# --- source 4: hard negatives (WSC/WLC structure) ---------------------------------
def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        t for t in _WORD_RE.findall(text.lower()) if t not in _HARDNEG_STOPWORDS)


def generate_hard_negative_candidates(
    confessions_db: Path,
    *,
    target: int = HARDNEG_TARGET,
) -> list[dict]:
    """Doctrinally-adjacent WSC/WLC question pairs by answer token overlap.

    Deterministic: pairs ranked by (Jaccard desc, document, key pair asc); the
    output stores ANCHORS (not chunk ids) so the tracked file survives corpus
    rebuilds. Each question appears in at most one triple (greedy, so the suite
    spans many doctrines instead of clustering on one formula family).
    """
    conn = sqlite3.connect(confessions_db)
    try:
        shortcodes = dict(conn.execute("SELECT id, shortcode FROM document"))
        sections: dict[str, list[tuple[str, str, str]]] = {}
        for document in ("wsc", "wlc"):
            for key, text in conn.execute(
                "SELECT key, text FROM section WHERE document=? ORDER BY ord",
                (document,),
            ):
                qa = split_catechism(text)
                if qa:
                    sections.setdefault(document, []).append((key, qa[0], qa[1]))
    finally:
        conn.close()

    scored: list[tuple[float, str, str, str]] = []
    for document, rows in sorted(sections.items()):
        toks = {key: _content_tokens(answer) for key, _q, answer in rows}
        keys = [key for key, _q, _a in rows]
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                ta, tb = toks[ka], toks[kb]
                if not ta or not tb:
                    continue
                j = len(ta & tb) / len(ta | tb)
                if j >= HARDNEG_MIN_JACCARD:
                    scored.append((j, document, ka, kb))
    scored.sort(key=lambda s: (-s[0], s[1], _numkey(s[2]), _numkey(s[3])))

    q_by_key = {
        (document, key): (q, a)
        for document, rows in sections.items()
        for key, q, a in rows
    }
    used: set[tuple[str, str]] = set()
    out: list[dict] = []
    for j, document, ka, kb in scored:
        if len(out) >= target:
            break
        if (document, ka) in used or (document, kb) in used:
            continue
        used.add((document, ka))
        used.add((document, kb))
        sc = shortcodes.get(document, document.upper())
        out.append({
            "id": f"hn_{document}_{ka}_{kb}",
            "document": document,
            "query": q_by_key[(document, ka)][0],
            "relevant_anchor": f"{sc} {ka}",
            "hard_negative_anchor": f"{sc} {kb}",
            "hard_negative_question": q_by_key[(document, kb)][0],
            "answer_jaccard": round(j, 4),
        })
    return out


def _numkey(key: str) -> tuple:
    """Sort catechism keys numerically when possible ('9' < '10')."""
    return (0, int(key)) if key.isdigit() else (1, key)


def write_hard_negatives(path: Path, candidates: list[dict]) -> None:
    """Write the TRACKED hard-negative candidate file (DRAFT header field)."""
    payload = {
        "status": "DRAFT — pending theological-advisor review",
        "version": GOLD_VERSION,
        "method": (
            "WSC/WLC same-document question pairs ranked by answer content-token "
            f"Jaccard (stopworded), floor {HARDNEG_MIN_JACCARD}, greedy one-use-"
            "per-question; deterministic (no RNG)."
        ),
        "review_note": (
            "Each triple asserts that question A's answer chunk should outrank "
            "question B's for query A. An advisor may strike or amend pairs; "
            "edits to this file are honored verbatim by build-eval."
        ),
        "triples": candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def hardneg_queries(
    hardneg_path: Path,
    confessions_db: Path,
    emb_conn: sqlite3.Connection,
) -> tuple[list[GoldQuery], list[str]]:
    """Load (or generate + write) the hard-negative suite as gold queries."""
    notes: list[str] = []
    if not hardneg_path.exists():
        candidates = generate_hard_negative_candidates(confessions_db)
        write_hard_negatives(hardneg_path, candidates)
        notes.append(
            f"hardneg: generated {len(candidates)} DRAFT candidates -> "
            f"{HARDNEG_RELPATH.as_posix()}")
    payload = json.loads(hardneg_path.read_text(encoding="utf-8"))
    if "DRAFT" in str(payload.get("status", "")):
        notes.append("hardneg: suite is DRAFT — pending theological-advisor review")

    out: list[GoldQuery] = []
    for t in payload["triples"]:
        rel = _chunk_id_by_anchor(emb_conn, "confession", t["relevant_anchor"])
        neg = _chunk_id_by_anchor(emb_conn, "confession", t["hard_negative_anchor"])
        if rel is None or neg is None:
            notes.append(f"hardneg {t['id']}: anchor missing from chunk table; skipped")
            continue
        out.append(GoldQuery(
            id=t["id"],
            category="hardneg",
            query=t["query"],
            relevant=[rel],
            exclude=[],
            hard_negative=neg,
            label=f"{t['relevant_anchor']} vs {t['hard_negative_anchor']}",
        ))
    return sorted(out, key=lambda q: q.id), notes


# --- top-level build ---------------------------------------------------------------
def build_gold(output_dir: Path, repo_root: Path) -> dict:
    """Build the full gold set. Returns the JSON-ready payload (also written)."""
    emb_db = output_dir / "embeddings.sqlite"
    emb_conn = sqlite3.connect(emb_db)
    try:
        meta = dict(emb_conn.execute("SELECT key, value FROM meta").fetchall())
        scripture = ScriptureChunkIndex(emb_db)

        pt, pt_notes = prooftext_queries(
            output_dir / "confessions.sqlite", emb_conn, scripture)
        xr, threshold, xr_notes = crossref_queries(
            output_dir / "crossrefs.sqlite", output_dir / "bibles.sqlite", scripture)
        ca, ca_notes = commentary_queries(
            output_dir / "commentaries.sqlite", emb_conn, scripture)
        hn, hn_notes = hardneg_queries(
            repo_root / HARDNEG_RELPATH, output_dir / "confessions.sqlite", emb_conn)
    finally:
        emb_conn.close()

    queries = pt + xr + ca + hn
    corpus_version = ""
    manifest = repo_root / "corpus_manifest.json"
    if manifest.exists():
        corpus_version = json.loads(
            manifest.read_text(encoding="utf-8")).get("corpus_version", "")

    payload = {
        "version": GOLD_VERSION,
        "seed": EVAL_SEED,
        "corpus": {
            "corpus_version": corpus_version,
            "model_revision": meta.get("model_revision", ""),
            "n_chunks": int(meta.get("n_chunks", 0)),
        },
        "thresholds": {"crossref_votes_min": threshold},
        "counts": _category_counts(queries),
        "notes": pt_notes + xr_notes + ca_notes + hn_notes,
        "queries": [q.as_dict() for q in queries],
    }
    out_path = output_dir / GOLD_FILENAME
    out_path.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _category_counts(queries: list[GoldQuery]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for q in queries:
        counts[q.category] = counts.get(q.category, 0) + 1
    return dict(sorted(counts.items()))


def load_gold(output_dir: Path) -> dict:
    """Read the built gold set (build_gold must have run)."""
    path = output_dir / GOLD_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))
