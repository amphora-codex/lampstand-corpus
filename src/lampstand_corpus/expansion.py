"""Rank 7 (mining half) — query-expansion table: archaic→modern + suffix classes.

Three deterministic products, zero hand-curation in the shipped table:

1. ARCHAIC→MODERN pairs mined from the aligned KJV/ASV/WEB/BSB parallel verses:
   a token that occurs in the archaic translations (KJV/ASV) but NEVER in the
   modern ones (BSB/WEB) is paired with the modern token that co-occurs in the
   BSB rendering of the same verses with the highest containment
   (``believeth→believe``, ``saith→says``, ``thee→you``). Mechanical pairs
   (same word, different inflection/orthography — detected by shared prefix)
   ship in the expansion table, BIDIRECTIONALLY, so a modern query matches KJV
   text and vice versa.

2. SUFFIX-EQUIVALENCE classes over the corpus BM25 vocabulary: a light,
   vocabulary-validated suffix stripper (``justify/justified/justification``,
   ``believe/believeth/believes``) — a stripped stem counts only when the stem
   (or stem+e) itself exists in the vocabulary, so this never invents words.
   Members of a class expand to each other.

3. THEOLOGICAL synonym candidates (``charity→love``, ``propitiation→atoning``):
   the NON-mechanical residue of the archaic mining — lexical substitutions,
   not inflections. These are doctrinally sensitive, so they are written to a
   TRACKED DRAFT file (``data/eval/theological_synonyms_v1.json``, status:
   DRAFT — pending theological-advisor review) and are NOT shipped in the
   expansion table until approved; once the advisor marks the file approved,
   ``build_expansion_rows`` folds the approved pairs in automatically.

The shipped table (search pack, ``expansion`` table, format ``expansion-v1``):
``(term, expansion, kind archaic|suffix|synonym, weight)``. Weights are the
mining containment score (archaic/synonym) or 1.0 (suffix); the RETRIEVER
applies its own global down-weight to expansion terms at query time (the eval
harness uses ``EXPANSION_TERM_WEIGHT``) — table weights are informational for
future per-pair tuning.

Deterministic: fixed thresholds, sorted iteration, no RNG, no timestamps.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .embeddings import tokenize

EXPANSION_FORMAT = "expansion-v1"
GLOSS_FORMAT = "gloss-v1"
SYNONYMS_RELPATH = Path("data") / "eval" / "theological_synonyms_v1.json"

# Mining thresholds (documented; all deterministic).
ARCHAIC_MIN_DF = 3          # archaic token must appear in >= this many verses
ARCHAIC_MIN_SCORE = 0.55    # containment of the best modern partner
# The partner must be ENRICHED in the archaic term's verses vs its base rate —
# P(m | V(a)) / P(m) — which admits "saith→says" while rejecting "thee→the".
ARCHAIC_MIN_LIFT = 5.0
SUFFIX_MIN_DF = 5           # vocabulary df floor for suffix-class members
SUFFIX_MIN_STEM = 4         # stems shorter than this are never stripped to
SUFFIX_MAX_CLASS = 8        # cap on members per class
# The eval harness's flat down-weight for expansion terms at query time.
EXPANSION_TERM_WEIGHT = 0.3

# Light suffix rules, ORDERED longest-first. (suffix, replacement) — the
# resulting stem must itself be a vocabulary word (directly or via +e).
_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ications", "y"), ("ication", "y"),   # justification -> justify
    ("ousness", "ous"), ("fulness", "ful"),  # righteousness -> righteous
    ("ingly", ""), ("edly", ""),
    ("ieth", "y"),                         # justifieth -> justify
    ("eth", ""), ("est", ""),              # believeth -> believe
    ("ings", ""), ("ing", ""),
    ("ied", "y"), ("ies", "y"),            # justified -> justify
    ("ed", ""), ("es", ""), ("s", ""),
]


# --- archaic mining -----------------------------------------------------------
def _verse_tokens(bibles_db: Path) -> dict[str, dict[tuple, set[str]]]:
    """{translation: {(book, chapter, verse_start): token set}}."""
    conn = sqlite3.connect(bibles_db)
    out: dict[str, dict[tuple, set[str]]] = {}
    try:
        for tid, b, ch, vs, text in conn.execute(
                "SELECT translation, book, chapter, verse_start, text "
                "FROM verse WHERE text <> ''"):
            out.setdefault(tid, {})[(b, ch, vs)] = set(tokenize(text))
    finally:
        conn.close()
    return out


def mine_archaic_pairs(bibles_db: Path) -> list[dict]:
    """Archaic→modern pairs from parallel-verse co-occurrence.

    Returns ``[{archaic, modern, score, n_verses, mechanical}, ...]`` sorted by
    archaic token. ``mechanical`` marks inflection/orthography pairs (shared
    prefix >= 4 chars covering >= half the shorter token) — the shippable kind;
    non-mechanical pairs are theological-synonym CANDIDATES (DRAFT file).
    """
    by_tid = _verse_tokens(bibles_db)
    archaic_src = [t for t in ("asv", "kjv") if t in by_tid]
    modern_src = [t for t in ("bsb", "web") if t in by_tid]
    if not archaic_src or "bsb" not in by_tid:
        return []

    # Vocabulary df per side.
    def df(tids: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tid in tids:
            for toks in by_tid[tid].values():
                for t in toks:
                    counts[t] = counts.get(t, 0) + 1
        return counts

    df_arch = df(archaic_src)
    del modern_src  # candidacy tests against the BSB spine only (below)
    bsb = by_tid["bsb"]
    n_bsb_verses = len(bsb)
    df_bsb: dict[str, int] = {}
    for toks in bsb.values():
        for t in toks:
            df_bsb[t] = df_bsb.get(t, 0) + 1

    # Verse sets for candidate archaic tokens: absent from the BSB spine (WEB
    # retains some archaisms — e.g. "charity" — so it does not veto candidacy).
    candidates = {
        t for t, n in df_arch.items()
        if n >= ARCHAIC_MIN_DF and t not in df_bsb and t.isalpha()
    }
    verses_of: dict[str, set[tuple]] = {t: set() for t in candidates}
    for tid in archaic_src:
        for ref, toks in by_tid[tid].items():
            for t in toks & candidates:
                verses_of[t].add(ref)

    out: list[dict] = []
    for a in sorted(candidates):
        refs = [r for r in verses_of[a] if r in bsb]
        if len(refs) < ARCHAIC_MIN_DF:
            continue
        counts: dict[str, int] = {}
        for r in refs:
            for m in bsb[r]:
                counts[m] = counts.get(m, 0) + 1
        def _prefix(x: str, y: str) -> int:
            n = 0
            for cx, cy in zip(x, y, strict=False):
                if cx != cy:
                    break
                n += 1
            return n

        eligible = []
        for m in sorted(counts):
            if m == a or not m.isalpha() or len(m) < 3:
                continue
            score = counts[m] / len(refs)
            if score < ARCHAIC_MIN_SCORE:
                continue
            base_rate = df_bsb.get(m, 0) / n_bsb_verses
            if base_rate <= 0 or score / base_rate < ARCHAIC_MIN_LIFT:
                continue  # not enriched vs base rate (near-stopword partner)
            eligible.append((score, m))
        if not eligible:
            continue
        # Best partner: containment desc, rarer first, then prefix affinity to
        # the archaic form (breaks exact ties toward the inflection partner),
        # then token asc — fully deterministic.
        score, m = min(
            eligible,
            key=lambda e: (-e[0], df_bsb.get(e[1], 0), -_prefix(a, e[1]), e[1]))
        prefix = 0
        for x, y in zip(a, m, strict=False):
            if x != y:
                break
            prefix += 1
        mechanical = prefix >= 3 and prefix * 2 >= min(len(a), len(m))
        out.append({
            "archaic": a, "modern": m, "score": round(score, 4),
            "n_verses": len(refs), "mechanical": mechanical,
        })
    return out


# --- suffix classes -------------------------------------------------------------
def suffix_classes(vocab_df: dict[str, int]) -> dict[str, list[str]]:
    """Vocabulary-validated suffix-equivalence classes.

    ``vocab_df`` maps term → document frequency. Returns {stem: sorted members}
    for classes with >= 2 members (df >= SUFFIX_MIN_DF each, capped at
    SUFFIX_MAX_CLASS by df desc then term asc).
    """
    def stem_of(term: str) -> str | None:
        for suf, rep in _SUFFIX_RULES:
            if term.endswith(suf) and len(term) - len(suf) + len(rep) >= SUFFIX_MIN_STEM:
                base = term[: len(term) - len(suf)] + rep
                for cand in (base, base + "e"):
                    if cand in vocab_df and cand != term:
                        return cand
        return None

    classes: dict[str, set[str]] = {}
    for term, n in vocab_df.items():
        if n < SUFFIX_MIN_DF or not term.isalpha():
            continue
        stem = stem_of(term)
        if stem is not None and vocab_df.get(stem, 0) >= SUFFIX_MIN_DF:
            classes.setdefault(stem, set()).add(term)
    out: dict[str, list[str]] = {}
    for stem in sorted(classes):
        members = sorted(
            classes[stem] | {stem},
            key=lambda t: (-vocab_df.get(t, 0), t))[:SUFFIX_MAX_CLASS]
        if len(members) >= 2:
            out[stem] = sorted(members)
    return out


# --- DRAFT synonym file ------------------------------------------------------------
def write_synonym_candidates(path: Path, pairs: list[dict]) -> None:
    """Write the TRACKED theological-synonym candidate file (DRAFT header)."""
    payload = {
        "status": "DRAFT — pending theological-advisor review",
        "format": EXPANSION_FORMAT,
        "method": (
            "Non-mechanical residue of the archaic→modern parallel-verse "
            "mining: lexical substitutions (charity→love), not inflections. "
            "NOT shipped in the expansion table until this file's status is "
            "changed to APPROVED by the advisor; entries may be struck or "
            "amended and are honored verbatim."
        ),
        "pairs": pairs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def load_approved_synonyms(path: Path) -> list[dict]:
    """Approved synonym pairs from the tracked file ([] until ADVISOR-approved).

    The theological-synonym gate is specifically the theological ADVISOR's gate
    (synonym wiring changes query-time scoring and must not ship on architect
    clearance alone). So a file that is ``status: approved`` by the ARCHITECT —
    but not yet advisor-signed — stays unwired: this returns ``[]`` for it. The
    file is wired ONLY when it records advisor approval, i.e. either
    ``approved_by`` names the advisor or the status reads "APPROVED by advisor".
    Architect clearance is recorded in the header for provenance but does not
    trip this gate.
    """
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload.get("status", ""))
    approved_by = str(payload.get("approved_by", ""))
    if not status.upper().startswith("APPROVED"):
        return []
    advisor_signed = (
        "advisor" in approved_by.lower() or "advisor" in status.lower())
    if not advisor_signed:
        return []  # architect-only clearance: unwired (measurement-null)
    return payload.get("pairs", [])


# --- shipped table rows -----------------------------------------------------------
def build_expansion_rows(
    bibles_db: Path,
    vocab_df: dict[str, int],
    repo_root: Path | None = None,
) -> tuple[list[tuple[str, str, str, float]], dict]:
    """All ``(term, expansion, kind, weight)`` rows for the search pack.

    Archaic pairs ship bidirectionally; suffix classes all-to-all within a
    class; synonym pairs only once the tracked DRAFT file is APPROVED. When a
    ``repo_root`` is given and the DRAFT file is absent, the candidate file is
    generated (tracked, for the advisor).

    The same (term, expansion) pair can arrive from MULTIPLE sources (e.g.
    believeth↔believes is both a mined archaic pair and a suffix-class pair).
    The table's primary key is (term, expansion), so pairs are deduped with a
    DOCUMENTED, arrival-order-independent precedence: the higher ``weight``
    wins; on a weight tie the kind wins by fixed rank synonym > archaic >
    suffix (curated > mined > rule). Rows are unique and sorted.
    """
    kind_rank = {"synonym": 0, "archaic": 1, "suffix": 2}
    best: dict[tuple[str, str], tuple[float, str]] = {}

    def add(term: str, exp: str, kind: str, weight: float) -> None:
        cur = best.get((term, exp))
        if cur is None or (-weight, kind_rank[kind]) < (-cur[0], kind_rank[cur[1]]):
            best[(term, exp)] = (weight, kind)

    mined = mine_archaic_pairs(bibles_db)
    for p in mined:
        if not p["mechanical"]:
            continue
        w = float(p["score"])
        add(p["archaic"], p["modern"], "archaic", w)
        add(p["modern"], p["archaic"], "archaic", w)
    for _stem, members in suffix_classes(vocab_df).items():
        for a in members:
            for b in members:
                if a != b:
                    add(a, b, "suffix", 1.0)

    # DRAFT candidates: strongest evidence first, both tokens content-shaped,
    # capped so the advisor's review stays tractable.
    candidates = sorted(
        (p for p in mined
         if not p["mechanical"] and len(p["archaic"]) >= 4
         and len(p["modern"]) >= 4),
        key=lambda p: (-p["n_verses"], p["archaic"]))[:100]
    if repo_root is not None:
        syn_path = repo_root / SYNONYMS_RELPATH
        if not syn_path.exists():
            write_synonym_candidates(syn_path, candidates)
        for p in load_approved_synonyms(syn_path):
            w = float(p.get("score", 1.0))
            add(p["archaic"], p["modern"], "synonym", w)
            add(p["modern"], p["archaic"], "synonym", w)

    rows = sorted(
        (term, exp, kind, weight)
        for (term, exp), (weight, kind) in best.items())
    stats = {
        "n_archaic_pairs": sum(1 for p in mined if p["mechanical"]),
        "n_synonym_candidates": len(candidates),
        "n_suffix_classes": len(suffix_classes(vocab_df)),
        "n_rows": len(rows),
    }
    return rows, stats


# --- directional gloss rows (tap-to-gloss; display-only) --------------------------
def build_gloss_rows(
    bibles_db: Path,
    repo_root: Path | None = None,
) -> tuple[list[tuple[str, str, str, float]], dict]:
    """One-way ``(term, modern_gloss, kind, weight)`` rows for tap-to-gloss.

    Unlike the symmetric ``expansion`` table (which ships both directions so a
    modern query can match archaic text and vice versa), this is the DIRECTIONAL
    slice the app's tap-to-gloss needs: the archaic surface form → its plain
    modern gloss, and ONLY that direction. It is DISPLAY-ONLY and never enters
    query-time scoring (the retriever never reads it).

    Source is the SAME ``mine_archaic_pairs`` directional data that feeds the
    expansion table, so provenance is identical:

    - ``kind="archaic"`` — mechanical inflection/orthography residue
      (``believeth→believes``, ``sepulchre`` stays a synonym below): the archaic
      surface form and its mined modern partner.
    - ``kind="synonym"`` — the non-mechanical lexical residue, folded in ONLY
      once the advisor has approved ``theological_synonyms_v1.json`` (same gate
      as the expansion table's synonym rows: ``load_approved_synonyms``).
      ``sepulchre→tomb``, ``candlestick→lampstand``, ``devils→demons``, etc.

    Every archaic key is, by construction of the mining, ABSENT from the BSB
    spine (``a not in df_bsb``), so a gloss only ever maps a genuinely archaic
    surface form forward — never a modern word. The primary key is ``term``
    (one canonical gloss per surface form); on the rare collision the higher
    mining ``weight`` wins, then kind rank synonym > archaic, then gloss asc —
    fully deterministic. Pairs not present in the mined data are NOT fabricated.
    """
    kind_rank = {"synonym": 0, "archaic": 1}
    best: dict[str, tuple[str, str, float]] = {}  # term -> (gloss, kind, weight)

    def add(term: str, gloss: str, kind: str, weight: float) -> None:
        if term == gloss:
            return
        cur = best.get(term)
        if cur is None:
            best[term] = (gloss, kind, weight)
            return
        cur_key = (-cur[2], kind_rank[cur[1]], cur[0])
        new_key = (-weight, kind_rank[kind], gloss)
        if new_key < cur_key:
            best[term] = (gloss, kind, weight)

    mined = mine_archaic_pairs(bibles_db)
    n_mechanical = 0
    for p in mined:
        if not p["mechanical"]:
            continue
        add(p["archaic"], p["modern"], "archaic", float(p["score"]))
        n_mechanical += 1

    n_synonym = 0
    if repo_root is not None:
        syn_path = repo_root / SYNONYMS_RELPATH
        for p in load_approved_synonyms(syn_path):
            add(p["archaic"], p["modern"], "synonym", float(p.get("score", 1.0)))
            n_synonym += 1

    rows = sorted(
        (term, gloss, kind, weight)
        for term, (gloss, kind, weight) in best.items())
    stats = {
        "format": GLOSS_FORMAT,
        "direction": "archaic-to-modern",
        "n_mechanical": n_mechanical,
        "n_synonym_approved": n_synonym,
        "n_rows": len(rows),
    }
    return rows, stats


def load_expansion_map(
    rows: list[tuple[str, str, str, float]],
) -> dict[str, list[str]]:
    """{term: [expansion terms]} for the query-time expander (order stable)."""
    out: dict[str, list[str]] = {}
    for term, exp, _kind, _w in rows:
        out.setdefault(term, []).append(exp)
    return {t: sorted(set(v)) for t, v in out.items()}
