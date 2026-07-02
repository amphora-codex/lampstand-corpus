"""P6 — chunk extraction, BGE-small encoding, and a deterministic BM25 tokenizer.

This module reads the *already built* per-resource SQLite databases (bibles,
commentaries, confessions, lexicons) and turns them into the retrieval chunks the
app's hybrid (dense + sparse) retriever consumes. Cross-references are a graph,
not prose, and are deliberately excluded (spec §4.3).

Chunk granularity (the Rank-8 re-chunk; the ONE id-changing release):
  * Scripture   — DUAL granularity. PARENTS are real pericopes delimited by the
    BSB's ``\\s`` section headings (bibles.sqlite ``heading`` table; boundaries
    shared across all four translations so parents stay range-aligned for the
    translation dedup), with the deterministic ``PERICOPE_TARGET``-verse window
    as the per-chapter fallback where no headings exist, and an oversized-
    section sub-split at ``PARENT_MAX`` verses. Parents are CONTEXT-ONLY
    (``indexed=False``): no BM25 postings, no vectors. CHILDREN are single
    verse rows (bridges kept as-is), retrieval-precise, each linked to its
    parent via ``Chunk.parent`` — retrieve at verse precision, expand to the
    parent for LLM context. Only BSB children carry dense vectors (``embed``);
    KJV/ASV/WEB children stay BM25-only (Rank 8d).
  * Commentary  — paragraph-level; paragraphs that would exceed the encoder's
    512-wordpiece window are split at sentence boundaries into SIBLINGS sharing
    the verse anchor (Rank 8c; ``COMMENT_SPLIT_CHARS``).
  * Confessions — section / Q&A level, carrying the catechism metadata
    (question / Lord's Day / chapter.section / article — Rank 14).
  * Lexicons    — entry-level (Strong's / BDB / TBESG definitions).

Every indexed chunk carries a deterministic STRUCTURAL HEADER built purely from
curated fields ("Psalms 23:1 — ", "Calvin on Genesis 1:1 — ", "Strong's Greek
G26 — ", "Westminster Shorter Catechism Q. 33 — "); the embedded and
BM25-indexed text is ``Chunk.index_text`` = header + body (Rank 8e). The body
``text`` stays clean for display.

Every chunk carries its provenance anchor (resource type, source id, and a
VerseRef or key) so a retrieval hit resolves back to a real location.

Determinism: chunk *id*s are stable content-addressed strings (the checksum
covers header + body, so a header change re-keys the chunk), chunks are emitted
in a fixed canonical order, and encoding is run on CPU with fixed seeds (see
``encode_chunks``). The BM25 tokenizer is a plain lowercase word tokenizer with
no stemming — the same text always yields the same tokens.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import books

# --- Embedding model identity (spec §4.2; recorded in the manifest) ----------
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Pinned HuggingFace revision — the model is an input to reproducibility, so the
# exact commit is fixed here and its file hashes are recorded at build time.
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBED_DIM = 384  # BGE-small hidden size
# BGE retrieval convention: queries are prefixed; passages are embedded bare.
# We store passage (chunk) vectors here; the app prepends this to user queries.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Deterministic pericope window for Scripture chunking (verses per PARENT) in
# chapters with no BSB section headings.
PERICOPE_TARGET = 10
# A trailing remainder smaller than this is folded into the previous window so we
# don't emit 1-2 verse orphan chunks at chapter ends.
PERICOPE_MIN_TAIL = 4
# A heading-delimited section longer than this many verses is sub-split into
# PERICOPE_TARGET windows so an LLM-context parent stays a readable unit.
PARENT_MAX = 25

# Cap on chunk text length fed to the encoder (characters). BGE truncates at 512
# tokens internally; we record any chunk we hard-truncate so it's auditable.
MAX_CHARS = 6000

# Commentary paragraphs whose header+body exceeds this many characters are split
# at sentence boundaries into siblings (Rank 8c). Calibrated against the pinned
# BGE tokenizer on the real corpus: long commentary prose runs ≥ 4.0 chars per
# wordpiece at the 5th percentile, so 1800 chars ≈ ≤ 450 wordpieces — safely
# under the encoder's 512 window including the structural header.
COMMENT_SPLIT_CHARS = 1800


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit, resource-agnostic, with a resolvable anchor."""

    id: str               # stable content+anchor hash
    resource_type: str    # 'scripture' | 'commentary' | 'confession' | 'lexicon'
    source: str           # e.g. 'bsb', 'henry', 'wcf', 'strongs-greek'
    anchor: str           # human/resolvable anchor: "GEN 1:1-5", "WCF 11.1", "G25"
    book: str | None      # canonical book id when Scripture-anchored, else None
    chapter: int | None
    verse_start: int | None
    verse_end: int | None
    key: str | None       # lexicon Strong's / confession section key, else None
    text: str             # display body text (already trimmed)
    text_checksum: str    # SHA-256 of header + body (chunk-stability audit)
    truncated: bool = False
    # Rank 8: structural header ("Psalms 23:1 — "); index_text = header + text.
    header: str = ""
    # Scripture dual granularity: children link to their pericope parent's id.
    parent: str | None = None
    # False for context-only parents: no BM25 postings, no vector, never ranked.
    indexed: bool = True
    # True when this chunk gets a dense vector (indexed AND (non-scripture OR
    # the BSB spine) — Rank 8d).
    embed: bool = True
    # Rank 14 catechism metadata (confession chunks only; None elsewhere).
    question: int | None = None
    lords_day: int | None = None
    conf_chapter: int | None = None
    conf_section: int | None = None
    article: int | None = None

    @property
    def index_text(self) -> str:
        """The text actually embedded and BM25-indexed (header + body)."""
        return f"{self.header}{self.text}" if self.header else self.text


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_id(resource_type: str, source: str, anchor: str, text_checksum: str) -> str:
    """Stable, collision-resistant id: never depends on insertion order."""
    h = hashlib.sha256()
    h.update(f"{resource_type}\x00{source}\x00{anchor}\x00{text_checksum}".encode())
    return f"{resource_type[:3]}_{h.hexdigest()[:24]}"


def _mk_chunk(
    resource_type: str,
    source: str,
    anchor: str,
    text: str,
    *,
    book: str | None = None,
    chapter: int | None = None,
    verse_start: int | None = None,
    verse_end: int | None = None,
    key: str | None = None,
    header: str = "",
    parent: str | None = None,
    indexed: bool = True,
    embed: bool | None = None,
    question: int | None = None,
    lords_day: int | None = None,
    conf_chapter: int | None = None,
    conf_section: int | None = None,
    article: int | None = None,
) -> Chunk:
    trimmed = text.strip()
    truncated = False
    if len(trimmed) > MAX_CHARS:
        trimmed = trimmed[:MAX_CHARS]
        truncated = True
    # The checksum covers the INDEXED text (header + body): a header-format
    # change re-keys the chunk, so an incremental encode can never wrongly
    # reuse a vector embedded under a different header.
    cs = _checksum(f"{header}{trimmed}" if header else trimmed)
    if embed is None:
        embed = indexed
    return Chunk(
        id=_chunk_id(resource_type, source, anchor, cs),
        resource_type=resource_type,
        source=source,
        anchor=anchor,
        book=book,
        chapter=chapter,
        verse_start=verse_start,
        verse_end=verse_end,
        key=key,
        text=trimmed,
        text_checksum=cs,
        truncated=truncated,
        header=header,
        parent=parent,
        indexed=indexed,
        embed=embed and indexed,
        question=question,
        lords_day=lords_day,
        conf_chapter=conf_chapter,
        conf_section=conf_section,
        article=article,
    )


# --- Scripture: deterministic pericope windows -------------------------------
def _window_chapter(verses: list[tuple[int, int, str]]) -> Iterator[tuple[int, int, str]]:
    """Group one chapter's verses into pericope windows.

    ``verses`` is ``[(verse_start, verse_end, text), ...]`` in canonical order.
    Yields ``(window_start, window_end, joined_text)``. Windows never cross the
    chapter boundary (callers pass one chapter at a time). The last window absorbs
    a short remainder so we don't emit orphan 1-2 verse chunks.
    """
    n = len(verses)
    i = 0
    while i < n:
        end = min(i + PERICOPE_TARGET, n)
        remaining = n - end
        if 0 < remaining < PERICOPE_MIN_TAIL:
            end = n  # fold the short tail into this window
        group = verses[i:end]
        win_start = group[0][0]
        win_end = group[-1][1]
        text = " ".join(t for (_s, _e, t) in group if t)
        yield win_start, win_end, text
        i = end


def _heading_windows(
    verses: list[tuple[int, int, str]], boundaries: set[int]
) -> Iterator[tuple[int, int, str]]:
    """Group one chapter's verses into BSB-heading-delimited windows.

    ``boundaries`` are verse numbers that OPEN a section (a heading precedes
    them). The window before the first boundary (when a chapter doesn't start
    with a heading) is kept as its own window. Oversized sections are sub-split
    with the deterministic PERICOPE_TARGET window (tail-merged) so a parent
    stays a readable LLM-context unit.
    """
    segments: list[list[tuple[int, int, str]]] = []
    cur: list[tuple[int, int, str]] = []
    for row in verses:
        if row[0] in boundaries and cur:
            segments.append(cur)
            cur = []
        cur.append(row)
    if cur:
        segments.append(cur)
    for seg in segments:
        if len(seg) > PARENT_MAX:
            yield from _window_chapter(seg)
        else:
            text = " ".join(t for (_s, _e, t) in seg if t)
            yield seg[0][0], seg[-1][1], text


def scripture_chunks(db: Path) -> tuple[list[Chunk], list[str]]:
    """Dual-granularity Scripture chunks for all four translations.

    Parents: BSB-heading pericopes (fallback ``PERICOPE_TARGET`` windows),
    context-only. Children: single verse rows, BM25-indexed, linked to their
    parent; only BSB children carry vectors. Returns ``(chunks, skipped)`` —
    ``skipped`` lists anchors with no body text (omitted critical-text verses),
    flagged, not silently dropped.
    """
    conn = sqlite3.connect(db)
    chunks: list[Chunk] = []
    skipped: list[str] = []
    try:
        book_names = dict(conn.execute("SELECT id, name FROM book"))
        # Pericope boundaries come from the BSB heading apparatus and are shared
        # by every translation, so parents stay range-aligned across the four.
        boundaries: dict[tuple[str, int], set[int]] = {}
        for b, ch, vs in conn.execute(
                "SELECT book, chapter, verse_start FROM heading "
                "WHERE translation='bsb'"):
            boundaries.setdefault((b, ch), set()).add(vs)

        translations = [r[0] for r in conn.execute(
            "SELECT id FROM translation ORDER BY id")]
        for tid in translations:
            for book_id in books.ORDER:
                rows = conn.execute(
                    "SELECT chapter, verse_start, verse_end, text, omitted "
                    "FROM verse WHERE translation=? AND book=? "
                    "ORDER BY chapter, verse_start",
                    (tid, book_id),
                ).fetchall()
                if not rows:
                    continue
                bname = book_names.get(book_id, book_id)
                by_chapter: dict[int, list[tuple[int, int, str]]] = {}
                for ch, vs, ve, text, omitted in rows:
                    body = "" if omitted else (text or "")
                    by_chapter.setdefault(ch, []).append((vs, ve, body))
                for ch in sorted(by_chapter):
                    ch_verses = by_chapter[ch]
                    bset = boundaries.get((book_id, ch))
                    windows = (
                        _heading_windows(ch_verses, bset) if bset
                        else _window_chapter(ch_verses)
                    )
                    for win_start, win_end, text in windows:
                        ref = (f"{ch}:{win_start}"
                               + (f"-{win_end}" if win_end != win_start else ""))
                        anchor = f"{book_id} {ref}"
                        if not text.strip():
                            skipped.append(f"{tid}:{anchor} (empty window)")
                            continue
                        # The "pericope" anchor prefix keeps a single-verse
                        # parent distinct from its own child (same source,
                        # range, and text would otherwise collide to one id).
                        parent = _mk_chunk(
                            "scripture", tid, f"{tid}:pericope {anchor}", text,
                            book=book_id, chapter=ch,
                            verse_start=win_start, verse_end=win_end,
                            header=f"{bname} {ref} — ",
                            indexed=False,
                        )
                        chunks.append(parent)
                        # Children: the verse rows inside this window.
                        for vs, ve, body in ch_verses:
                            if not (win_start <= vs <= win_end):
                                continue
                            if not body.strip():
                                continue  # omitted rows: parent covers the ref
                            vref = f"{ch}:{vs}" + (f"-{ve}" if ve != vs else "")
                            chunks.append(_mk_chunk(
                                "scripture", tid, f"{tid}:{book_id} {vref}", body,
                                book=book_id, chapter=ch,
                                verse_start=vs, verse_end=ve,
                                header=f"{bname} {vref} — ",
                                parent=parent.id,
                                indexed=True,
                                embed=(tid == "bsb"),
                            ))
    finally:
        conn.close()
    return chunks, skipped


# --- Commentary: paragraph chunks, sentence-split when encoder-oversized -----
# Sentence boundary: terminal punctuation, optional close-quote, whitespace,
# then an upper-case/numeral/open-quote start. Deterministic, no NLP.
_SENTENCE_RE = re.compile(r"(?<=[.!?])[\"'’”\)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")


def split_sentences_to_limit(text: str, limit: int) -> list[str]:
    """Pack sentences into segments of at most ``limit`` characters.

    A single sentence longer than ``limit`` is hard-split at the last word
    boundary at/below the limit (never mid-word). Deterministic.
    """
    if len(text) <= limit:
        return [text]
    sentences: list[str] = []
    for s in _SENTENCE_RE.split(text):
        while len(s) > limit:
            cut = s.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            sentences.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            sentences.append(s)
    segments: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        extra = len(s) + (1 if cur else 0)
        if cur and cur_len + extra > limit:
            segments.append(" ".join(cur))
            cur, cur_len = [], 0
            extra = len(s)
        cur.append(s)
        cur_len += extra
    if cur:
        segments.append(" ".join(cur))
    return segments


def commentary_chunks(db: Path) -> tuple[list[Chunk], list[str]]:
    conn = sqlite3.connect(db)
    chunks: list[Chunk] = []
    skipped: list[str] = []
    try:
        # Header uses the curated AUTHOR field ("John Calvin"), not the long
        # work title ("John Calvin's Commentaries (NT + Psalms + Genesis)").
        commentator_names = dict(conn.execute("SELECT id, author FROM commentator"))
        book_names = dict(conn.execute("SELECT id, name FROM book"))
        rows = conn.execute(
            "SELECT commentator, key, ord, book, chapter, verse_start, verse_end, "
            "passage, text FROM comment ORDER BY commentator, ord"
        ).fetchall()
        for (commentator, key, ordn, book, chapter, vs, ve, passage, text) in rows:
            anchor = passage or key or f"{book} {chapter}"
            if not (text and text.strip()):
                skipped.append(f"{commentator}:{anchor} (empty paragraph)")
                continue
            cname = commentator_names.get(commentator, commentator)
            bname = book_names.get(book, book)
            if vs:
                ref = f"{chapter}:{vs}" + (f"-{ve}" if ve and ve != vs else "")
            else:
                ref = str(chapter)  # chapter-introduction note
            header = f"{cname} on {bname} {ref} — "
            # ord makes the per-paragraph anchor unique within a passage.
            uniq = f"{commentator}:{key}#{ordn}"
            segments = split_sentences_to_limit(
                text.strip(), COMMENT_SPLIT_CHARS - len(header))
            for si, seg in enumerate(segments, start=1):
                seg_anchor = uniq if len(segments) == 1 else f"{uniq}#s{si}"
                chunks.append(_mk_chunk(
                    "commentary", commentator, seg_anchor, seg,
                    book=book, chapter=chapter,
                    verse_start=vs or None, verse_end=ve or None,
                    key=key, header=header,
                ))
    finally:
        conn.close()
    return chunks, skipped


# --- Confessions: section / Q&A chunks ---------------------------------------
def confession_chunks(db: Path) -> tuple[list[Chunk], list[str]]:
    conn = sqlite3.connect(db)
    chunks: list[Chunk] = []
    skipped: list[str] = []
    try:
        # Shortcode per document for a readable anchor (e.g. "WCF 11.1"), and
        # the full name for the structural header.
        docs = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT id, shortcode, name FROM document")}
        rows = conn.execute(
            "SELECT document, key, ord, title, text, chapter, section, "
            "question, article, lords_day FROM section ORDER BY document, ord"
        ).fetchall()
        for (document, key, _ordn, title, text, conf_chapter, conf_section,
             question, article, lords_day) in rows:
            sc, dname = docs.get(document, (document.upper(), document))
            anchor = f"{sc} {key}"
            if not (text and text.strip()):
                skipped.append(f"{anchor} (empty section)")
                continue
            # Header from curated fields: "Westminster Shorter Catechism
            # Q. 33 — ", "Belgic Confession Article 22 — ", "Westminster
            # Confession of Faith 11.1 — ".
            if question:
                ref = f"Q. {question}"
            elif article:
                ref = f"Article {article}"
            else:
                ref = key
            header = f"{dname} {ref} — "
            # Prepend the section title so the heading is searchable alongside body.
            embed_text = f"{title}\n{text}" if title else text
            chunks.append(_mk_chunk(
                "confession", document, anchor, embed_text, key=key,
                header=header,
                question=question, lords_day=lords_day,
                conf_chapter=conf_chapter, conf_section=conf_section,
                article=article,
            ))
    finally:
        conn.close()
    return chunks, skipped


# --- Lexicons: entry-level chunks --------------------------------------------
def lexicon_chunks(db: Path) -> tuple[list[Chunk], list[str]]:
    """Entry chunks for every lexicon definition.

    Entries with no English-embeddable text (the BDB lemma-only stubs whose
    Strong's link is empty and which carry only a Hebrew lemma) are skipped and
    flagged — embedding a bare Hebrew lemma into an English-only model is noise,
    not signal. They are never dropped from the lexicon DB itself.
    """
    conn = sqlite3.connect(db)
    chunks: list[Chunk] = []
    skipped: list[str] = []
    try:
        lex_names = dict(conn.execute("SELECT id, name FROM lexicon"))
        rows = conn.execute(
            "SELECT lexicon, strongs, raw_key, lemma, translit, definition, "
            "derivation, kjv_def FROM entry "
            "ORDER BY lexicon, strongs, raw_key"
        ).fetchall()
        for (lexicon, strongs, raw_key, lemma, translit, definition,
             derivation, kjv_def) in rows:
            key = strongs or raw_key or lemma or ""
            anchor = f"{lexicon}:{key}"
            # Build the embeddable text from the English-bearing fields. Lemma /
            # transliteration are included as light context but are not sufficient
            # on their own.
            english = " ".join(p for p in (definition, derivation, kjv_def)
                               if p and p.strip())
            if not english.strip():
                skipped.append(f"{anchor} (no English definition; lemma-only stub)")
                continue
            head = " ".join(p for p in (lemma, translit) if p and p.strip())
            embed_text = f"{head} — {english}" if head else english
            # Header from curated fields: "Strong's Greek G26 — ".
            lname = lex_names.get(lexicon, lexicon)
            header = f"{lname} {key} — " if key else f"{lname} — "
            chunks.append(_mk_chunk(
                "lexicon", lexicon, anchor, embed_text, key=key or None,
                header=header,
            ))
    finally:
        conn.close()
    return chunks, skipped


@dataclass
class ExtractedChunks:
    """All chunks plus per-resource skip flags."""

    chunks: list[Chunk] = field(default_factory=list)
    skipped: dict[str, list[str]] = field(default_factory=dict)

    def by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.resource_type] = out.get(c.resource_type, 0) + 1
        return out

    def by_source(self) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        for c in self.chunks:
            k = (c.resource_type, c.source)
            out[k] = out.get(k, 0) + 1
        return out


def extract_all(output_dir: Path) -> ExtractedChunks:
    """Extract every chunk from the built DBs, in a fixed resource order.

    Final chunk order is deterministic: resource type order
    (scripture, commentary, confession, lexicon), each already emitted in a fixed
    intra-resource order by its extractor.
    """
    ec = ExtractedChunks()
    s_chunks, s_skip = scripture_chunks(output_dir / "bibles.sqlite")
    c_chunks, c_skip = commentary_chunks(output_dir / "commentaries.sqlite")
    f_chunks, f_skip = confession_chunks(output_dir / "confessions.sqlite")
    l_chunks, l_skip = lexicon_chunks(output_dir / "lexicons.sqlite")
    ec.chunks = s_chunks + c_chunks + f_chunks + l_chunks
    ec.skipped = {
        "scripture": s_skip,
        "commentary": c_skip,
        "confession": f_skip,
        "lexicon": l_skip,
    }
    return ec


# --- BM25 tokenizer (deterministic; documented) ------------------------------
# Lowercase, Unicode-NFKC-normalized, split on non-alphanumeric. No stemming, no
# stopword removal, no language-specific rules — so the same text always yields
# the same token stream regardless of platform or library version. Apostrophes
# inside a word are kept (e.g. "lord's") then stripped if leading/trailing.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Deterministic BM25 tokenizer (see module docstring)."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_RE.findall(norm)
