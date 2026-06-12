"""Validation pass over the embeddings + BM25 index (CLAUDE.md pipeline rules).

The report records:
  * total chunks embedded per resource type + per source; the embedding dim;
  * any chunk that failed or was *skipped* during extraction (flagged, never
    silently dropped — e.g. lemma-only BDB stubs, empty Scripture windows);
  * any chunk hard-truncated at the character cap (flagged);
  * BM25 vocabulary size + total postings;
  * a small retrieval smoke test (dense top-k for a few canonical queries) so a
    human can eyeball that "justification by faith" lands in Romans/Galatians and
    "the LORD is my shepherd" lands in Psalm 23;
  * the determinism outcome (bit-for-bit, or a flagged within-tolerance deviation).

The pipeline never marks output ship-ready — the architect's 23-point spot-check
gates ship (CLAUDE.md §Human spot-check).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .embeddings import EMBED_DIM, ExtractedChunks

EXAMPLE_LIMIT = 20


@dataclass
class SmokeHit:
    rank: int
    score: float
    resource_type: str
    source: str
    anchor: str
    preview: str


@dataclass
class SmokeQuery:
    query: str
    expect: str            # human description of the expected neighborhood
    hits: list[SmokeHit] = field(default_factory=list)
    passed: bool = False   # did any top-k hit match the expectation predicate?


@dataclass
class DeterminismResult:
    method: str            # "bit-for-bit" | "cosine-tolerance"
    identical: bool
    max_abs_diff: float = 0.0
    min_cosine: float = 1.0
    note: str = ""


@dataclass
class IncrementalStats:
    """Incremental re-encode accounting (corpus-update path)."""

    n_total: int = 0
    n_reused: int = 0       # vectors copied verbatim from the prior DB
    n_encoded: int = 0      # changed/new chunks re-encoded this run
    n_dropped: int = 0      # prior chunk ids absent from the new set (vectors gone)
    prior_model_revision: str = ""
    full_reencode: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class EmbeddingReport:
    embedding_dim: int = EMBED_DIM
    n_chunks: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, list[str]] = field(default_factory=dict)
    n_truncated: int = 0
    truncated_examples: list[str] = field(default_factory=list)

    vocab_size: int = 0
    n_postings: int = 0
    avgdl: float = 0.0

    model_name: str = ""
    model_revision: str = ""
    model_combined_sha256: str = ""

    smoke: list[SmokeQuery] = field(default_factory=list)
    determinism: DeterminismResult | None = None
    incremental: IncrementalStats | None = None
    wall_seconds: float = 0.0

    @property
    def n_skipped(self) -> int:
        return sum(len(v) for v in self.skipped.values())

    @property
    def error_total(self) -> int:
        # Skips are flagged (kept-for-review), not hard errors; the only thing that
        # rises to "error" here is a determinism deviation or a failed smoke query.
        det_fail = 1 if (self.determinism and not self.determinism.identical) else 0
        smoke_fail = sum(1 for s in self.smoke if not s.passed)
        return det_fail + smoke_fail

    @property
    def flag_total(self) -> int:
        return self.n_skipped + (1 if self.n_truncated else 0)


def summarize_extract(ec: ExtractedChunks) -> EmbeddingReport:
    rep = EmbeddingReport()
    rep.n_chunks = len(ec.chunks)
    rep.by_type = ec.by_type()
    by_source: dict[str, int] = {}
    for (rt, src), n in sorted(ec.by_source().items()):
        by_source[f"{rt}/{src}"] = n
    rep.by_source = by_source
    rep.skipped = ec.skipped
    rep.n_truncated = sum(1 for c in ec.chunks if c.truncated)
    rep.truncated_examples = [c.anchor for c in ec.chunks if c.truncated][:EXAMPLE_LIMIT]
    return rep


def render_embedding_report(rep: EmbeddingReport) -> str:
    lines: list[str] = []
    lines.append("LampStand corpus — embeddings + BM25 validation report (M1-P6)")
    lines.append("=" * 64)
    lines.append("")
    lines.append("CANDIDATE ONLY. Not ship-ready: the architect's 23-point "
                 "spot-check gates ship (CLAUDE.md §Human spot-check).")
    lines.append("")

    lines.append("EMBEDDING MODEL (provenance)")
    lines.append("-" * 64)
    lines.append(f"  model:        {rep.model_name}")
    lines.append(f"  revision:     {rep.model_revision}")
    lines.append(f"  combined sha: {rep.model_combined_sha256}")
    lines.append(f"  dim:          {rep.embedding_dim}")
    lines.append("  vectors:      L2-normalized float32 (little-endian)")
    lines.append("  weights are gitignored (models/); never committed.")
    lines.append("")

    lines.append("ERROR / FLAG SUMMARY")
    lines.append("-" * 64)
    lines.append(f"  total errors: {rep.error_total}")
    lines.append(f"  total flags:  {rep.flag_total}")
    lines.append("")

    lines.append("CHUNKS EMBEDDED")
    lines.append("-" * 64)
    lines.append(f"  total chunks: {rep.n_chunks}")
    lines.append("  by resource type:")
    for rt in sorted(rep.by_type):
        lines.append(f"    {rt:<12} {rep.by_type[rt]}")
    lines.append("  by source:")
    for src in sorted(rep.by_source):
        lines.append(f"    {src:<26} {rep.by_source[src]}")
    lines.append("")

    lines.append("SKIPPED CHUNKS (flagged, NOT dropped from their source DB)")
    lines.append("-" * 64)
    if rep.n_skipped == 0:
        lines.append("  (none)")
    for rt in sorted(rep.skipped):
        items = rep.skipped[rt]
        if not items:
            continue
        lines.append(f"  {rt}: {len(items)}")
        for s in items[:EXAMPLE_LIMIT]:
            lines.append(f"    {s}")
        if len(items) > EXAMPLE_LIMIT:
            lines.append(f"    ... (+{len(items) - EXAMPLE_LIMIT} more)")
    lines.append("")
    if rep.n_truncated:
        lines.append(f"  hard-truncated at char cap: {rep.n_truncated}")
        for a in rep.truncated_examples:
            lines.append(f"    {a}")
        if rep.n_truncated > len(rep.truncated_examples):
            lines.append(f"    ... (+{rep.n_truncated - len(rep.truncated_examples)} more)")
        lines.append("")

    lines.append("BM25 KEYWORD INDEX")
    lines.append("-" * 64)
    lines.append("  tokenizer:     NFKC + casefold + [a-z0-9']+ ; no stemming")
    lines.append(f"  vocabulary:    {rep.vocab_size} terms")
    lines.append(f"  postings:      {rep.n_postings}")
    lines.append(f"  avg doc len:   {rep.avgdl:.2f} tokens")
    lines.append("")

    lines.append("RETRIEVAL SMOKE TEST (dense top-k; eyeball only)")
    lines.append("-" * 64)
    for sq in rep.smoke:
        status = "PASS" if sq.passed else "REVIEW"
        lines.append(f"  [{status}] \"{sq.query}\"")
        lines.append(f"          expect: {sq.expect}")
        for h in sq.hits:
            lines.append(
                f"    {h.rank}. ({h.score:.3f}) {h.resource_type}/{h.source} "
                f"{h.anchor} — {h.preview}"
            )
        lines.append("")

    lines.append("INCREMENTAL RE-ENCODE")
    lines.append("-" * 64)
    if rep.incremental is None:
        lines.append("  (full encode — no prior-vector reuse recorded this run)")
    else:
        inc = rep.incremental
        if inc.full_reencode:
            lines.append("  mode: FULL re-encode (reuse unavailable)")
        else:
            lines.append("  mode: incremental (reused unchanged vectors)")
        lines.append(f"  total chunks:  {inc.n_total}")
        lines.append(f"  reused:        {inc.n_reused}  "
                     f"(vectors copied verbatim from prior DB)")
        lines.append(f"  re-encoded:    {inc.n_encoded}  (changed / new chunks)")
        lines.append(f"  dropped:       {inc.n_dropped}  "
                     f"(prior chunk ids no longer present)")
        if inc.prior_model_revision:
            lines.append(f"  prior model revision: {inc.prior_model_revision}")
        lines.append("  reuse key: content-addressed chunk id "
                     "(resource_type+source+anchor+text_checksum); an id match "
                     "guarantees byte-identical text under the pinned model.")
        for note in inc.notes:
            lines.append(f"  note: {note}")
    lines.append("")

    lines.append("DETERMINISM")
    lines.append("-" * 64)
    if rep.determinism is None:
        lines.append("  (not run)")
    else:
        d = rep.determinism
        lines.append(f"  method:   {d.method}")
        lines.append(f"  identical: {d.identical}")
        if d.method != "bit-for-bit" or not d.identical:
            lines.append(f"  max abs diff: {d.max_abs_diff:.3e}")
            lines.append(f"  min cosine:   {d.min_cosine:.8f}")
        if d.note:
            lines.append(f"  note: {d.note}")
    lines.append(f"  encode wall-time: {rep.wall_seconds:.1f}s (CPU, deterministic)")
    lines.append("")

    lines.append("FLAGGED FOR HUMAN REVIEW")
    lines.append("-" * 64)
    n = 1
    lex_skips = rep.skipped.get("lexicon", [])
    if lex_skips:
        # Split the two distinct reasons so the flag is honest: BDB lemma-only
        # stubs vs. Strong's "Not Used" placeholders (the CC0 Greek edition ships
        # explicit placeholders for skipped Strong's numbers; they carry no English
        # to embed, correctly skipped).
        bdb = [s for s in lex_skips if s.startswith("bdb:")]
        strongs_nu = [s for s in lex_skips if s.startswith(("strongs-greek:",
                                                            "strongs-hebrew:"))]
        if bdb:
            lines.append(
                f"  {n}. {len(bdb)} BDB lexicon entries skipped from embedding — "
                f"lemma-only stubs with no English definition (a bare Hebrew lemma "
                f"is noise to an English encoder). They remain in lexicons.sqlite; "
                f"confirm none should instead be merged into a linked Strong's entry."
            )
            n += 1
        if strongs_nu:
            lines.append(
                f"  {n}. {len(strongs_nu)} Strong's entries skipped from embedding — "
                f"'Not Used' placeholder numbers (Strong assigned but never "
                f"populated; the new CC0 Greek edition ships them explicitly where "
                f"the prior CC-BY-SA edition omitted them). No English text to embed; "
                f"kept in lexicons.sqlite so the coverage span is explicit."
            )
            n += 1
    scr = rep.skipped.get("scripture", [])
    if scr:
        lines.append(
            f"  {n}. {len(scr)} Scripture pericope windows were empty (made entirely "
            f"of omitted critical-text verses) and skipped; verify these are the "
            f"expected omitted-variant gaps, not a chunking error."
        )
        n += 1
    if rep.n_truncated:
        lines.append(
            f"  {n}. {rep.n_truncated} chunks exceeded the character cap and were "
            f"truncated before encoding (the encoder also truncates at 512 tokens). "
            f"Confirm no long commentary block lost material it needed for retrieval."
        )
        n += 1
    review_smoke = [s for s in rep.smoke if not s.passed]
    if review_smoke:
        qs = ", ".join(f'"{s.query}"' for s in review_smoke)
        lines.append(
            f"  {n}. Smoke queries that did not surface their expected passage in "
            f"top-k: {qs}. Eyeball the ranked hits above before trusting retrieval."
        )
        n += 1
    if rep.determinism and not rep.determinism.identical:
        lines.append(
            f"  {n}. Embeddings were NOT bit-for-bit reproducible across two CPU "
            f"runs (min cosine {rep.determinism.min_cosine:.8f}, max abs diff "
            f"{rep.determinism.max_abs_diff:.3e}). Recorded as within-tolerance and "
            f"flagged for the architect per CLAUDE.md rule 6 — do not accept "
            f"silently."
        )
        n += 1
    if n == 1:
        lines.append("  (none)")
    lines.append("")
    return "\n".join(lines) + "\n"
