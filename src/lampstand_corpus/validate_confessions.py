"""Validation pass over normalized confession chunks.

Per document, flags (never silently fixes):
  * per-document section / Q&A count vs. the known total (CLAUDE.md Part B scope)
  * gaps in the question/section sequence
  * empty section text
  * proof-text VerseRefs whose book/chapter/verse fall outside the canonical spine
  * every CCEL structural ambiguity surfaced by the parser (carried verbatim)

Residuals a human must adjudicate are listed verbatim. The pipeline never marks
output ship-ready — the architect's spot-check gates ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import books
from .confessions import (
    DORT_HEAD_COUNTS,
    DORT_HEADS,
    EXPECTED_COUNTS,
    HEIDELBERG_LORDS_DAYS,
    ParsedConfession,
)


@dataclass
class ConfessionReport:
    document: str
    n_chunks: int = 0
    n_chapters: int = 0          # WCF only
    n_questions: int = 0         # catechisms only
    n_lords_days: int = 0        # Heidelberg only
    n_heads: int = 0             # Dort only
    n_articles: int = 0          # Dort only (positive articles)
    n_rejections: int = 0        # Dort only (rejection-of-errors paragraphs)
    n_proof_texts: int = 0
    expected: int = 0
    count_ok: bool = False
    sequence_gaps: list[str] = field(default_factory=list)
    empty_sections: list[str] = field(default_factory=list)
    bad_proof_refs: list[str] = field(default_factory=list)
    parser_flags: list[str] = field(default_factory=list)

    @property
    def error_total(self) -> int:
        return (0 if self.count_ok else 1) + len(self.empty_sections) + len(
            self.bad_proof_refs)

    @property
    def flag_total(self) -> int:
        return len(self.sequence_gaps) + len(self.parser_flags)


def _valid_ref(book: str, chapter: int, verse: int) -> bool:
    if book not in books.CANON:
        return False
    counts = books.VERSE_COUNTS.get(book)
    if not counts or not (1 <= chapter <= len(counts)):
        return False
    return 1 <= verse <= counts[chapter - 1]


def validate_confession(pc: ParsedConfession) -> ConfessionReport:
    rep = ConfessionReport(document=pc.id)
    rep.n_chunks = len(pc.chunks)
    rep.parser_flags = list(pc.flags)
    rep.expected = EXPECTED_COUNTS.get(pc.id, 0)

    is_wcf = pc.id == "wcf"
    is_dort = pc.id == "dort"
    nums: list[int] = []
    for ch in pc.chunks:
        m = ch.meta
        if not ch.text.strip():
            rep.empty_sections.append(f"{pc.id}:{ch.key} (empty text)")
        proofs = m.get("proof_texts") or []
        rep.n_proof_texts += len(proofs)
        for pt in proofs:
            if not _valid_ref(pt.get("book", ""), pt.get("chapter", 0),
                              pt.get("verse_start", 0)):
                rep.bad_proof_refs.append(
                    f"{pc.id}:{ch.key} -> "
                    f"{pt.get('book')} {pt.get('chapter')}:{pt.get('verse_start')}"
                )
        if is_wcf:
            nums.append(m["chapter"])
        elif is_dort:
            kind = m.get("kind")
            if kind == "article":
                rep.n_articles += 1
            elif kind == "rejection":
                rep.n_rejections += 1
        else:
            nums.append(m["question"])

    if is_dort:
        heads = {c.meta.get("head") for c in pc.chunks
                 if c.meta.get("head") not in (None, "conclusion")}
        rep.n_heads = len(heads)
        rep.expected = DORT_HEADS
        # "count_ok" for Dort = the 4 head-sections each match their known
        # article/rejection totals (per-head deviations are surfaced as flags).
        rep.count_ok = (
            len(heads) == len(DORT_HEAD_COUNTS)
            and rep.n_articles == sum(a for a, _ in DORT_HEAD_COUNTS.values())
            and rep.n_rejections == sum(r for _, r in DORT_HEAD_COUNTS.values())
        )
    elif is_wcf:
        chapters = sorted(set(nums))
        rep.n_chapters = len(chapters)
        rep.count_ok = rep.n_chapters == rep.expected
        rep.sequence_gaps = [
            f"missing original chapter {n}"
            for n in range(1, rep.expected + 1) if n not in set(chapters)
        ]
    else:
        questions = sorted(set(nums))
        rep.n_questions = len(questions)
        rep.count_ok = rep.n_questions == rep.expected
        if questions:
            rep.sequence_gaps = [
                f"missing question {n}"
                for n in range(1, questions[-1] + 1) if n not in set(questions)
            ]
        if pc.id == "heidelberg":
            rep.n_lords_days = len({
                ch.meta.get("lords_day") for ch in pc.chunks
                if ch.meta.get("lords_day")
            })

    return rep


def validate_all_confessions(
    parsed: dict[str, ParsedConfession]
) -> dict[str, ConfessionReport]:
    return {did: validate_confession(parsed[did]) for did in sorted(parsed)}


# Documents in scope that we could not cleanly source from canonical references
# and therefore SKIPPED (surfaced for human sourcing; never substituted). Each
# entry: (label, why).
SKIPPED_DOCUMENTS: list[tuple[str, str]] = [
    ("1689 London Baptist Confession (32 chapters)",
     "CCEL/Schaff (Creeds of Christendom III, 'Baptist Confession of 1688 / "
     "Philadelphia Confession') prints only the editorial DIFFERENCES from the "
     "Westminster/Savoy text, not the full 32 chapters. No standalone clean "
     "full-text 1689 ThML exists on CCEL. Reconstructing it would mean splicing "
     "Westminster text with Schaff's noted deltas — guesswork, which CLAUDE.md "
     "forbids. ARCHITECT: approve a canonical full-text source before inclusion."),
    ("Belgic Confession (37 articles)",
     "Present in Schaff Creeds III only as a 2-column French/Latin-English TABLE "
     "(54 tables; English in the right cell) whose 37 articles are fragmented "
     "mid-sentence across rows/tables with OCR artifacts ('A rt. II.', "
     "'G uy de B rès'). Clean per-article reconstruction needs heuristic stitching "
     "+ OCR repair — a guess, not a parse. Re-source a clean canonical English "
     "Belgic before inclusion."),
    ("Apostles' / Nicene / Athanasian creeds",
     "Tentative per spec §6.2. CCEL lists them under the Creeds subject but exposes "
     "only rendered HTML (no standalone ThML download), and in Schaff they sit among "
     "many historical variant creeds with Greek/Latin parallels. Not cleanly "
     "available -> skipped per the 'include only if clean' rule."),
]


def render_confession_report(
    reports: dict[str, ConfessionReport]
) -> str:
    lines: list[str] = []
    lines.append("LampStand corpus — P2 confessions & catechisms validation report")
    lines.append("Granularity: section / Q&A. Counts checked against known totals.")
    lines.append("Every CCEL structural ambiguity is a FLAG for human review, not a")
    lines.append("silent fix. The pipeline never marks a corpus ship-ready.")
    lines.append("=" * 72)

    lines.append("")
    lines.append("## KNOWN TOTALS (target)")
    lines.append("  WCF=33 chapters  WLC=196 Q  WSC=107 Q  "
                 f"Heidelberg=129 Q / {HEIDELBERG_LORDS_DAYS} Lord's Days")
    _dort_a = sum(a for a, _ in DORT_HEAD_COUNTS.values())
    _dort_r = sum(r for _, r in DORT_HEAD_COUNTS.values())
    lines.append(f"  Dort={DORT_HEADS} heads of doctrine "
                 f"({_dort_a} articles + {_dort_r} rejection paragraphs; "
                 "3rd & 4th heads combined) + Conclusion")

    for did in sorted(reports):
        r = reports[did]
        lines.append("")
        lines.append(f"## {did.upper()}")
        if did == "wcf":
            lines.append(f"  chapters={r.n_chapters}/{r.expected}  "
                         f"sections={r.n_chunks}  proof-texts={r.n_proof_texts}")
        elif did == "dort":
            lines.append(f"  head-sections={r.n_heads}/{len(DORT_HEAD_COUNTS)} "
                         f"(= {r.expected} canonical heads, 3rd & 4th combined)  "
                         f"articles={r.n_articles}  rejections={r.n_rejections}  "
                         f"chunks={r.n_chunks} (incl. Conclusion)")
        elif did == "heidelberg":
            lines.append(f"  questions={r.n_questions}/{r.expected}  "
                         f"Lord's Days={r.n_lords_days}/{HEIDELBERG_LORDS_DAYS}  "
                         f"proof-texts={r.n_proof_texts}")
        else:
            lines.append(f"  questions={r.n_questions}/{r.expected}  "
                         f"proof-texts={r.n_proof_texts}")
        lines.append(f"  count_ok={r.count_ok}  errors={r.error_total}  "
                     f"flags={r.flag_total}")

        def block(title: str, items: list[str], cap: int = 50) -> None:
            if not items:
                return
            lines.append(f"  {title} ({len(items)}):")
            for it in items[:cap]:
                lines.append(f"    - {it}")
            if len(items) > cap:
                lines.append(f"    ... and {len(items) - cap} more")

        block("SEQUENCE GAPS", r.sequence_gaps)
        block("EMPTY SECTIONS", r.empty_sections)
        block("PROOF-TEXT REFS OUTSIDE CANON (review)", r.bad_proof_refs)
        block("PARSER / CCEL AMBIGUITY FLAGS (review)", r.parser_flags)

    lines.append("")
    lines.append("## DOCUMENTS IN SCOPE BUT SKIPPED (flagged for human sourcing)")
    for label, why in SKIPPED_DOCUMENTS:
        lines.append(f"  - {label}")
        lines.append(f"      {why}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("END. Human spot-check still required before ship "
                 "(architect's 23-point: 3 confession sections vs authoritative).")
    return "\n".join(lines) + "\n"
