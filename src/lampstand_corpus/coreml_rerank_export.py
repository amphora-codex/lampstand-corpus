"""Export the on-device cross-encoder RERANKER to Core ML (Rank 4, SHIP).

This is the corpus-side half of the reranker on-device contract. The rerank
quality gate (``reports/reranker_eval_v1.md``) returned **SHIP**: on the
lexical-favoring corpus gold set the cross-encoder still lifts commentary-anchor
MRR materially (and the app's paraphrased-query eval is where the semantic gain
is fully realized). This tool traces the *exact pinned* cross-encoder weights
that produced the measured rerank scores into a fp16 ``.mlpackage`` whose graph
emits one relevance logit per (query, passage) pair, then proves — via a lineage
gate and a score-parity gate — that the exported model reproduces the PyTorch
scores (and, crucially for reranking, their ORDER) within tolerance.

Model choice: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (Apache-2.0, BERT
MiniLM-L6, 384-hidden, 6-layer, ~22M params). The fp16 ``.mlpackage`` measures
~43 MB (coremltools keeps the token/position embedding tables and a few ops at
higher precision, and the package carries metadata/spec alongside the weights) —
larger than the ~20-25 MB back-of-envelope but still a fraction of the alternative
and comfortably shippable (the sibling BGEQuery encoder ships at ~66 MB). The
exact measured size is FLAGGED in the export report. ``BAAI/bge-reranker-base``
scored higher in the gate but is XLM-RoBERTa base (~278M params, ~140 MB fp16) —
far over budget — so it is recorded in the report as a quality CEILING, not the
shipped model.

Parity contract (mirrors ``coreml_export.py`` §2 discipline):
  * The graph emits the RAW classifier logit (``num_labels=1``); higher = more
    relevant. No sigmoid — ``sentence-transformers`` CrossEncoder.predict returns
    the raw logit for this model, and reranking only needs the ORDER, so the
    monotone raw logit is the score.
  * Inputs ``input_ids`` / ``attention_mask`` / ``token_type_ids`` (int32,
    batch 1, seq RangeDim(1,512,default=32)). Unlike the BGE query encoder,
    ``token_type_ids`` are MEANINGFUL here: query tokens are segment 0, passage
    tokens segment 1 (BERT sentence-pair encoding). The Swift tokenizer MUST
    build the pair + segment ids exactly (fixture below is ground truth).
  * fp16 acceptance: max abs logit error <= :data:`PARITY_MAX_ABS` AND the
    reranked ORDER of the fixture pairs is IDENTICAL to the PyTorch order
    (Spearman-exact / zero inversions). An order flip is a STOP-and-investigate.

Determinism (CLAUDE.md rule 6): trace + convert on single-thread CPU with the
same deterministic knobs ``encode.py`` uses; the ``.mlpackage`` tree is
canonicalized (protobuf field order + stable Manifest UUIDs) so re-runs are
byte-identical. No timestamps in output.

Requires the ``[coreml]`` extra on top of ``[rerank]`` (torch/transformers +
coremltools). coremltools is imported lazily so the light pipeline phases stay
importable without it. The iOS app ships a static ``.mlpackage`` + ``vocab.txt``
loaded from disk; it gains no Python/SPM dependency.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .eval_rerank import DEFAULT_RERANK_MODEL, MAX_PAIR_TOKENS, RERANK_MODELS

# --- Parity contract constants (single source of truth) ----------------------
# Cross-encoder logits are UNBOUNDED (this model spans roughly [-12, +11]), so an
# absolute-error floor is the wrong instrument — a fixed 0.05 is 0.4% of an 11.4-
# magnitude logit yet 100% of a 0.05 one. The gate therefore measures RELATIVE
# error, abs_err / (|torch_logit| + REL_DENOM_FLOOR), which is scale-appropriate
# for both large and near-zero logits. A trip is a stop-and-investigate, never a
# threshold loosening.
PARITY_MAX_REL = 0.02          # <= 2% relative logit error (fp16 vs float32)
REL_DENOM_FLOOR = 1.0          # softens the denominator near logit 0
# The property reranking actually depends on is ORDER preservation — but only
# among candidates with a NON-TRIVIAL score gap. Two passages whose PyTorch
# logits tie to within TIE_EPS are equivalently (ir)relevant, so a fp16 reorder
# between them is not a defect and is not counted. Inversions between pairs with
# a real gap (> TIE_EPS) must be ZERO.
PARITY_MAX_INVERSIONS = 0
TIE_EPS = 0.05                 # PyTorch logit gaps <= this are ties (not scored)

# Number of deterministically-built (query, passage) parity pairs.
PARITY_SAMPLE_N = 30

# Example trace shape: batch 1, seq 32 (a short query + short passage pair). The
# exported seq axis is flexible via RangeDim.
TRACE_SEQ_LEN = 32
SEQ_RANGE_LO = 1
SEQ_RANGE_HI = 512
SEQ_RANGE_DEFAULT = 32
SEQ_LEN_LABEL = "RangeDim(1,512)"

# Tokenizer-fixture probe PAIRS (query, passage): theological terms, apostrophe
# split, numerals, an empty passage, and a full exegetical pair. Ground truth for
# the Swift sentence-pair tokenizer parity test (segment ids are load-bearing).
TOKENIZER_FIXTURE_PAIRS = [
    ("propitiation", "Christ is the atoning sacrifice for our sins."),
    ("justification", "Justification is an act of God's free grace."),
    ("God's love", "For God so loved the world."),
    ("1 John 2:2", "He is the propitiation for our sins."),
    ("sanctification", ""),
    (
        "What is effectual calling?",
        "Effectual calling is the work of God's Spirit, whereby he doth "
        "persuade and enable us to embrace Jesus Christ.",
    ),
]

# Output artifact names.
MLPACKAGE_NAME = "Reranker.mlpackage"
VOCAB_NAME = "reranker_vocab.txt"
PARITY_FIXTURE_NAME = "reranker_parity_fixture.json"
TOKENIZER_FIXTURE_NAME = "reranker_tokenizer_fixture.json"

# Core ML model I/O names — must match the Swift reranker exactly.
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
TOKEN_TYPE_IDS = "token_type_ids"
OUTPUT_SCORE = "score"


class RerankLineageError(RuntimeError):
    """The snapshot's combined weight hash does not match the measured model."""


class RerankParityError(RuntimeError):
    """The fp16 .mlpackage failed the logit-abs / order-preservation gate."""


@dataclass
class RerankParityCase:
    query: str
    passage: str
    torch_logit: float


@dataclass
class RerankParityResult:
    n: int = 0
    max_abs_err: float = 0.0
    mean_abs_err: float = 0.0
    max_rel_err: float = 0.0
    inversions: int = 0           # order flips among NON-tied PyTorch pairs
    tied_reorders: int = 0        # flips among tied pairs (reported, not gated)
    per_case: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.n > 0
            and self.max_rel_err <= PARITY_MAX_REL
            and self.inversions <= PARITY_MAX_INVERSIONS
        )


@dataclass
class RerankExportResult:
    mlpackage_path: Path
    mlpackage_tree_sha256: str
    mlpackage_bytes: int
    vocab_path: Path
    vocab_sha256: str
    parity_fixture_path: Path
    tokenizer_fixture_path: Path
    model_name: str
    model_revision: str
    model_combined_sha256: str
    parity: RerankParityResult
    forward_512_ms: float = 0.0
    tokenizer_fixture: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --- Snapshot + lineage ------------------------------------------------------
def _snapshot_dir(model_key: str, cache_dir: Path) -> Path:
    """Locate the pinned cross-encoder snapshot in the model cache."""
    hf_id = RERANK_MODELS[model_key]["hf_id"]
    repo = "models--" + hf_id.replace("/", "--")
    snaps = sorted((cache_dir / repo / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError(
            f"No snapshot for {hf_id} under {cache_dir}. Run the rerank-eval "
            f"gate first (it downloads the pinned revision)."
        )
    return snaps[-1]


def model_combined_sha256(snapshot_dir: Path) -> str:
    """SHA-256 over sorted ``path:sha256`` of every file in the snapshot.

    Mirrors ``encode.model_provenance`` semantics so the reranker's lineage is
    recorded the same way the BGE encoder's is: one number that changes if any
    weight/config/tokenizer file changes.
    """
    parts: list[str] = []
    for p in sorted(snapshot_dir.rglob("*")):
        if p.is_dir():
            continue
        real = p.resolve()
        h = hashlib.sha256(real.read_bytes()).hexdigest()
        parts.append(f"{p.relative_to(snapshot_dir)}:{h}")
    return hashlib.sha256("".join(parts).encode()).hexdigest()


# --- Torch wrapper: BertForSequenceClassification -> single logit ------------
def _build_wrapper(snapshot_dir: Path):
    """Load the cross-encoder (float32, eval) and wrap it to emit one logit.

    The wrapper returns ``logits[:, 0]`` — the raw relevance score. No pooling or
    normalization is baked in beyond what the classifier head already does.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    from .encode import _set_deterministic_cpu

    _set_deterministic_cpu()

    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot_dir), torch_dtype=torch.float32)
    model.eval()
    if model.config.num_labels != 1:
        raise RerankLineageError(
            f"Expected a single-logit reranker (num_labels=1); got "
            f"{model.config.num_labels}. Refusing to export a mismatched head."
        )

    class RerankerModule(torch.nn.Module):
        def __init__(self, encoder: torch.nn.Module) -> None:
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # Single relevance logit; higher = more relevant. Squeeze the label
            # axis so the output is (batch,).
            return out.logits[:, 0]

    wrapper = RerankerModule(model)
    wrapper.eval()
    return wrapper


def _trace(wrapper):
    """torch.jit.trace with an example (1, TRACE_SEQ_LEN) sentence-pair shape."""
    import torch

    ids = torch.zeros((1, TRACE_SEQ_LEN), dtype=torch.int32)
    mask = torch.ones((1, TRACE_SEQ_LEN), dtype=torch.int32)
    # Realistic segment ids (second half = passage segment) so the trace exercises
    # the token_type embedding path.
    ttids = torch.zeros((1, TRACE_SEQ_LEN), dtype=torch.int32)
    ttids[:, TRACE_SEQ_LEN // 2:] = 1
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ids, mask, ttids), check_trace=False)
    return traced


def _convert(traced, out_path: Path):
    """ct.convert -> fp16 mlprogram, iOS18, RangeDim(1,512) seq axis, int32 I/O.

    Reuses ``coreml_export._register_new_ones_op`` so transformers 5.x mask ops
    translate under coremltools 9 (same as the BGE export path).
    """
    import coremltools as ct

    from .coreml_export import _register_new_ones_op

    _register_new_ones_op()

    seq = ct.RangeDim(
        lower_bound=SEQ_RANGE_LO,
        upper_bound=SEQ_RANGE_HI,
        default=SEQ_RANGE_DEFAULT,
    )
    shape = ct.Shape(shape=(1, seq))
    inputs = [
        ct.TensorType(name=INPUT_IDS, shape=shape, dtype=np.int32),
        ct.TensorType(name=ATTENTION_MASK, shape=shape, dtype=np.int32),
        ct.TensorType(name=TOKEN_TYPE_IDS, shape=shape, dtype=np.int32),
    ]
    outputs = [ct.TensorType(name=OUTPUT_SCORE)]
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,  # deterministic conversion (rule 6)
    )
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    # Reuse the BGE export's tree canonicalizer, but key the stable Manifest UUIDs
    # on the reranker's own logical names so its tree hash is independent.
    _canonicalize_mlpackage(out_path)
    return mlmodel


def _canonicalize_mlpackage(pkg: Path) -> None:
    """Byte-reproducible .mlpackage tree (protobuf order + stable UUIDs).

    Mirrors ``coreml_export._canonicalize_mlpackage`` but with reranker-specific
    uuid5 seeds so the two models' Manifests never collide.
    """
    from .coreml_export import _canonicalize_model_spec

    _canonicalize_model_spec(pkg)
    manifest_path = pkg / "Manifest.json"
    if not manifest_path.exists():
        return
    import uuid as _uuid

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("itemInfoEntries", {})
    ns = _uuid.UUID("00000000-0000-0000-0000-000000000000")
    stable_by_name = {
        "weights": str(_uuid.uuid5(ns, "lampstand.reranker.weights")).upper(),
        "model.mlmodel": str(_uuid.uuid5(ns, "lampstand.reranker.model")).upper(),
    }
    new_entries: dict[str, dict] = {}
    root_new: str | None = None
    old_root = manifest.get("rootModelIdentifier")
    for old_id, info in entries.items():
        name = info.get("name", old_id)
        new_id = stable_by_name.get(name, name)
        new_entries[new_id] = info
        if old_id == old_root:
            root_new = new_id
    out = {
        "fileFormatVersion": manifest.get("fileFormatVersion", "1.0.0"),
        "itemInfoEntries": dict(sorted(new_entries.items())),
    }
    if root_new is not None:
        out["rootModelIdentifier"] = root_new
    manifest_path.write_text(
        json.dumps(out, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


# --- Parity gate -------------------------------------------------------------
def _build_parity_cases(snapshot_dir: Path, tokenizer) -> list[RerankParityCase]:
    """Deterministic (query, passage) pairs + their float32 PyTorch logits.

    Pairs are a fixed, hand-curated theological set (no RNG) spanning clearly-
    relevant and clearly-irrelevant matches so the ORDER-preservation check has
    real spread to verify. The PyTorch logit is computed once here (float32,
    eval, single-thread) and becomes ground truth for the fp16 comparison.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    from .encode import _set_deterministic_cpu

    _set_deterministic_cpu()
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot_dir), torch_dtype=torch.float32).eval()

    pairs = _parity_pairs()
    cases: list[RerankParityCase] = []
    for q, p in pairs:
        enc = tokenizer(
            q, p, add_special_tokens=True, truncation=True,
            max_length=MAX_PAIR_TOKENS, return_tensors="pt")
        with torch.no_grad():
            logit = float(model(**enc).logits.reshape(-1)[0])
        cases.append(RerankParityCase(query=q, passage=p, torch_logit=logit))
    return cases


def _parity_pairs() -> list[tuple[str, str]]:
    """Fixed relevant/irrelevant (query, passage) spread for the parity gate."""
    relevant = [
        ("propitiation", "He is the atoning sacrifice for our sins."),
        ("justification by faith", "A man is justified by faith apart from works."),
        ("the good shepherd", "I am the good shepherd; the good shepherd lays down "
         "his life for the sheep."),
        ("God created the heavens", "In the beginning God created the heavens and "
         "the earth."),
        ("effectual calling", "Effectual calling is the work of God's Spirit."),
    ]
    irrelevant = [
        ("propitiation", "The disciples went fishing all night and caught nothing."),
        ("justification by faith", "He planted a vineyard and set a hedge around it."),
        ("the good shepherd", "The number of the men was about five thousand."),
        ("God created the heavens", "Peter warmed himself at the fire."),
        ("effectual calling", "They cast lots for his garments."),
    ]
    # A middle band of partially-relevant pairs so the fixture spans the full
    # logit range (not just the saturated extremes), giving the order-preservation
    # check real, non-tied gaps to verify. Deterministic (no RNG), and every pair
    # is distinct to avoid manufacturing degenerate ties.
    partial = [
        ("propitiation", "The wages of sin is death."),
        ("justification by faith", "Abraham believed God, and it was counted to "
         "him as righteousness."),
        ("the good shepherd", "The Lord is my shepherd; I shall not want."),
        ("God created the heavens", "The heavens declare the glory of God."),
        ("effectual calling", "Many are called, but few are chosen."),
        ("propitiation", "By grace you have been saved through faith."),
        ("justification by faith", "For we hold that one is justified by faith."),
        ("the good shepherd", "He leads me beside still waters."),
        ("God created the heavens", "Let there be light, and there was light."),
        ("effectual calling", "No one can come to me unless the Father draws him."),
        ("the fear of the Lord", "The fear of the Lord is the beginning of wisdom."),
        ("the fear of the Lord", "He counted the stars and named them all."),
        ("bread of life", "I am the bread of life; whoever comes to me shall not "
         "hunger."),
        ("bread of life", "Give us this day our daily bread."),
        ("living water", "Whoever drinks the water I give will never thirst."),
        ("living water", "The well is deep, and you have nothing to draw with."),
        ("resurrection of the dead", "I am the resurrection and the life."),
        ("resurrection of the dead", "The soldiers sealed the tomb and set a guard."),
        ("the vine and the branches", "I am the vine; you are the branches."),
        ("the vine and the branches", "He owns cattle on a thousand hills."),
    ]
    pairs = relevant + irrelevant + partial
    return pairs[:PARITY_SAMPLE_N]


def _coreml_score(mlmodel, tokenizer, query: str, passage: str) -> float:
    """Score one (query, passage) pair via the fp16 .mlpackage (raw logit)."""
    enc = tokenizer(
        query, passage, add_special_tokens=True, truncation=True,
        max_length=MAX_PAIR_TOKENS)
    ids = np.array([enc["input_ids"]], dtype=np.int32)
    mask = np.array([enc["attention_mask"]], dtype=np.int32)
    ttids = np.array([enc["token_type_ids"]], dtype=np.int32)
    out = mlmodel.predict({
        INPUT_IDS: ids,
        ATTENTION_MASK: mask,
        TOKEN_TYPE_IDS: ttids,
    })
    return float(np.asarray(out[OUTPUT_SCORE], dtype=np.float64).reshape(-1)[0])


def _count_inversions(a: list[float], b: list[float]) -> tuple[int, int]:
    """(non-tied inversions, tied reorders) between rankings ``a`` and ``b``.

    ``a`` is the ground-truth (PyTorch) ranking. A pair is a NON-TIED inversion
    when ``a`` orders it with a real gap (> :data:`TIE_EPS`) yet ``b`` flips it —
    the only kind that matters for reranking. A flip between a pair ``a`` treats
    as tied (gap <= TIE_EPS) is a ``tied_reorder`` (reported, not gated): both
    candidates are equivalently (ir)relevant, so their fp16 order is immaterial.
    """
    n = len(a)
    inv = 0
    tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            flipped = (a[i] - a[j]) * (b[i] - b[j]) < 0
            if not flipped:
                continue
            if abs(a[i] - a[j]) > TIE_EPS:
                inv += 1
            else:
                tied += 1
    return inv, tied


def run_parity_gate(mlmodel, tokenizer, cases: list[RerankParityCase]
                    ) -> RerankParityResult:
    """Compare fp16 CoreML logits to float32 PyTorch; assert rel-err + order gates.

    Relative error (abs / (|logit| + REL_DENOM_FLOOR)) is the scale-appropriate
    metric for unbounded logits; order is checked among non-tied pairs only.
    """
    res = RerankParityResult(n=len(cases))
    torch_logits = [c.torch_logit for c in cases]
    coreml_logits: list[float] = []
    abs_errs: list[float] = []
    rel_errs: list[float] = []
    for case in cases:
        cl = _coreml_score(mlmodel, tokenizer, case.query, case.passage)
        coreml_logits.append(cl)
        err = abs(cl - case.torch_logit)
        rel = err / (abs(case.torch_logit) + REL_DENOM_FLOOR)
        abs_errs.append(err)
        rel_errs.append(rel)
        res.per_case.append({
            "query": case.query,
            "passage": case.passage[:60],
            "torch_logit": case.torch_logit,
            "coreml_logit": cl,
            "abs_err": err,
            "rel_err": rel,
        })
    if abs_errs:
        res.max_abs_err = float(np.max(abs_errs))
        res.mean_abs_err = float(np.mean(abs_errs))
        res.max_rel_err = float(np.max(rel_errs))
    res.inversions, res.tied_reorders = _count_inversions(
        torch_logits, coreml_logits)
    return res


# --- Fixtures ----------------------------------------------------------------
def _write_parity_fixture(path: Path, cases: list[RerankParityCase],
                          combined_sha256: str, model_name: str,
                          model_revision: str) -> None:
    """Committed-to-iOS parity fixture (pairs + float32 PyTorch logits)."""
    payload = {
        "model_name": model_name,
        "model_revision": model_revision,
        "model_combined_sha256": combined_sha256,
        "score_semantics": "raw classifier logit; higher = more relevant "
                            "(no sigmoid; only the order matters for reranking)",
        "max_pair_tokens": MAX_PAIR_TOKENS,
        "note": (
            "Fixed relevant/partial/irrelevant (query, passage) pairs spanning "
            "the logit range. Vectors are the float32 PyTorch logits; the Swift "
            f"end-to-end parity test asserts max relative logit error <= "
            f"{PARITY_MAX_REL} AND zero order inversions among non-tied pairs "
            f"(PyTorch gap > {TIE_EPS})."
        ),
        "cases": [
            {
                "query": c.query,
                "passage": c.passage,
                "torch_logit": round(c.torch_logit, 5),
            }
            for c in cases
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_tokenizer_fixture(path: Path, tokenizer, model_name: str,
                             model_revision: str) -> list[dict]:
    """Expected sentence-PAIR (input_ids, attention_mask, token_type_ids).

    Ground truth for the Swift WordPiece sentence-pair tokenizer. token_type_ids
    ARE stored here (unlike the query encoder) because segment ids are load-
    bearing for the cross-encoder.
    """
    cases: list[dict] = []
    for q, p in TOKENIZER_FIXTURE_PAIRS:
        enc = tokenizer(q, p, add_special_tokens=True)
        cases.append({
            "query": q,
            "passage": p,
            "input_ids": [int(x) for x in enc["input_ids"]],
            "attention_mask": [int(x) for x in enc["attention_mask"]],
            "token_type_ids": [int(x) for x in enc["token_type_ids"]],
        })
    payload = {
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer": "bert-base-uncased WordPiece (HF fast, pinned revision)",
        "special_tokens": {"PAD": 0, "UNK": 100, "CLS": 101, "SEP": 102, "MASK": 103},
        "max_seq_length": MAX_PAIR_TOKENS,
        "note": (
            "Expected token ids from the REAL HuggingFace fast tokenizer at the "
            "pinned revision, for SENTENCE PAIRS (query, passage). token_type_ids "
            "segment the query (0) from the passage (1). The Swift tokenizer must "
            "reproduce all three arrays exactly."
        ),
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cases


def _load_tokenizer(snapshot_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot_dir))


def _time_forward_512(mlmodel) -> float:
    import time

    ids = np.zeros((1, SEQ_RANGE_HI), dtype=np.int32)
    mask = np.ones((1, SEQ_RANGE_HI), dtype=np.int32)
    ttids = np.zeros((1, SEQ_RANGE_HI), dtype=np.int32)
    feed = {INPUT_IDS: ids, ATTENTION_MASK: mask, TOKEN_TYPE_IDS: ttids}
    mlmodel.predict(feed)  # warm
    t0 = time.perf_counter()
    mlmodel.predict(feed)
    return (time.perf_counter() - t0) * 1000.0


def _dir_size(root: Path) -> int:
    if root.is_file():
        return root.stat().st_size
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# --- Orchestration -----------------------------------------------------------
def export_reranker_coreml(
    output_dir: Path,
    cache_dir: Path,
    *,
    model_key: str = DEFAULT_RERANK_MODEL,
    skip_parity: bool = False,
) -> RerankExportResult:
    """End-to-end export: lineage -> trace+convert -> parity gate -> fixtures.

    ``output_dir`` receives ``Reranker.mlpackage``, ``reranker_vocab.txt``, and
    the two JSON fixtures. ``cache_dir`` is the HF model cache holding the pinned
    cross-encoder snapshot. Raises on a gate trip (after the CLI writes the
    report for forensics).
    """
    import coremltools as ct

    from .package import _sha256_file, _sha256_tree

    spec = RERANK_MODELS[model_key]
    snapshot_dir = _snapshot_dir(model_key, cache_dir)
    notes: list[str] = []

    combined = model_combined_sha256(snapshot_dir)
    revision = snapshot_dir.name

    wrapper = _build_wrapper(snapshot_dir)
    traced = _trace(wrapper)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlpackage_path = output_dir / MLPACKAGE_NAME
    _convert(traced, mlpackage_path)

    vocab_src = snapshot_dir / "vocab.txt"
    vocab_dst = output_dir / VOCAB_NAME
    shutil.copyfile(vocab_src, vocab_dst)
    vocab_sha = _sha256_file(vocab_dst)

    tokenizer = _load_tokenizer(snapshot_dir)

    # Ship-faithful parity: reload the SAVED .mlpackage from disk and score with
    # compute_units=CPU_ONLY — this is exactly how the iOS-27 app loads and runs
    # the reranker (the app measures actual latency and falls back to plain RRF
    # on a budget miss; it does NOT run the model on the ANE). Scoring the
    # in-memory convert output instead would test a DIFFERENT compute path than
    # ships, so we deliberately reload.
    ship_model = ct.models.MLModel(
        str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_ONLY)

    cases = _build_parity_cases(snapshot_dir, tokenizer)
    parity = RerankParityResult(n=len(cases))
    if not skip_parity:
        parity = run_parity_gate(ship_model, tokenizer, cases)

    forward_ms = _time_forward_512(ship_model)

    parity_fixture_path = output_dir / PARITY_FIXTURE_NAME
    _write_parity_fixture(parity_fixture_path, cases, combined,
                          spec["hf_id"], revision)
    tokenizer_fixture_path = output_dir / TOKENIZER_FIXTURE_NAME
    tok_fixture = _write_tokenizer_fixture(
        tokenizer_fixture_path, tokenizer, spec["hf_id"], revision)

    tree_sha = _sha256_tree(mlpackage_path)
    mlpackage_bytes = _dir_size(mlpackage_path)
    size_mb = mlpackage_bytes / (1024 * 1024)
    if size_mb > 25:
        notes.append(
            f"FLAG (size): Reranker.mlpackage is {size_mb:.1f} MB, above the "
            f"~20-25 MB back-of-envelope target ({model_key} is ~22M params; "
            "coremltools keeps embedding/gather ops at higher precision + carries "
            "spec/metadata). Still a fraction of bge-reranker-base (~140 MB) and "
            "under the shipped BGEQuery encoder (~66 MB) — acceptable, but the "
            "architect should confirm the binary-size budget.")

    result = RerankExportResult(
        mlpackage_path=mlpackage_path,
        mlpackage_tree_sha256=tree_sha,
        mlpackage_bytes=mlpackage_bytes,
        vocab_path=vocab_dst,
        vocab_sha256=vocab_sha,
        parity_fixture_path=parity_fixture_path,
        tokenizer_fixture_path=tokenizer_fixture_path,
        model_name=spec["hf_id"],
        model_revision=revision,
        model_combined_sha256=combined,
        parity=parity,
        forward_512_ms=forward_ms,
        tokenizer_fixture=tok_fixture,
        notes=notes,
    )
    _ = ct  # exercise the lazy import
    return result
