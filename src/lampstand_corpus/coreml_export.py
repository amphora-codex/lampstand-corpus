"""M4 T0 — export the on-device BGE-small query-embedding model to Core ML.

This is the corpus-side half of the M4 Tier-1 retrieval parity contract
(``docs/m4-technical-design.md`` §2). It traces the *exact pinned* BGE-small
weights that produced the stored corpus vectors into a fp16 ``.mlpackage`` whose
graph bakes in **CLS pooling + L2-normalize**, then proves — via a lineage gate
and a cosine-parity gate — that the exported model lands query vectors in the same
normalized space as the float32 passage vectors in ``bundled_search.sqlite``.

The exported model is a *build artifact*, not source: like the corpus ``.sqlite``
packs it is gitignored and synced into the iOS app (``scripts/sync-corpus.sh``)
with a manifest checksum. The iOS app gains no Python or SPM dependency from this
step — it ships a static ``.mlpackage`` it loads strictly from disk.

Parity contract honored here (verbatim from §2):
  * P2 — L2-normalize (and CLS pooling) baked INTO the Core ML graph; every output
    vector is unit-norm 384-d. Swift normalizes neither side.
  * P3 — passages are embedded BARE (no ``query_instruction`` prefix); the fixture
    cases are bare passage text validated against the stored bare-passage vectors.
  * P4 — fp16 acceptance: mean cosine >= 0.999 AND min cosine >= 0.995 vs the
    stored float32 passage vectors, every output L2 norm in [0.999, 1.001],
    dim == 384. A trip is a STOP-and-investigate, never a threshold loosening.
  * P5 — the tokenizer fixture is regenerated from the REAL HuggingFace fast
    tokenizer at the pinned revision so the Swift WordPiece tokenizer test asserts
    against ground truth, not hand-typed numbers.

Conversion path (§3.1):
  AutoModel.from_pretrained(snapshot, float32, eval) -> nn.Module wrapper
  [BERT encoder -> last_hidden_state[:,0] (CLS) -> F.normalize(p=2,dim=1)]
  -> torch.jit.trace((1,16)) -> ct.convert(mlprogram, FLOAT16, iOS18).
  Inputs: input_ids / attention_mask / token_type_ids (int32, batch 1,
  seq RangeDim(1,512,default=16)). Output: ``embedding`` (384-d, unit norm).

Determinism (CLAUDE.md pipeline rule 6): trace + convert on CPU with the same
deterministic knobs ``encode.py`` uses; no timestamps in output. The tool records
in the report whether a re-run yields a byte-identical ``.mlpackage`` tree.

Requires the ``[coreml]`` extra (``pip install -e ".[coreml]"``) on top of the
``[embeddings]`` torch/transformers stack; coremltools is imported lazily so the
light pipeline phases stay importable without it.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embeddings import EMBED_DIM, MODEL_NAME, MODEL_REVISION

# --- Parity contract constants (single source of truth, §2) ------------------
# fp16 acceptance floors (P4). A trip is a stop-and-investigate, never loosened.
PARITY_MEAN_MIN = 0.999
PARITY_WORST_MIN = 0.995
# Per-output unit-norm tolerance (P2). The graph bakes in L2-normalize.
NORM_LO = 0.999
NORM_HI = 1.001

# Number of deterministically-sampled bundled chunks in the parity fixture (§3.1).
PARITY_SAMPLE_N = 30

# Example trace shape (§3.1): batch 1, seq 16. The exported seq axis is flexible.
TRACE_SEQ_LEN = 16
# Core ML sequence axis bounds (§2 cross-cut 7): RangeDim(1, 512, default=16).
SEQ_RANGE_LO = 1
SEQ_RANGE_HI = 512
SEQ_RANGE_DEFAULT = 16
SEQ_LEN_LABEL = "RangeDim(1,512)"  # recorded in the manifest so Swift padding matches

# Expected lineage hash: the model_combined_sha256 stored in bundled_search.sqlite
# meta (the weights behind the corpus vectors). Verified against encode.model_provenance().
EXPECTED_MODEL_COMBINED_SHA256 = (
    "f6a428ffdb6afebc801a94fd3454aec2705f198ad710434a7919faa2a41b361b"
)
# Expected vocab.txt sha256 at the pinned revision (flagged, not failed, on drift).
EXPECTED_VOCAB_SHA256 = (
    "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
)

# Tokenizer-fixture probe strings (§3.1 / §3.2): accent, apostrophe-split, numerals,
# punctuation, empty, and the full query-path string. Ground truth for the Swift
# tokenizer parity test (T1/T6).
TOKENIZER_FIXTURE_STRINGS = [
    "propitiation",
    "justification",
    "God's love",
    "régénération",
    "1 John 2:2",
    "Yahweh-Jireh",
    "",
    "Represent this sentence for searching relevant passages: propitiation",
]

# Output artifact names.
MLPACKAGE_NAME = "BGEQuery.mlpackage"
VOCAB_NAME = "vocab.txt"
PARITY_FIXTURE_NAME = "bge_parity_fixture.json"
TOKENIZER_FIXTURE_NAME = "bge_tokenizer_fixture.json"

# Core ML model I/O names — must match the Swift embedder exactly (§2 cross-cut 7).
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
TOKEN_TYPE_IDS = "token_type_ids"
OUTPUT_EMBEDDING = "embedding"


class LineageError(RuntimeError):
    """The snapshot's model_combined_sha256 does not match the corpus vectors'."""


class ParityError(RuntimeError):
    """The fp16 .mlpackage failed the P4 cosine/norm/dim acceptance gate."""


@dataclass
class ParityCase:
    chunk_id: str
    resource_type: str
    anchor: str
    text: str
    stored_vector: np.ndarray  # float32 (EMBED_DIM,), L2-normalized passage vector


@dataclass
class ParityResult:
    n: int = 0
    mean_cosine: float = 0.0
    min_cosine: float = 0.0
    max_cosine: float = 0.0
    min_norm: float = 0.0
    max_norm: float = 0.0
    dim: int = 0
    per_case: list[dict] = field(default_factory=list)
    multi_subword_anchors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.dim == EMBED_DIM
            and self.mean_cosine >= PARITY_MEAN_MIN
            and self.min_cosine >= PARITY_WORST_MIN
            and self.min_norm >= NORM_LO
            and self.max_norm <= NORM_HI
        )


@dataclass
class ExportResult:
    mlpackage_path: Path
    mlpackage_tree_sha256: str
    mlpackage_bytes: int
    vocab_path: Path
    vocab_sha256: str
    vocab_sha256_matches_expected: bool
    parity_fixture_path: Path
    tokenizer_fixture_path: Path
    model_combined_sha256: str
    parity: ParityResult
    forward_512_ms: float = 0.0
    tokenizer_fixture: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --- Snapshot location (reuse encode.py's pinned-snapshot logic) -------------
def _snapshot_dir() -> Path:
    from .encode import _snapshot_dir as enc_snapshot_dir

    return enc_snapshot_dir()


# --- Lineage gate ------------------------------------------------------------
def assert_lineage(snapshot_dir: Path) -> str:
    """Compute the pipeline's own model provenance and assert it matches the corpus.

    Reuses ``encode.model_provenance()`` (sha256 over sorted ``path:sha256`` lines)
    so the number computed here is bit-identical to the one recorded in
    ``bundled_search.sqlite.meta``. A mismatch means the snapshot being converted is
    NOT the weights behind the stored corpus vectors — a hard stop.
    """
    from .encode import model_provenance

    prov = model_provenance()
    combined = prov["combined_sha256"]
    if combined != EXPECTED_MODEL_COMBINED_SHA256:
        raise LineageError(
            "LINEAGE GATE FAILED: the snapshot at\n"
            f"  {snapshot_dir}\n"
            f"hashes to combined_sha256={combined}\n"
            f"but the corpus vectors were built from "
            f"{EXPECTED_MODEL_COMBINED_SHA256}.\n"
            "The .mlpackage would NOT trace to the weights behind the stored "
            "vectors. Re-snapshot the pinned revision and rebuild."
        )
    return combined


# --- Torch wrapper: BERT -> CLS pooling -> L2 normalize (P2) ------------------
def _build_wrapper(snapshot_dir: Path):
    """Load BGE-small (float32, eval) and wrap it to emit a unit-norm CLS vector.

    Verifies the pooling config IS CLS (not mean) at load time — the contract (P2)
    depends on it. The wrapper bakes BOTH pooling and L2-normalize into the graph so
    the exported model needs no Swift-side pooling or normalization.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel

    from .encode import _set_deterministic_cpu

    _set_deterministic_cpu()

    # Hard-verify CLS pooling (NOT mean) against the snapshot's own config — the
    # contract is load-bearing and must not be assumed.
    pooling_cfg = json.loads(
        (snapshot_dir / "1_Pooling" / "config.json").read_text(encoding="utf-8")
    )
    if not pooling_cfg.get("pooling_mode_cls_token", False) or pooling_cfg.get(
        "pooling_mode_mean_tokens", False
    ):
        raise RuntimeError(
            "Pooling config is not CLS-only: "
            f"{pooling_cfg!r}. The wrapper bakes CLS pooling; refusing to proceed "
            "on a mean-pooled config (would silently corrupt parity)."
        )

    bert = AutoModel.from_pretrained(str(snapshot_dir), torch_dtype=torch.float32)
    bert.eval()

    class BGEQueryModule(torch.nn.Module):
        def __init__(self, encoder: torch.nn.Module) -> None:
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # CLS pooling: the [CLS] token's last hidden state (verified above).
            cls = out.last_hidden_state[:, 0]
            # L2-normalize so the output lands in the same space as corpus vectors.
            return F.normalize(cls, p=2, dim=1)

    wrapper = BGEQueryModule(bert)
    wrapper.eval()
    return wrapper


def _trace(wrapper):
    """torch.jit.trace the wrapper with an example shape (1, TRACE_SEQ_LEN)."""
    import torch

    ids = torch.zeros((1, TRACE_SEQ_LEN), dtype=torch.int32)
    mask = torch.ones((1, TRACE_SEQ_LEN), dtype=torch.int32)
    ttids = torch.zeros((1, TRACE_SEQ_LEN), dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ids, mask, ttids), check_trace=False)
    return traced


_NEW_ONES_REGISTERED = False


def _register_new_ones_op() -> None:
    """Register an ``aten::new_ones`` MIL translation (idempotent).

    transformers 5.x's ``masking_utils`` emits ``Tensor.new_ones(size, dtype=...)``
    while building the attention mask (``q_idx.new_ones((), dtype=torch.bool)``),
    and coremltools 9 has no built-in translation for it. ``new_ones`` is exactly
    ``ones`` with a leading *reference tensor* argument: it returns a tensor of the
    given ``size`` filled with 1, in the requested ``dtype`` (defaulting to the
    reference's). We mirror coremltools' own ``ones`` op (``mb.fill`` + cast).

    Traced node layout (verified): ``[self, size, dtype, layout, device, pin]``.

    Safety: this is a mask-construction helper, not a weight path. Any error in this
    translation would surface as a cosine miss in the P4 parity gate (the gate
    embeds against the ground-truth corpus vectors), so a wrong translation is
    *caught*, never silently shipped. Registered lazily so importing this module
    without coremltools stays cheap.
    """
    global _NEW_ONES_REGISTERED
    if _NEW_ONES_REGISTERED:
        return
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import (
        NUM_TO_DTYPE_STRING,
        _cast_to,
        _get_inputs,
    )
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        register_torch_op,
    )

    @register_torch_op(torch_alias=["new_ones"], override=True)
    def lampstand_new_ones(context, node):  # noqa: ANN001 - coremltools signature
        inputs = _get_inputs(context, node, min_expected=2)
        size = inputs[1]
        dtype = inputs[2] if (len(inputs) > 2 and inputs[2] is not None) else None

        # Resolve the requested shape into an int32 shape Var/list for mb.fill.
        def _empty_size(s) -> bool:
            # The scalar case new_ones((), ...) traces as an empty size: an empty
            # python list, or a 0-length tensor/Var.
            if isinstance(s, (list, tuple)):
                return len(s) == 0
            val = getattr(s, "val", None)
            if val is not None:
                try:
                    return len(val) == 0
                except TypeError:
                    return False
            shp = getattr(s, "shape", None)
            return bool(shp) and shp == (0,)

        if _empty_size(size):
            # 0-d scalar tensor of value 1 → emit a 1-element tensor then squeeze.
            res = mb.fill(shape=[1], value=1.0)
            res = mb.squeeze(x=res)
        elif isinstance(size, (list, tuple)):
            res = mb.fill(shape=mb.concat(values=size, axis=0), value=1.0)
        else:
            # ``size`` is already an int32 shape Var.
            res = mb.fill(shape=size, value=1.0)
        dtype_str = NUM_TO_DTYPE_STRING[dtype.val] if dtype is not None else None
        if dtype_str is not None:
            res = _cast_to(res, dtype_str, node.name)
        context.add(res, node.name)

    _NEW_ONES_REGISTERED = True


def _convert(traced, out_path: Path):
    """ct.convert -> fp16 mlprogram, iOS18, RangeDim(1,512) seq axis, int32 I/O."""
    import coremltools as ct

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
    outputs = [ct.TensorType(name=OUTPUT_EMBEDDING)]
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
    _canonicalize_mlpackage(out_path)
    return mlmodel


def _canonicalize_model_spec(pkg: Path) -> None:
    """Re-serialize ``model.mlmodel`` with deterministic protobuf field ordering.

    Round-trips the spec through ``Model.ParseFromString`` ->
    ``SerializeToString(deterministic=True)``. This is a lossless canonicalization
    (same message, stable byte order); the model's predictions are unchanged
    (verified bit-identical). Without it, coremltools emits a differently-ordered
    protobuf each process, defeating the tree hash.
    """
    spec_path = pkg / "Data" / "com.apple.CoreML" / "model.mlmodel"
    if not spec_path.exists():
        return
    from coremltools.proto import Model_pb2

    spec = Model_pb2.Model()
    spec.ParseFromString(spec_path.read_bytes())
    spec_path.write_bytes(spec.SerializeToString(deterministic=True))


def _canonicalize_mlpackage(pkg: Path) -> None:
    """Make the .mlpackage tree byte-reproducible across separate processes.

    Two sources of cross-process non-determinism exist (both verified to NOT change
    model behavior — predictions are bit-identical, max abs diff 0.0):

      1. ``Manifest.json`` embeds a freshly-minted random UUID per item per
         conversion. Core ML only requires ``rootModelIdentifier`` to point at the
         model-spec entry, so we replace them with stable uuid5-derived ids.
      2. ``model.mlmodel`` is a protobuf whose field/op ordering varies across
         processes (coremltools' MIL->proto serialization is not stable run-to-run;
         the weights ``weight.bin`` ARE byte-identical). We re-serialize the spec
         with protobuf's ``deterministic=True`` flag, which canonicalizes field
         order without altering the message — yielding a byte-identical spec.

    Together these make the whole tree reproducible (CLAUDE.md rule 6) so the
    manifest tree-sha256 is constant across re-conversions and the iOS sync
    checksum never spuriously drifts. Neither step touches model behavior.
    """
    _canonicalize_model_spec(pkg)

    manifest_path = pkg / "Manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("itemInfoEntries", {})
    # Stable, valid-format UUIDs keyed on the item's logical name (not its random
    # id). Derived deterministically via uuid5 over a fixed namespace so they are
    # constant across runs yet still well-formed 8-4-4-4-12 hex.
    import uuid as _uuid

    ns = _uuid.UUID("00000000-0000-0000-0000-000000000000")
    stable_by_name = {
        "weights": str(_uuid.uuid5(ns, "lampstand.bgequery.weights")).upper(),
        "model.mlmodel": str(
            _uuid.uuid5(ns, "lampstand.bgequery.model")
        ).upper(),
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
        json.dumps(out, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# --- Parity gate (P4) --------------------------------------------------------
def _sample_parity_cases(search_db: Path, tokenizer) -> list[ParityCase]:
    """Deterministically sample bundled chunks for the parity fixture.

    Selection (fixed, reproducible — no RNG):
      1. The first ``PARITY_SAMPLE_N`` chunks by ``ORDER BY id`` (the stable
         content-addressed id), so the same DB always yields the same set.
      2. Guarantee WordPiece-splitting coverage: ensure at least a few chunks whose
         text contains a word that splits into >=3 subwords are present. Because the
         bundled BSB+WSC index renders Rom 3:25 "atoning sacrifice" (no
         "propitiation" string in it — verified), we inject the lowest-id chunks
         carrying multi-subword theological words (sanctification, transgression,
         redeemer, everlasting, glorify) if the first-N window missed them. This
         keeps the fixture deterministic while exercising the splitter (the §3.1
         multi-subword requirement). Flagged in the report since "propitiation"
         itself is absent from the public-domain bundled subset.
    """
    conn = sqlite3.connect(search_db)
    try:
        base = conn.execute(
            "SELECT c.id, c.resource_type, c.anchor, c.text, e.vector "
            "FROM chunk c JOIN embedding e ON e.chunk_id = c.id "
            "ORDER BY c.id LIMIT ?",
            (PARITY_SAMPLE_N,),
        ).fetchall()
        chosen_ids = {r[0] for r in base}

        # Multi-subword coverage words present in the bundled corpus (verified).
        coverage_words = [
            "sanctification", "transgression", "redeemer", "everlasting", "glorify",
        ]
        extra: list[tuple] = []
        for word in coverage_words:
            if _row_has_multi_subword(base, word, tokenizer):
                continue  # already covered by the first-N window
            row = conn.execute(
                "SELECT c.id, c.resource_type, c.anchor, c.text, e.vector "
                "FROM chunk c JOIN embedding e ON e.chunk_id = c.id "
                "WHERE lower(c.text) LIKE ? ORDER BY c.id LIMIT 1",
                ("%" + word + "%",),
            ).fetchone()
            if row and row[0] not in chosen_ids:
                extra.append(row)
                chosen_ids.add(row[0])
        rows = base + extra
    finally:
        conn.close()

    cases: list[ParityCase] = []
    for cid, rtype, anchor, text, blob in rows:
        vec = np.frombuffer(blob, dtype="<f4").astype(np.float32)
        cases.append(ParityCase(
            chunk_id=cid, resource_type=rtype, anchor=anchor, text=text,
            stored_vector=vec,
        ))
    return cases


def _row_has_multi_subword(rows: list[tuple], word: str, tokenizer) -> bool:
    """True if any sampled row's text contains ``word`` (a >=3-piece token)."""
    pieces = tokenizer(word, add_special_tokens=False)["input_ids"]
    if len(pieces) < 3:
        return False
    return any(word in (r[3] or "").lower() for r in rows)


def _coreml_embed_bare(mlmodel, tokenizer, text: str) -> np.ndarray:
    """Embed one BARE passage (no query prefix, P3) via the fp16 .mlpackage.

    Tokenizes with the real HF fast tokenizer (the Swift tokenizer's ground truth),
    feeds int32 arrays under the contract input names, and reads the 384-d output.
    """
    enc = tokenizer(
        text, add_special_tokens=True, truncation=True, max_length=SEQ_RANGE_HI
    )
    ids = np.array([enc["input_ids"]], dtype=np.int32)
    mask = np.array([enc["attention_mask"]], dtype=np.int32)
    ttids = np.zeros_like(ids, dtype=np.int32)
    out = mlmodel.predict({
        INPUT_IDS: ids,
        ATTENTION_MASK: mask,
        TOKEN_TYPE_IDS: ttids,
    })
    return np.asarray(out[OUTPUT_EMBEDDING], dtype=np.float32).reshape(-1)


def run_parity_gate(
    mlmodel, tokenizer, cases: list[ParityCase]
) -> ParityResult:
    """Embed each fixture case bare via the fp16 model; compare cosine to stored.

    Asserts the P4 floors and the unit-norm/dim invariants. Returns the populated
    :class:`ParityResult` either way (the CLI raises :class:`ParityError` on fail so
    the report is still written for forensics).
    """
    res = ParityResult(n=len(cases))
    cosines: list[float] = []
    norms: list[float] = []
    for case in cases:
        qv = _coreml_embed_bare(mlmodel, tokenizer, case.text)
        if res.dim == 0:
            res.dim = int(qv.shape[0])
        norm = float(np.linalg.norm(qv))
        norms.append(norm)
        sv = case.stored_vector
        # Both are (meant to be) unit-norm, so dot == cosine; guard with the actual
        # norms in case the stored vector drifted from 1.0.
        denom = norm * float(np.linalg.norm(sv)) + 1e-12
        cos = float(np.dot(qv, sv) / denom)
        cosines.append(cos)
        res.per_case.append({
            "chunk_id": case.chunk_id,
            "resource_type": case.resource_type,
            "anchor": case.anchor,
            "cosine": cos,
            "fp16_norm": norm,
        })
    if cosines:
        res.mean_cosine = float(np.mean(cosines))
        res.min_cosine = float(np.min(cosines))
        res.max_cosine = float(np.max(cosines))
    if norms:
        res.min_norm = float(np.min(norms))
        res.max_norm = float(np.max(norms))
    return res


# --- Fixtures ----------------------------------------------------------------
def _write_parity_fixture(
    path: Path, cases: list[ParityCase], combined_sha256: str
) -> None:
    """Write the committed-to-iOS parity fixture (~50 KB; bare passage vectors)."""
    payload = {
        "model_revision": MODEL_REVISION,
        "model_name": MODEL_NAME,
        "model_combined_sha256": combined_sha256,
        "embedding_dim": EMBED_DIM,
        "note": (
            "Bare-passage cases sampled deterministically (ORDER BY id) from the "
            "public-domain bundled BSB+WSC search index. Vectors are the stored "
            "float32 passage embeddings (no query_instruction prefix, P3). The "
            "Swift end-to-end parity test (T6) asserts cosine(fp16 query model, "
            "stored vector) >= 0.995 worst-case (P4)."
        ),
        "cases": [
            {
                "chunk_id": c.chunk_id,
                "resource_type": c.resource_type,
                "anchor": c.anchor,
                "text": c.text,
                # 6 decimals: float32 carries ~7 significant digits and the parity
                # gate only needs cosine >= 0.995, so this is lossless for the test
                # while roughly halving the on-disk fixture size.
                "vector": [round(float(x), 6) for x in c.stored_vector.tolist()],
            }
            for c in cases
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_tokenizer_fixture(path: Path, tokenizer) -> list[dict]:
    """Regenerate expected (input_ids, attention_mask) from the real HF tokenizer.

    Ground truth for the Swift WordPiece tokenizer parity test (T1/T6). Strings
    cover accent strip, apostrophe split, numerals, punctuation, the empty string,
    and the full query-path string.
    """
    cases: list[dict] = []
    for s in TOKENIZER_FIXTURE_STRINGS:
        enc = tokenizer(s, add_special_tokens=True)
        cases.append({
            "text": s,
            "input_ids": [int(x) for x in enc["input_ids"]],
            "attention_mask": [int(x) for x in enc["attention_mask"]],
        })
    payload = {
        "model_revision": MODEL_REVISION,
        "model_name": MODEL_NAME,
        "tokenizer": "bert-base-uncased WordPiece (HF fast, pinned revision)",
        "special_tokens": {"PAD": 0, "UNK": 100, "CLS": 101, "SEP": 102, "MASK": 103},
        "max_seq_length": SEQ_RANGE_HI,
        "note": (
            "Expected token ids from the REAL HuggingFace fast tokenizer at the "
            "pinned revision. token_type_ids are always all-zero (single segment) "
            "and are not stored here. The Swift BGETokenizer must reproduce these "
            "exactly (P5: NFD + drop-Mn, NO NFKC, apostrophe-splits)."
        ),
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return cases


# --- Orchestration -----------------------------------------------------------
def _load_tokenizer(snapshot_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot_dir))


def _time_forward_512(mlmodel) -> float:
    """Time a single 512-token forward pass (ms) for the RangeDim-residency note."""
    import time

    ids = np.zeros((1, SEQ_RANGE_HI), dtype=np.int32)
    mask = np.ones((1, SEQ_RANGE_HI), dtype=np.int32)
    ttids = np.zeros((1, SEQ_RANGE_HI), dtype=np.int32)
    feed = {INPUT_IDS: ids, ATTENTION_MASK: mask, TOKEN_TYPE_IDS: ttids}
    mlmodel.predict(feed)  # warm
    t0 = time.perf_counter()
    mlmodel.predict(feed)
    return (time.perf_counter() - t0) * 1000.0


def export_coreml(
    output_dir: Path,
    search_db: Path,
    *,
    skip_parity: bool = False,
) -> ExportResult:
    """End-to-end export: lineage gate -> trace+convert -> parity gate -> fixtures.

    ``output_dir`` receives ``BGEQuery.mlpackage``, ``vocab.txt``, and the two JSON
    fixtures. ``search_db`` is the bundled BSB+WSC index whose stored float32 passage
    vectors the parity gate validates against. Raises :class:`LineageError` or
    :class:`ParityError` on a gate trip (after writing the report from the CLI).
    """
    import coremltools as ct

    snapshot_dir = _snapshot_dir()
    notes: list[str] = []

    # 1) Lineage gate — hard stop if the snapshot is not the corpus weights.
    combined = assert_lineage(snapshot_dir)

    # 2) Trace + convert (fp16 mlprogram, CLS+L2 baked in, RangeDim seq axis).
    wrapper = _build_wrapper(snapshot_dir)
    traced = _trace(wrapper)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlpackage_path = output_dir / MLPACKAGE_NAME
    mlmodel = _convert(traced, mlpackage_path)

    # 3) Copy vocab.txt verbatim + checksum.
    from .package import _sha256_file, _sha256_tree

    vocab_src = snapshot_dir / "vocab.txt"
    vocab_dst = output_dir / VOCAB_NAME
    shutil.copyfile(vocab_src, vocab_dst)
    vocab_sha = _sha256_file(vocab_dst)
    vocab_matches = vocab_sha == EXPECTED_VOCAB_SHA256
    if not vocab_matches:
        notes.append(
            f"FLAG: vocab.txt sha256 {vocab_sha} != expected "
            f"{EXPECTED_VOCAB_SHA256} — verify the pinned snapshot."
        )

    tokenizer = _load_tokenizer(snapshot_dir)

    # 4) Parity gate (P4) over bare-passage fixture cases.
    cases = _sample_parity_cases(search_db, tokenizer)
    parity = ParityResult(n=len(cases), dim=EMBED_DIM)
    if not skip_parity:
        parity = run_parity_gate(mlmodel, tokenizer, cases)
        parity.multi_subword_anchors = _multi_subword_anchors(cases, tokenizer)

    # 5) Time a 512-token forward pass for the residency note.
    forward_ms = _time_forward_512(mlmodel)

    # 6) Write fixtures.
    parity_fixture_path = output_dir / PARITY_FIXTURE_NAME
    _write_parity_fixture(parity_fixture_path, cases, combined)
    tokenizer_fixture_path = output_dir / TOKENIZER_FIXTURE_NAME
    tok_fixture = _write_tokenizer_fixture(tokenizer_fixture_path, tokenizer)

    # 7) Hash the .mlpackage directory tree (manifest integrity for sync).
    tree_sha = _sha256_tree(mlpackage_path)
    mlpackage_bytes = _dir_size(mlpackage_path)

    # Flag the parity-fixture size vs the design-doc ~50 KB target. With 30+ cases
    # carrying both the full passage text AND a 384-float vector, ~50 KB is not
    # reachable (vectors alone are ~1.5 KB/case); the architect decides whether to
    # commit as-is or trim cases (O-F).
    fixture_kb = parity_fixture_path.stat().st_size / 1024
    if fixture_kb > 60:
        notes.append(
            f"FLAG (architect O-F): bge_parity_fixture.json is {fixture_kb:.0f} KB "
            f"(> the design-doc ~50 KB target). Driven by {len(cases)} cases x "
            f"(full passage text + 384 floats). Trimming cases would weaken the "
            f"gate; recommend committing as-is or the architect picking a case cap."
        )

    result = ExportResult(
        mlpackage_path=mlpackage_path,
        mlpackage_tree_sha256=tree_sha,
        mlpackage_bytes=mlpackage_bytes,
        vocab_path=vocab_dst,
        vocab_sha256=vocab_sha,
        vocab_sha256_matches_expected=vocab_matches,
        parity_fixture_path=parity_fixture_path,
        tokenizer_fixture_path=tokenizer_fixture_path,
        model_combined_sha256=combined,
        parity=parity,
        forward_512_ms=forward_ms,
        tokenizer_fixture=tok_fixture,
        notes=notes,
    )
    _ = ct  # ensure the import is exercised (lazy dependency check)
    return result


def _multi_subword_anchors(cases: list[ParityCase], tokenizer) -> list[str]:
    """Anchors of fixture cases that contain a >=3-subword word (coverage proof)."""
    coverage_words = [
        "sanctification", "transgression", "redeemer", "everlasting", "glorify",
    ]
    out: list[str] = []
    for case in cases:
        low = (case.text or "").lower()
        for word in coverage_words:
            if word in low and len(
                tokenizer(word, add_special_tokens=False)["input_ids"]
            ) >= 3:
                out.append(f"{case.anchor} ({word})")
                break
    return out


def _dir_size(root: Path) -> int:
    if root.is_file():
        return root.stat().st_size
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
