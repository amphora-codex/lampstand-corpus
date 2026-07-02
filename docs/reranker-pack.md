# On-device reranker pack — `Reranker.mlpackage` contract (app-side lane)

Authoritative app-integration contract for the on-device cross-encoder reranker
produced by `python -m lampstand_corpus.cli coreml-export-reranker`. The app lane
wires the reranker into `HybridRetriever` against THIS document. Quality gate +
SHIP verdict: `reports/reranker_eval_v1.md`; export provenance + parity numbers:
`reports/coreml_reranker_export.txt`.

This pack exists **only because the Rank-4 quality gate returned SHIP.** The gate
is eval-first by design: the reranker had to beat the much stronger re-chunked v2
hybrid (which roughly doubled ranking quality vs the pre-re-chunk retrieval the
audit measured), and it did — even on the lexical-favoring corpus gold set it
lifts commentary-anchor MRR +0.21 and overall MRR +0.08, with the full semantic
gain realized on the app's paraphrased-query eval.

## Source model + license

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — BERT MiniLM-L6 (384-hidden,
  6-layer, ~22M params), single-logit sequence-classification head.
- **License:** Apache-2.0 (model) / Apache-2.0 (bert-base-uncased WordPiece vocab).
  Recorded in `corpus_manifest.json` → `acknowledgements[id="reranker-model"]`
  and `packs.reranker`.
- **Why this model, not the higher-scoring one:** `BAAI/bge-reranker-base` (MIT)
  scored higher in the gate but is XLM-RoBERTa base (~278M params, ~140 MB fp16)
  — far over the on-device budget. It is recorded in `reranker_eval_v1.md` as a
  quality CEILING, not the shipped model. The shipped MiniLM still clears the
  SHIP bar and fits the device.
- **Shipped form:** a fp16 Core ML `.mlpackage` (the ORIGINAL weights are never
  shipped; the app loads the static `.mlpackage` from disk — no Python/SPM
  dependency, no runtime download). Measured size ~43 MB (FLAGGED in the export
  report; above the ~20-25 MB back-of-envelope but under the shipped ~66 MB
  BGEQuery encoder).

## Pack home + sync

Synced (never committed) exactly like the corpus packs and the BGEQuery encoder.
`corpus_manifest.json` → `packs.reranker` carries the `.mlpackage` (tree sha256)
+ `reranker_vocab.txt` (file sha256); `scripts/sync-corpus.sh verify()` picks
them up with its existing jq walk (`.packs | to_entries[] | .value.files[]`) —
zero jq changes. The `.mlpackage` sha256 is a deterministic **directory-tree**
hash (the tree is canonicalized: protobuf field order + stable Manifest UUIDs, so
re-exports are byte-identical — CLAUDE.md rule 6).

If the reranker pack is missing or fails checksum, retrieval degrades **gracefully
to plain RRF** — the reranker is a re-ordering pass over already-fused candidates
and must never be a hard dependency.

## Model I/O

Core ML `mlprogram`, `minimum_deployment_target = iOS18`, fp16.

| input | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int32 | (1, seq) | WordPiece ids for the `[CLS] query [SEP] passage [SEP]` pair |
| `attention_mask` | int32 | (1, seq) | 1 for real tokens, 0 for padding |
| `token_type_ids` | int32 | (1, seq) | **segment ids — LOAD-BEARING**: 0 over the query span (incl. leading `[CLS]` and the query's `[SEP]`), 1 over the passage span (incl. its trailing `[SEP]`) |

| output | dtype | meaning |
|---|---|---|
| `score` | fp | **raw relevance logit** — higher = more relevant. NO sigmoid. Only the ORDER matters for reranking; do not threshold the absolute value. |

- `seq` axis is flexible: `RangeDim(1, 512)`, default 32.
- **Max pair length ~192 tokens** (`meta`/`packs.reranker.max_pair_tokens = 192`).
  A short query + a single-verse or short-paragraph candidate fits comfortably;
  longer commentary candidates are truncated by the tokenizer at 192.

### `token_type_ids` differ from the BGE query encoder

The BGEQuery encoder always fed all-zero `token_type_ids` (single segment). The
reranker is a BERT **sentence-pair** model: the query is segment 0, the passage
segment 1. The Swift tokenizer MUST build the pair encoding and its segment ids
exactly. Ground truth: `reranker_tokenizer_fixture.json` (stores `input_ids`,
`attention_mask`, AND `token_type_ids` for six probe pairs). A segment-id bug is
the single most likely failure mode and would surface as an order flip in the
parity test below — do not ship on a fixture mismatch.

## Where it slots in the pipeline

```
BM25 half ─┐
           ├─ RRF fusion ── fused top-30 ── [RERANKER] ── tradition multiplier ── final
dense half ┘
```

- The reranker re-scores the **fused top-30** (query, passage) candidate pairs and
  re-sorts that window by descending `score`; the tail below 30 keeps its fused
  order. This is exactly the depth the quality gate measured (`RERANK_K = 30`) and
  the on-device pair budget (~30 forward passes/query).
- It runs **after RRF fusion and before the tradition multiplier** — the
  tradition weighting is applied to the reranked order, so tradition preference
  still has the last word over the semantically-reordered candidates.
- Ties / equal scores fall back to the fused order (stable sort), matching the
  app's id/anchor-ascending determinism carried through fusion.

## Latency budget + iOS-27 cpuOnly caveat

- The reranker is exported and **validated for `computeUnits = .cpuOnly`**. On
  iOS-27 the app runs the model cpuOnly (the parity gate reloads the saved
  `.mlpackage` and scores it cpuOnly precisely because that is the ship path; the
  ANE path is NOT what ships and is not validated here).
- 30 forward passes of a ~192-token pair per query is the cost. The app **measures
  actual on-device latency** and, on a **budget miss, falls through to plain RRF**
  (the un-reranked fused order) rather than blocking the result. The reranker is
  a best-effort quality boost, never a correctness dependency.

## Parity contract (what the Swift end-to-end test must assert)

The corpus-side export runs a ship-faithful gate (fp16 cpuOnly Core ML, reloaded
from disk, vs float32 PyTorch) over a fixed set of relevant / partial / irrelevant
pairs. Cross-encoder logits are **unbounded** (~[-12, +11]), so the gate is:

- **Relative** logit error, `|coreml − torch| / (|torch| + 1)`, must be
  `<= 0.02` (2%). An absolute-error floor is the wrong instrument for unbounded
  logits and is reported only for context.
- **Order preservation**: zero inversions among pairs whose PyTorch logits differ
  by more than a tie epsilon (0.05). Flips between logits tied within that epsilon
  are equivalently-(ir)relevant reorderings and are reported but not gated.

Ground-truth fixtures shipped to the app (under `output/models/`, synced):
`reranker_parity_fixture.json` (pairs + float32 PyTorch logits) and
`reranker_tokenizer_fixture.json` (expected token ids + attention mask + segment
ids). The Swift `RerankerTests` should assert the same relative-error + order
floors against these fixtures. A trip is a **stop-and-investigate** (almost
always a tokenizer / `token_type_ids` bug), never a threshold loosening.
