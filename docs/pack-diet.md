# Pack diet v2 — search/vectors pack contract (app-side reader migration)

This is the authoritative schema + encoding contract for the v2 corpus packs
produced by `python -m lampstand_corpus.cli package`. The corpus half is done;
the app-side pack-resolution migration (SearchStore / DenseRetriever /
ChunkRepository in `lampstand-ios`) builds against THIS document.

Measured outcome and rationale: `reports/pack_diet_v1.md`.

## What changed (v1 → v2)

| v1 | v2 |
|---|---|
| `ondemand_embeddings.sqlite` — byte-copy of the built `embeddings.sqlite` (1.99 GB) | **`ondemand_search.sqlite`** (BM25 + chunk metadata + display text) + **`ondemand_vectors.sqlite`** (dense vectors) |
| `bundled_search.sqlite` (same v1 schema, BSB+WSC scope, 57.5 MB) | `bundled_search.sqlite` — same name/scope, v2 encoding, vectors embedded in the same file |
| `chunk.id` TEXT (28-char content hash) everywhere | `chunk.id` INTEGER (stable per corpus version); the string id is preserved as `chunk.string_id` |
| `bm25_posting` table — one row per (term, chunk) + `(term_id, chunk_id)` PK index + `idx_posting_chunk` | one `postings` BLOB per term on `bm25_term` (varint-delta; see below). The posting table and both its indexes are GONE |
| `bm25_doc` table (chunk_id → length) | folded into `chunk.doc_len` |
| `embedding.vector` float32-le, `embedding.dim` column | `embedding.vector` int8 (default) with `embedding.scale` REAL; dim comes from `meta.embedding_dim`. `package fp32` keeps float32 bytes (`scale = 1.0`) |

Nothing else moved: `bundled_bibles/confessions`, `ondemand_bibles/confessions/
commentaries/lexicons/crossrefs` are unchanged. The BUILD artifact
`output/embeddings.sqlite` also keeps its v1 schema — only the packs changed.

A sibling bundled pack, `bundled_crossrefs.sqlite` (TSK edge table + the
per-pericope expansion over the SAME integer chunk ids), has its own contract:
`docs/crossrefs-pack.md`.

## Integer chunk ids

- Assigned **1-based, by ascending string chunk id**, over the WHOLE corpus
  (bundled subset included), so a chunk has ONE integer id across every pack of
  a corpus version (`meta.id_assignment = "ascending-string-chunk-id-1based"`).
- Deterministic: string ids are content-addressed, so the same inputs always
  yield the same mapping.
- **Per-corpus-version only**: adding/removing any chunk renumbers. Cross-version
  identity stays with `string_id`. Never persist integer ids across corpus
  updates on the app side.

## `ondemand_search.sqlite` / `bundled_search.sqlite` (`meta.format = "search-pack-v2"`)

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chunk (
    id            INTEGER PRIMARY KEY,  -- stable int id (per corpus version)
    string_id     TEXT NOT NULL UNIQUE, -- content-addressed id (provenance)
    resource_type TEXT NOT NULL,        -- scripture|commentary|confession|lexicon
    source        TEXT NOT NULL,
    anchor        TEXT NOT NULL,
    book          TEXT, chapter INTEGER, verse_start INTEGER, verse_end INTEGER,
    key           TEXT,
    text          TEXT,                 -- display text; NULL when meta.text_included="0", and ALWAYS NULL for pericope parents (indexed=0) — reconstruct from children (see "Dual granularity")
    text_checksum TEXT NOT NULL,        -- sha256 of header+text
    truncated     INTEGER NOT NULL DEFAULT 0,
    doc_len       INTEGER NOT NULL,     -- BM25 token count (0 for parents)
    header        TEXT NOT NULL DEFAULT '', -- structural header ("Psalms 23:1 — ")
    parent_id     INTEGER,              -- pericope parent int id (Scripture children)
    indexed       INTEGER NOT NULL DEFAULT 1, -- 0 = context-only pericope parent
    question      INTEGER,              -- catechism metadata (confession chunks)
    lords_day     INTEGER,
    conf_chapter  INTEGER,
    conf_section  INTEGER,
    article       INTEGER
);
CREATE INDEX idx_chunk_resource ON chunk (resource_type);
CREATE INDEX idx_chunk_source   ON chunk (source);
CREATE INDEX idx_chunk_ref      ON chunk (book, chapter, verse_start);
CREATE INDEX idx_chunk_parent   ON chunk (parent_id);
CREATE TABLE bm25_term (
    term_id  INTEGER PRIMARY KEY,
    term     TEXT NOT NULL UNIQUE,
    doc_freq INTEGER NOT NULL,
    postings BLOB NOT NULL               -- see "Posting blob encoding"
);
CREATE TABLE bm25_stats (key TEXT PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE expansion (              -- Rank 7 query expansion (expansion-v1)
    term      TEXT NOT NULL,          -- query token (pipeline tokenizer form)
    expansion TEXT NOT NULL,          -- token to ALSO score, down-weighted
    kind      TEXT NOT NULL,          -- archaic | suffix | synonym (approved only)
    weight    REAL NOT NULL,          -- mining containment (informational)
    PRIMARY KEY (term, expansion)
);
-- bundled_search.sqlite ONLY additionally contains the `embedding` table below.
```

### Dual granularity (Rank 8)

Scripture rows come in two populations: **children** (single verses,
`indexed=1`, `parent_id` set — the ONLY rankable Scripture rows) and
**pericope parents** (`indexed=0`, `parent_id NULL`, `doc_len 0`, present in no
posting blob and no vectors pack). Retrieve at verse precision; expand a hit to
its parent for LLM context via one `chunk` lookup on `parent_id`.
Every indexed chunk's EMBEDDED/BM25 text was `header || text` — query-side
scoring needs no change, but display should render `text` (and may show
`header` as an eyebrow label). Dense vectors exist for BSB children +
non-scripture only (KJV/ASV/WEB children are BM25-only).

> **CONTRACT NOTE — parent chunk `text` is NULL (app must reconstruct).**
> As of the pack-diet size fix, **pericope PARENT rows (`indexed=0`) store
> `text = NULL`** in `ondemand_search.sqlite` / `bundled_search.sqlite`. A
> parent's display/context text is exactly the concatenation of its children's
> verse `text`, and those children are ALSO in the pack, so storing the parent
> text was pure redundancy (~20 MiB in `ondemand_search`).
>
> **The app must reconstruct a parent's text by concatenating its children's
> `text` in child order.** Child order = ascending `(book, chapter,
> verse_start)` — equivalently the `idx_chunk_ref` order — over the rows whose
> `parent_id` equals the parent's `id`. Join with a single indexed query:
> `SELECT text FROM chunk WHERE parent_id = ? AND indexed = 1
>  ORDER BY chapter, verse_start`. Insert a single space between verses (the
> children's `text` carries no leading/trailing whitespace); prepend the
> parent's own `header` if a labeled context block is wanted. Do NOT expect a
> parent `text` column value — it is always NULL for `indexed=0` rows.
>
> Children remain fully self-sufficient: every `indexed=1` row keeps its full
> `text`, so search-result rendering (which only ever surfaces child hits)
> needs no reconstruction and is unaffected. `text_checksum` on a parent is
> preserved as build-time provenance (it hashes the original `header||text`)
> and therefore no longer verifies against the now-NULL `text` — do not
> checksum-validate parent rows.

### Query expansion (`expansion` table)

At query time, for each tokenized query term look up `expansion` rows and ALSO
score those tokens at a flat down-weight (the corpus eval uses **0.3**; the
`weight` column is informational per-pair mining confidence). Kinds: `archaic`
(mined KJV/ASV↔BSB pairs, bidirectional), `suffix` (vocabulary-validated
inflection classes), `synonym` (theological pairs — present only once the
tracked DRAFT file `data/eval/theological_synonyms_v1.json` is APPROVED by the
advisor).

`bm25_stats` keys are unchanged: `n_docs`, `avgdl`, `vocab_size`,
`total_tokens`, `k1`, `b`. BM25 statistics are still scope-correct (the bundled
pack's stats are recomputed over the BSB+WSC subset, exactly as v1 did).

Key `meta` rows: `schema_version=2`, `format=search-pack-v2`, `scope`,
`bm25_tokenizer=nfkc-casefold-alnum-no-stemming`,
`posting_format=uvarint-gap-tf-v1`, `id_assignment`, `text_included` ("1"/"0"),
`n_chunks`. The bundled pack also carries the vector meta block (below).

### Posting blob encoding (`uvarint-gap-tf-v1`)

For each term, `postings` is `doc_freq` back-to-back pairs:

```
[ gap uvarint ][ tf uvarint ]  × doc_freq
```

- Integer chunk ids listed ASCENDING; `gap` = id minus previous id; the FIRST
  gap is the first id itself (ids ≥ 1, so every gap ≥ 1).
- `uvarint` = unsigned LEB128: 7 value bits per byte, low byte first, high bit
  set on every byte except the last.
- Decode until the blob is exhausted (its pair count always equals `doc_freq`).

Swift reader sketch:

```swift
var pos = 0, id = 0
while pos < blob.count {
    id += readUvarint(blob, &pos)      // gap -> absolute int chunk id
    let tf = readUvarint(blob, &pos)
    score(id, tf)
}
```

Scoring math (k1/b/avgdl/N, Lucene non-negative IDF, tokenizer) is UNCHANGED —
only where tf/df/dl live changed: `dl` now comes from `chunk.doc_len`
(join or preloaded array), `tf` from the blob.

## `ondemand_vectors.sqlite` (`meta.format = "vectors-pack-v2"`)

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE embedding (
    chunk_id INTEGER PRIMARY KEY,  -- SAME integer id space as the search pack
    vector   BLOB NOT NULL,        -- meta.embedding_dim int8 bytes (or float32-le)
    scale    REAL NOT NULL         -- per-vector dequant scale (1.0 for float32)
);
```

Key `meta` rows: `schema_version=2`, `format=vectors-pack-v2`,
`id_assignment`, `model_name`, `model_revision`, `model_combined_sha256`,
`embedding_dim` (384), `vector_format`, `query_instruction`, `n_vectors`.

### Vector quantization (`vector_format = "int8-symmetric-per-vector-le"`)

- `scale = max(|v|) / 127` (1.0 for an all-zero vector);
  `q_i = clip(rint(v_i / scale), -127, 127)` stored as signed int8.
- Scoring against the float32 query vector:
  `score = scale × Σ_i (q_i × query_i)` — no dequantized copy needed
  (`vDSP`/accumulate over int8 promoted to float). This is bit-identical to
  scoring the dequantized float32 vector.
- Measured retrieval-quality delta vs float32: `reports/pack_diet_v1.md` §4.
- `package fp32` produces `vector_format = "float32-le"` with the EXACT stored
  float32 bytes and `scale = 1.0` (the v1 fidelity escape hatch).

## App-side migration checklist (other lane)

1. Pack resolution: expect `ondemand_search.sqlite` + `ondemand_vectors.sqlite`
   instead of `ondemand_embeddings.sqlite`; `bundled_search.sqlite` keeps its
   name (detect v2 via `meta.format`).
2. `SearchStore`: read postings by decoding `bm25_term.postings`; doc length
   from `chunk.doc_len`; hits identified by INTEGER chunk id (hydration joins
   `chunk` by int id; `string_id` available for logging/provenance).
3. `DenseRetriever`: stream `embedding` (int8 + scale) from the vectors pack
   (or from `bundled_search.sqlite` when only bundled is present); the
   dense/BM25 join key is the integer chunk id.
4. `ChunkRepository`: unchanged semantics; the Scripture translation dedup key
   is still (book, chapter, verse_start, verse_end).
5. Never persist integer ids across corpus versions (see above).
6. Download tiering (F5): `ondemand_vectors.sqlite` is **default-tier**, not a
   separate opt-in. Read `corpus_manifest.json` → `packs.on_demand.files[]` and
   fetch every file whose `tier == "default"` on first launch (all on-demand
   packs today). `ondemand_search` + `ondemand_vectors` share
   `download_group == "retrieval-index"` and should be treated as one unit; the
   dense arm needs `ondemand_vectors`, BM25 needs only `ondemand_search`. If the
   vectors pack is missing, retrieval degrades **gracefully to BM25-only** — the
   search pack is self-sufficient and must never be gated on the vectors pack.
   `packs.on_demand.default_note` / `app_reader` in the manifest state this
   contract; `packs.on_demand.tiers.default.bytes` is the default-set size.
