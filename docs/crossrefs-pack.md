# TSK cross-reference pack — `bundled_crossrefs.sqlite` contract (app-side lane)

Authoritative schema + encoding contract for the bundled TSK cross-reference
pack produced by `python -m lampstand_corpus.cli package`. The app lane builds
the reading-panel UI and the `hybridContext` graph boost against THIS document.
Measured size + retrieval deltas: `reports/crossref_pack_v1.md`. Shares the
pack-diet conventions (`docs/pack-diet.md`): uvarint = unsigned LEB128; integer
chunk ids are the search pack's ids (per-corpus-version, never persisted
across corpus updates).

## Pack home

A separate tiny BUNDLED pack (always ships in the binary). The full-fidelity
`ondemand_crossrefs.sqlite` (v1 row-per-edge schema, 24.2 MB) is unchanged and
still ships on-demand; the app's crossref reader needs only this bundled pack.

Data source: OpenBible.info TSK (**CC-BY** — the required attribution string is
in `meta.attribution` and must be rendered in the app's acknowledgements).
Only RESOLVING edges are packed (`src_resolves=1 AND tgt_resolves=1`;
344,794 of 344,799 — the 5 non-resolving refs stay in the build DB + flags).
Signed community votes (−86..+1278) are preserved verbatim.

## Tables

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE crossref (
    src_verse INTEGER PRIMARY KEY,  -- verse key, see below
    n_targets INTEGER NOT NULL,
    targets   BLOB NOT NULL         -- vote-rank order (OpenBible file order)
);
CREATE TABLE chunk_crossref (
    chunk_id    INTEGER PRIMARY KEY, -- pack-diet int chunk id (search pack)
    n_neighbors INTEGER NOT NULL,
    neighbors   BLOB NOT NULL        -- aggregated-weight desc
);
```

Key `meta` rows: `format=crossrefs-pack-v1`, `verse_key`, `target_format`,
`neighbor_format`, `id_assignment`, `expansion_top_n` (8),
`expansion_neighbor_scope` (`bsb pericope chunks`), `n_sources`, `n_edges`,
`n_expanded_chunks`, `license`, `attribution`.

## Verse key (no lookup table)

```
verse_key = book_ord * 1_000_000 + chapter * 1_000 + verse
```

`book_ord` is the 0-based index in the canonical 66-book order (the app's
`BookId` order == `books.ORDER`). Chapters ≤ 150 and verses ≤ 176 on the KJV
spine, so components never collide; keys are monotone in canonical Scripture
order. Decode: `book_ord = key / 1e6`, `chapter = key % 1e6 / 1e3`,
`verse = key % 1e3`.

## `crossref.targets` blob (`target_format`)

`n_targets` back-to-back triples, in the OpenBible per-source vote-rank order:

```
[ start_key uvarint ][ (end_key − start_key) uvarint ][ zigzag(votes) uvarint ]
```

- Single-verse targets have delta 0. Ranges may cross chapter and (18 cases)
  book boundaries — decode both endpoints via the verse-key arithmetic.
- `zigzag(n)`: 0,−1,1,−2 → 0,1,2,3 (`n<0 ? 2*(-n)-1 : 2*n`); negative votes
  are community-downvoted refs — the UI may de-emphasize or hide them, but
  they are present verbatim.

Swift reader sketch:

```swift
var pos = 0
for _ in 0..<nTargets {
    let start = readUvarint(blob, &pos)
    let end   = start + readUvarint(blob, &pos)
    let votes = zigzagDecode(readUvarint(blob, &pos))
    emit(start, end, votes)   // already vote-ranked; no sort needed
}
```

## `chunk_crossref.neighbors` blob (`neighbor_format`)

For each Scripture retrieval chunk (EVERY translation has its own row, so any
hit's chunk id resolves), the top-8 TSK-adjacent pericopes:

```
[ neighbor_chunk_id uvarint ][ weight uvarint ]   × n_neighbors
```

- Neighbors are **BSB pericope chunks** (int ids from the search pack); the
  app's translation dedup treats them as the same row as their KJV/ASV/WEB
  twins, so the rows work against both the bundled and full indexes.
- `weight` = the SUM of signed votes of all TSK edges from the source chunk's
  verse window landing in that neighbor pericope; self-range excluded;
  net-non-positive targets dropped (weight ≥ 1 always). Order: weight desc,
  ties by neighbor's string chunk id asc.
- `expansion_top_n = 8`: with the boost drawing from the top ~5 Scripture hits
  the injected candidate list stays ≤ 40 — within one retriever's fusion
  budget next to `bm25PerType=20` / `denseDepth=20`.

## Suggested `hybridContext` boost shape (measured by the corpus harness)

The corpus-side measurement (see `reports/crossref_pack_v1.md` §2) fused the
graph candidates of the top-5 Scripture hits as a third RRF list (rank order:
source hit rank, then pack weight; candidates already ranked by verse-range or
excluded are skipped). `hybrid-graph` = equal retriever weight;
`hybrid-graph-weak` = ⅓ weight. The app lane owns the final shape; re-measure
any deviation with `validate-retrieval` before shipping constants.
