# sources/

Versioned snapshots of the **public-domain** texts the pipeline ingests, plus a provenance manifest (URL · retrieval date · SHA-256) for each.

**Policy**
- Public-domain / open-licensed texts only. No restricted or modern-licensed content is ever placed here.
- Each snapshot is recorded with its source URL, fetch date, and SHA-256 — the checksum is the unit of reproducibility.
- Builds read from these snapshots, never from live URLs.

> Whether the raw snapshot bytes are committed here (vs. a manifest-only + fetch-on-build approach, possibly via git-lfs for the larger texts) is a setup decision flagged for the architect at P1. CLAUDE.md's pipeline rule 1 leans toward committing snapshots for full reproducibility.
