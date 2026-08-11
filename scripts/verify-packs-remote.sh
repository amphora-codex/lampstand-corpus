#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# verify-packs-remote.sh — download every on-demand pack from the public URL
# the app will use and verify byte size + SHA-256 against corpus_manifest.json.
# This is the end-to-end proof the transport serves EXACTLY the bytes the app's
# baked checksums will accept — run it once after upload, before TestFlight.
#
# Usage:  scripts/verify-packs-remote.sh https://packs.thelampstand.app
#         (or the bucket's r2.dev URL)
# Downloads ~610 MB to a temp dir; deletes it on success.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${1:?usage: verify-packs-remote.sh <public-base-url>}"
VERSION=$(jq -r '.corpus_version' "$ROOT/corpus_manifest.json")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fail=0
jq -r '.packs.on_demand.files[] | "\(.name)\t\(.bytes)\t\(.sha256)"' "$ROOT/corpus_manifest.json" |
while IFS=$'\t' read -r name bytes sha; do
  url="$BASE/$VERSION/$name"
  printf "  ↓ %-32s" "$name"
  curl -fsSL --retry 2 -o "$TMP/$name" "$url" || { echo "DOWNLOAD FAILED ($url)"; exit 1; }
  actual_bytes=$(stat -f%z "$TMP/$name")
  actual_sha=$(shasum -a 256 "$TMP/$name" | cut -d' ' -f1)
  if [ "$actual_bytes" = "$bytes" ] && [ "$actual_sha" = "$sha" ]; then
    echo "OK ($bytes bytes)"
  else
    echo "MISMATCH (size $actual_bytes vs $bytes, sha ${actual_sha:0:12}… vs ${sha:0:12}…)"
    exit 1
  fi
  rm -f "$TMP/$name"
done

echo "✅ every pack served at $BASE/$VERSION matches the manifest — the app will accept them."
