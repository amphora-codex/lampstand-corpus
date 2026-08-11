#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# upload-packs-r2.sh — push the on-demand corpus packs to the Cloudflare R2
# bucket the app's RemoteCorpusPackInstaller downloads from.
#
# One-time setup (architect, ~3 clicks + one login):
#   1. Cloudflare dashboard → R2 → Create bucket → name: lampstand-packs
#   2. Bucket → Settings → Public access: either
#        a. Custom domain  packs.thelampstand.app   (needs the zone on CF), or
#        b. r2.dev development URL (works instantly; swap the app's
#           RemoteCorpusPackInstaller.baseURL to it — one line).
#   3. `npx wrangler login` once on this machine (browser OAuth).
#
# Then:  scripts/upload-packs-r2.sh [packs-dir]
#   • Reads corpus_manifest.json for the corpus version + on-demand file list.
#   • Uploads each pack to  <bucket>/<corpus_version>/<filename>  so future
#     corpora coexist side by side and clients never see a mid-upgrade mix.
#   • Verifies each object landed by re-downloading its head and comparing
#     sizes; full checksum verification is scripts/verify-packs-remote.sh.
#
# Packs dir defaults to the app repo's staged copies (the exact bytes the
# checksums were computed from).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUCKET="lampstand-packs"
PACKS_DIR="${1:-$HOME/Lampstand/Resources/Corpus}"

command -v jq >/dev/null || { echo "error: jq required"; exit 1; }
command -v npx >/dev/null || { echo "error: node/npx required for wrangler"; exit 1; }

VERSION=$(jq -r '.corpus_version' "$ROOT/corpus_manifest.json")
SHIP_READY=$(jq -r '.ship_ready' "$ROOT/corpus_manifest.json")
[ "$SHIP_READY" = "true" ] || { echo "error: manifest ship_ready is not true — tag before uploading"; exit 1; }
echo "==> uploading $VERSION on-demand packs from $PACKS_DIR to r2://$BUCKET/$VERSION/"

jq -r '.packs.on_demand.files[] | "\(.name)\t\(.bytes)\t\(.sha256)"' "$ROOT/corpus_manifest.json" |
while IFS=$'\t' read -r name bytes sha; do
  local_file="$PACKS_DIR/$name"
  [ -f "$local_file" ] || { echo "error: $local_file missing"; exit 1; }
  actual_bytes=$(stat -f%z "$local_file")
  [ "$actual_bytes" = "$bytes" ] || { echo "error: $name size $actual_bytes != manifest $bytes"; exit 1; }
  actual_sha=$(shasum -a 256 "$local_file" | cut -d' ' -f1)
  [ "$actual_sha" = "$sha" ] || { echo "error: $name checksum mismatch vs manifest"; exit 1; }
  echo "  ↑ $name ($bytes bytes, $actual_sha)"
  npx wrangler r2 object put "$BUCKET/$VERSION/$name" --file "$local_file" --remote \
    --content-type application/octet-stream
done

echo "✅ upload complete. Now run scripts/verify-packs-remote.sh <public-base-url>"
echo "   e.g. scripts/verify-packs-remote.sh https://packs.thelampstand.app"
