#!/usr/bin/env bash
# Assembles the full Pages site (poster PDFs/previews, slide deck, index.html)
# into an output directory. Used by both CI and `serve_local.sh` so the two
# never drift apart.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IROS_DIR="$(dirname "$SCRIPTS_DIR")"
SITE_DIR="$IROS_DIR/site"
OUT_DIR="${1:-$SITE_DIR/build}"

(cd "$SCRIPTS_DIR" && uv run python export_poster.py)
(cd "$IROS_DIR/presentation" && uv run inkflow build)

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp "$SITE_DIR/index.html" "$OUT_DIR/"
cp "$IROS_DIR/poster/poster_portrait.pdf" "$OUT_DIR/"
cp "$IROS_DIR/poster/poster_landscape.pdf" "$OUT_DIR/"
cp "$IROS_DIR/assets/figures/poster_portrait_preview.png" "$OUT_DIR/"
cp "$IROS_DIR/assets/figures/poster_landscape_preview.png" "$OUT_DIR/"
cp -r "$IROS_DIR/assets" "$OUT_DIR/assets"
cp -r "$IROS_DIR/presentation/build" "$OUT_DIR/slides"

echo "Site assembled at $OUT_DIR"
