#!/usr/bin/env bash
# Assembles the full Pages site (poster PDFs/previews, slide deck, index.html)
# into an output directory. Used by both CI and `serve_local.sh` so the two
# never drift apart.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IROS_DIR="$(dirname "$SCRIPTS_DIR")"
REPO_ROOT="$(dirname "$IROS_DIR")"
SITE_DIR="$IROS_DIR/site"
OUT_DIR="${1:-$SITE_DIR/build}"

(cd "$SCRIPTS_DIR" && uv run python export_poster.py)
rm -rf "$IROS_DIR/presentation/build"
(cd "$IROS_DIR/presentation" && uv run inkflow build)

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp "$SITE_DIR/index.html" "$OUT_DIR/"
cp "$IROS_DIR/poster/poster_portrait.pdf" "$OUT_DIR/"
cp "$IROS_DIR/poster/poster_landscape.pdf" "$OUT_DIR/"
cp "$IROS_DIR/assets/figures/poster_portrait_preview.png" "$OUT_DIR/"
cp "$IROS_DIR/assets/figures/poster_landscape_preview.png" "$OUT_DIR/"

SITE_ASSETS=(
    logos/email.svg
    logos/iros-2026-logo.png
    logos/kit-logo.svg
    logos/linkedin.svg
    photos/marlon.jpg
    photos/nick.jpg
)
for asset in "${SITE_ASSETS[@]}"; do
    mkdir -p "$OUT_DIR/assets/$(dirname "$asset")"
    cp "$IROS_DIR/assets/$asset" "$OUT_DIR/assets/$asset"
done

mkdir -p "$OUT_DIR/assets/logos"
cp "$REPO_ROOT/assets/mosaic.png" "$OUT_DIR/assets/logos/mosaic-logo.png"

cp -r "$IROS_DIR/presentation/build" "$OUT_DIR/slides"

# Make sure all assets are present in the output directory
missing=0
while read -r asset; do
    [[ -e "$OUT_DIR/$asset" ]] || {
        echo "index.html references missing $asset" >&2
        missing=1
    }
done < <(grep -o 'assets/[A-Za-z0-9._/-]*' "$SITE_DIR/index.html" | sort -u)
[[ $missing -eq 0 ]] || exit 1

echo "Site assembled at $OUT_DIR"
