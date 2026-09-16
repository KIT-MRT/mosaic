#!/usr/bin/env bash
# Builds the full Pages site and serves it locally, so `index.html` shows the
# same poster/slides links it will have once deployed instead of 404s.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(dirname "$SCRIPTS_DIR")/site/build"

"$SCRIPTS_DIR/build_site.sh" "$OUT_DIR"

PORT="${1:-8000}"
echo "Serving at http://localhost:$PORT"
python3 -m http.server "$PORT" --directory "$OUT_DIR"
