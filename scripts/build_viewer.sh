#!/usr/bin/env bash
# Build the viewer (viewer/) into the bundle the server serves at /viewer
# (src/videoannotator/viewer_static/).
#
#   bash scripts/build_viewer.sh          # build, then replace the bundle
#   bash scripts/build_viewer.sh --check  # build, then fail if the committed bundle differs
#
# The bundle is replaced wholesale: copying over it left files from older builds
# behind. --check is what CI runs, so viewer changes can't ship with a stale bundle.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEWER_DIR="$ROOT_DIR/viewer"
BUNDLE_DIR="$ROOT_DIR/src/videoannotator/viewer_static"

check=false
case "${1:-}" in
  "") ;;
  --check) check=true ;;
  *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

if ! command -v bun >/dev/null 2>&1; then
  echo "bun not found: install it from https://bun.sh (the dev container has it)" >&2
  exit 1
fi

cd "$VIEWER_DIR"
bun install --frozen-lockfile
bun run build:embedded

if $check; then
  # Line endings are ignored: a Windows checkout's CRLF copy serves the same page.
  if ! diff -r --strip-trailing-cr "$VIEWER_DIR/dist" "$BUNDLE_DIR" >/dev/null; then
    diff -rq --strip-trailing-cr "$VIEWER_DIR/dist" "$BUNDLE_DIR" >&2 || true
    echo >&2
    echo "The committed viewer bundle doesn't match viewer/." >&2
    echo "Run 'bash scripts/build_viewer.sh' and commit src/videoannotator/viewer_static/." >&2
    exit 1
  fi
  echo "Viewer bundle is up to date."
else
  rm -rf "$BUNDLE_DIR"
  cp -r "$VIEWER_DIR/dist" "$BUNDLE_DIR"
  echo "Updated $BUNDLE_DIR from viewer/."
fi
