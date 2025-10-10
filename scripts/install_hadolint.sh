#!/usr/bin/env bash
# Install hadolint into the active Python virtualenv (if available) or system PATH.
# This script downloads the hadolint binary for Linux x86_64 and places it into the
# project's .venv/bin directory so pre-commit can find it when running hooks.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"

# Detect virtualenv bin dir
if [ -n "${VIRTUAL_ENV:-}" ]; then
  DEST_DIR="$VIRTUAL_ENV/bin"
else
  # Fallback to .venv in project root
  if [ -d "$ROOT_DIR/.venv" ]; then
    DEST_DIR="$ROOT_DIR/.venv/bin"
  else
    echo "[ERROR] No virtualenv detected and .venv not found; installing to /usr/local/bin requires sudo." >&2
    echo "Run this script inside your project's virtualenv or create a .venv via 'python -m venv .venv'" >&2
    exit 1
  fi
fi

mkdir -p "$DEST_DIR"

HADOLINT_VERSION="v2.12.0"
HADOLINT_URL="https://github.com/hadolint/hadolint/releases/download/${HADOLINT_VERSION}/hadolint-Linux-x86_64"

TMPFILE="$(mktemp)"
trap 'rm -f "$TMPFILE"' EXIT

echo "[START] Downloading hadolint ${HADOLINT_VERSION}..."
curl -L --fail -o "$TMPFILE" "$HADOLINT_URL"
chmod +x "$TMPFILE"

DEST_PATH="$DEST_DIR/hadolint"
mv "$TMPFILE" "$DEST_PATH"

echo "[OK] hadolint installed to $DEST_PATH"
echo "You can verify by running: $DEST_PATH --version"

exit 0
