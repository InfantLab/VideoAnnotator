#!/usr/bin/env bash
# Install hadolint into the active Python virtualenv (if available) or system PATH.
# This script downloads the hadolint binary for Linux x86_64 and places it into the
# project's .venv/bin directory so pre-commit can find it when running hooks.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"

# Allow callers (e.g., container builds) to override the destination.
# Example: HADOLINT_DEST_DIR=/usr/local/bin bash scripts/install_hadolint.sh
if [ -n "${HADOLINT_DEST_DIR:-}" ]; then
  DEST_DIR="$HADOLINT_DEST_DIR"
else
  # Detect virtualenv bin dir
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    DEST_DIR="$VIRTUAL_ENV/bin"
  else
    # Fallback to .venv in project root
    if [ -d "$ROOT_DIR/.venv" ]; then
      DEST_DIR="$ROOT_DIR/.venv/bin"
    else
      echo "[ERROR] No virtualenv detected and .venv not found." >&2
      echo "Run inside your project's virtualenv, create a .venv via 'uv sync'," >&2
      echo "or set HADOLINT_DEST_DIR to install elsewhere (e.g., /usr/local/bin)." >&2
      exit 1
    fi
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
