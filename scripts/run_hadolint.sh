#!/usr/bin/env bash
# Run hadolint across Dockerfiles and produce a human-friendly report.
# This script returns exit code 0 for local convenience but writes the raw exit
# code into .hadolint_last_exit so CI or callers can inspect it if needed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"

HADOLINT_BIN="${HADOLINT_BIN:-${VIRTUAL_ENV:-$ROOT_DIR/.venv}/bin/hadolint}"
if [ ! -x "$HADOLINT_BIN" ]; then
  echo "[ERROR] hadolint not found at $HADOLINT_BIN" >&2
  echo "Run: source .venv/bin/activate && bash scripts/install_hadolint.sh" >&2
  exit 2
fi

echo "[START] Running hadolint via $HADOLINT_BIN"

FILES=(Dockerfile* Dockerfile.* docker/* Dockerfile)
FOUND=()
for f in "${FILES[@]}"; do
  for match in $ROOT_DIR/$f; do
    if [ -f "$match" ]; then
      FOUND+=("$match")
    fi
  done
done

if [ ${#FOUND[@]} -eq 0 ]; then
  echo "[OK] No Dockerfiles found to lint."
  echo 0 > "$ROOT_DIR/.hadolint_last_exit"
  exit 0
fi

set +e
"$HADOLINT_BIN" --ignore DL3008 --ignore DL3009 "${FOUND[@]}"
RET=$?
set -e

echo "$RET" > "$ROOT_DIR/.hadolint_last_exit"

if [ $RET -eq 0 ]; then
  echo "[OK] hadolint passed"
else
  echo "[WARN] hadolint returned $RET — see output above for details"
fi

# Return 0 for local convenience; CI should inspect .hadolint_last_exit if needed.
exit 0
