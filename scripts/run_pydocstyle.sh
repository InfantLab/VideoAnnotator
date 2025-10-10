#!/usr/bin/env bash
# Run pydocstyle and summarize results into reports/
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
REPORTS_DIR="$ROOT/reports"
mkdir -p "$REPORTS_DIR"

RAW="$REPORTS_DIR/pydocstyle_raw.txt"
SUMMARY="$REPORTS_DIR/pydocstyle_summary.md"

echo "[START] Running pydocstyle across src/ and tests/"

if ! command -v pydocstyle >/dev/null 2>&1; then
  echo "[ERROR] pydocstyle not found in PATH. Install via 'uv add --dev pydocstyle' or 'uv run pip install pydocstyle'" >&2
  exit 2
fi

# Run pydocstyle and capture raw output
pydocstyle src tests > "$RAW" || true

cat > "$SUMMARY" <<'MD'
# pydocstyle summary

Ran pydocstyle on 'src' and 'tests'. Raw output saved to pydocstyle_raw.txt.

MD

if [ ! -s "$RAW" ]; then
  echo "[OK] No pydocstyle issues found." >> "$SUMMARY"
  echo "[OK] No pydocstyle issues found."
  exit 0
fi

echo "Parsing raw results to group by error code and top files..."

# Extract error codes (like D401) and count occurrences
awk -F: '{print $4}' "$RAW" | sed 's/^ *//' | awk '{print $1}' | sort | uniq -c | sort -rn > "$REPORTS_DIR/pydocstyle_by_code.txt"

# Top offending files
awk -F: '{print $1}' "$RAW" | sed 's/:$//' | sort | uniq -c | sort -rn > "$REPORTS_DIR/pydocstyle_by_file.txt"

cat >> "$SUMMARY" <<'MD'
## Top error codes (count)
```
MD
cat "$REPORTS_DIR/pydocstyle_by_code.txt" >> "$SUMMARY"
cat >> "$SUMMARY" <<'MD'
```

## Top files (count)
```
MD
head -n 30 "$REPORTS_DIR/pydocstyle_by_file.txt" >> "$SUMMARY"
cat >> "$SUMMARY" <<'MD'
```
MD

echo "[OK] pydocstyle summary written to $SUMMARY"
echo "[OK] Raw output: $RAW"

exit 0
