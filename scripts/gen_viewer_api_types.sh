#!/usr/bin/env bash
# Regenerate viewer/src/api/schema.d.ts from the server's own OpenAPI schema.
# Run after changing an API route or response model; commit the result.
set -euo pipefail
cd "$(dirname "$0")/.."

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

# Written by Python, not redirected: importing the app logs to stdout.
${PYTHON:-uv run python} -c '
import json, sys
from videoannotator.api.main import app
with open(sys.argv[1], "w") as f:
    json.dump(app.openapi(), f)
' "$tmp_dir/openapi.json" > /dev/null

cd viewer
bunx openapi-typescript "$tmp_dir/openapi.json" -o src/api/schema.d.ts
