# Contract: CI on every pull request and push to master

| Job | Runners | Python | Blocks merge |
|---|---|---|---|
| `test` | ubuntu-latest, macos-latest | 3.12, 3.13 | yes |
| `test` | windows-latest | 3.12, 3.13 | no (existing allowance: SQLite file locking) |
| lint (ruff, format) and mypy | ubuntu-latest | 3.13 | yes |
| `viewer` | ubuntu-latest | n/a | yes |

- Each `test` job sets `UV_PYTHON` to its matrix version, so `.python-version` (3.13) doesn't
  override the 3.12 jobs.
- Each `test` job runs `pytest -m "not real_models"`, as today.
- Integration, performance and Docker jobs: unchanged, on the default (3.13).
