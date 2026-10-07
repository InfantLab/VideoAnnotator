# Contract: CI on every pull request and push to master

| Job | Runners | Python | Blocks merge |
|---|---|---|---|
| `test` | ubuntu-latest, macos-latest | 3.12, 3.13 | yes |
| `test` | windows-latest | 3.12, 3.13 | no (existing allowance: SQLite file locking) |
| lint (ruff, format) and mypy | ubuntu-latest | 3.13 | yes |
| `viewer` | ubuntu-latest | n/a | yes |

- Each `test` job sets `UV_PYTHON` to its matrix version, so `.python-version` (3.13) doesn't
  override the 3.12 jobs.
- Each `test` job runs `pytest -m "not real_models and not performance"`. Wall-clock
  `performance` tests run in their own job; on shared runners they failed by timing alone
  (macOS 3.13: 100 parses in 1.9 s against a 1 s limit, 2026-10-01).
- Integration, performance and Docker jobs: unchanged, on the default (3.13).
