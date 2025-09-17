# AGENTS.md

Unified guidance for AI coding assistants (Copilot, Claude, others) collaborating on the VideoAnnotator project.

## 1. Project Snapshot
- **Name**: VideoAnnotator
- **Current Release**: v1.2.0 (API-first, production-ready base)
- **Active Minor (In Progress)**: v1.2.1 (polish + pipeline specification/registry foundation)
- **Next Major (Planned)**: v1.3.0 (advanced ML, multi-modal, plugins, enterprise)
- **Primary Users**: Individual researchers (90%) needing simple local installs.
- **Secondary Users**: Lab / enterprise research groups (10%).
- **Core Pillars**: Modular pipelines, standardized outputs, reproducibility, minimal friction.

## 2. Primary Assistant Objectives
AI agents MUST optimize for:
1. Low-friction local usability (no mandatory external services).
2. Predictable, non-breaking incremental improvements (especially in v1.2.x series).
3. Standard formats first (COCO, WebVTT, RTTM) – no bespoke schema proliferation.
4. Future extensibility (registry metadata now → adaptive routing/plugins later).
5. Clear, ASCII-safe console / log output (Windows encoding constraint: NO emoji, no fancy unicode blocks that may fail in PowerShell).

## 3. Output & Logging Rules
| Concern | Rule | Rationale |
| ------- | ---- | --------- |
| Emoji / Unicode | Prohibited in console/log prints | Windows charmap crashes  |
| Log Directory | Use `setup_videoannotator_logging()`; do not reinvent handlers | Central consistency |
| Structured Logs | Prefer logger over `print`; CLI may still use `typer.echo` for UX | Unified diagnostics |
| Error Prefix | Use `[ERROR]`, `[WARNING]`, `[OK]`, `[START]` | Consistency + plain ASCII |
| JSON Output | Provide optional `--json` for machine parsing (v1.2.1 introduction) | Scripting support |

## 4. Pipeline Registry (v1.2.1 Scope)
Minimal YAML-driven registry located at `src/registry/metadata/` (to be added). Each file MUST define:
```
name: string (slug)
display_name: string
category: tracking|detection|analysis|segmentation|preprocessing
description: short sentence
outputs: 
  - format: COCO|RTTM|WebVTT|JSON
    types: [list of semantic output types]
config_schema: { field_name: { type, default, description } }
examples: optional list (cli / api usage)
version: integer schema version
```
Assistant tasks involving pipelines should:
- Load via registry once implemented rather than hard-coding.
- Offer graceful degradation if metadata missing (warn, skip).
- Avoid adding advanced fields early (resources, capabilities) unless explicitly approved (those are v1.3.0 targets).

## 5. API & CLI Conventions
| Aspect | Convention |
| ------ | ---------- |
| CLI Base Command | `videoannotator` (Typer app) |
| Job Submission | `videoannotator job submit` |
| Pipelines Listing | `videoannotator pipelines --detailed` (backed by registry) |
| Config Validation | `videoannotator config --validate path/to.yaml` |
| Health Check | `/health` (enriched incrementally) |
| Pipeline Endpoint | `/api/v1/pipelines` dynamic (replace mock) |

When adding new CLI commands:
- Provide `--json` if the output lists entities.
- Fail fast with exit codes; include a suggestion line for common errors.

## 6. Versioning & Consistency
- Single version source in `src/version.py` (no duplicated literals).
- Registry metadata includes its own `version` (schema evolution).
- Generated docs (pipeline specs) MUST be reproducible – diffs trigger CI failure.

## 7. Testing Expectations
| Tier | Path | Goal | Runtime |
| ---- | ---- | ---- | ------- |
| Unit | `tests/unit/` | Fast logic validation | < 1 min total |
| Integration | `tests/integration/` | Cross-component behaviors | ~5 min |
| Pipelines | `tests/pipelines/` | Real or simulated model runs | Longer |

Assistant-created code MUST include at least minimal tests for:
- Registry loader integrity.
- API pipeline endpoint parity with registry.
- Markdown generation (smoke: file contains each pipeline name).

## 8. Deferred (Do NOT Implement in v1.2.1 Without Explicit Approval)
- Plugin execution sandbox
- Multi-modal correlation graph engine
- Active learning feedback store
- Real-time streaming/WebRTC stack
- RBAC / SSO / multi-tenancy / audit log frameworks
- Complex resource scheduling or autoscaling logic
- Rich TUI or progress bar framework

## 9. Performance & Resource Principles
- Favor lazy initialization; avoid loading heavy models outside `initialize()`.
- Avoid hidden network calls during import time.
- Annotate any model download operations with explicit logging.
- Provide timing logs (execution time context manager available in `logging_config`).

## 10. Security & Data Handling
- Never log raw PII (e.g., full file system paths of user home beyond what’s already necessary).
- Validate file existence before processing.
- Reject unsupported video formats with clear error messages.
- Avoid embedding secrets or tokens in code or docs.

## 11. Contribution Style for Agents
| Action | Required Practice |
| ------ | ----------------- |
| File Edits | Small, atomic patches with rationale | 
| New Features | Add tests + docs stub in same PR scope |
| Refactors | Preserve public CLI/API unless release notes updated |
| Docs Generation | Provide script, never hand-edit generated file |
| Logging Additions | Use existing logger categories (`api`, `pipelines`, `requests`, `database`) |

## 12. Error & Validation Strategy
Standard error envelope (foundation for future):
```
{
  "error": {
    "code": "PIPELINE_NOT_FOUND",
    "message": "Pipeline 'xyz' not registered",
    "hint": "Run 'videoannotator pipelines --detailed' to list available pipelines"
  }
}
```
Assistant changes touching API endpoints should migrate ad hoc responses toward this structure (without breaking existing clients unless major release).

## 13. Windows Compatibility Checklist
Before approving output that touches user-visible text:
- [ ] No emojis or fancy unicode
- [ ] Paths use `Path` abstractions not hard-coded separators
- [ ] Avoid color control chars (future optional feature)

## 14. Roadmap Alignment Matrix
| Category | v1.2.1 Expectation | v1.3.0 Expansion |
| -------- | ------------------ | ---------------- |
| Registry | Minimal YAML + loader | Capability & resource aware, plugin aware |
| Config Validation | Presence & primitive types | Full schema + templates + precedence engine |
| Health | Basic enrichment (registry, GPU flag) | Deep performance & scaling metrics |
| Errors | Simple envelope | Taxonomy + GraphQL alignment |
| Docs | Generated pipeline specs | Marketplace + plugin publishing |
| CLI | JSON option & consistent formatting | Color/TUI, adaptive suggestions |

## 15. Assistant Decision Heuristics
1. Is the change needed for v1.2.1 scope? If not, defer.
2. Can it be undone easily if incorrect? If not, request human confirmation.
3. Does it introduce silent behavior change? If yes, add explicit note to changelog.
4. Is there a testable seam? Add or extend a test BEFORE large refactor.

## 16. Typical Workflows for Agents
### Add a New Pipeline (v1.2.1)
1. Implement class in `src/pipelines/<name>/`
2. Provide metadata YAML
3. Update tests (registry presence)
4. Regenerate docs (script) → commit

### Extend Registry Field (v1.3.0-forward)
1. Propose field (resources/capabilities) → doc it in metadata README
2. Add parsing with backward-compatible default
3. Update one metadata file as exemplar
4. Add validation + tests

## 17. Common Anti-Patterns to Avoid
- Adding model downloads in global import scope
- Introducing new colorized/emoji output prematurely
- Creating parallel config systems instead of extending registry
- Hard-coding pipeline lists in multiple places
- Adding advanced multi-modal logic in v1.2.1 timeframe

## 18. Quick Reference Checklist (Pre-Commit)
- [ ] No emoji / unicode hazards
- [ ] Registry unaffected OR metadata updated
- [ ] Tests added/updated (if feature code)
- [ ] Docs regenerated (if pipeline changes)
- [ ] Logging consistent & non-duplicative
- [ ] Version not duplicated outside `version.py`

## 19. Escalation & Deferral Tags (Suggest in PR Titles)
Use labels/phrases to keep scope controlled:
- `scope:v1.2.1-foundation`
- `defer:v1.3.0`
- `needs:registry-extension`
- `type:docs-generation`
- `type:cli-ux`

## 20. Azure-Specific Guidance (If Applicable)
If generating Azure deployment or infra code:

## 21. Package & Environment Management (uv Mandatory)
The project STANDARD is to use `uv` (https://github.com/astral-sh/uv) for all Python dependency resolution, execution, and testing. Assistants MUST NOT rely on a pre-activated conda / venv environment when giving commands or running tests.

Required practices:
1. Installation / Sync
  - To sync the environment exactly to `pyproject.toml` + `uv.lock`:
    - `uv sync` (never manually `pip install` inside the venv for project deps)
2. Adding / Removing Dependencies
  - Add runtime dependency: `uv add <package>`
  - Add dev/test dependency: `uv add --dev <package>`
  - Remove: `uv remove <package>`
  - After changes: commit updated `uv.lock`.
3. Running Code
  - Module / script: `uv run python -m videoannotator ...`
  - CLI entry (if defined): `uv run videoannotator ...`
4. Tests & Quality
  - Full test suite: `uv run pytest -q` (or with markers/coverage flags)
  - Ruff lint (if configured): `uv run ruff check .`
  - Mypy (if configured): `uv run mypy src/`
5. REPL / One-off
  - `uv run python` (never rely on external shell’s site-packages)
6. Caching & Performance
  - Let `uv` manage wheels; do not manually clear caches unless diagnosing a build issue.
7. Consistency Enforcement
  - If guidance or scripts previously referenced bare `pytest` / `python`, update them to `uv run` form.
  - Any test failure reproduction steps in issues/PR descriptions MUST use `uv run`.
8. CI Alignment
  - CI should mirror: `uv sync` then `uv run pytest ...`. Do not introduce alternative installers.

Common Anti-Patterns to Avoid (uv):
- Using `pip install -e .` directly (use `uv sync` instead).
- Activating a conda environment and running `pytest` without `uv run`.
- Editing `requirements.txt` (the project is pyproject+lock driven; no requirements.txt unless generated artifact).

Assistant Heuristic Update:
- Before running tests, always ensure commands are prefixed with `uv run` unless explicitly operating inside a scripted `make` target that already does so.
- If a user asks for install instructions, default to `uv` commands.

Quick Reference:
```
uv sync                     # create/update environment
uv run pytest -q            # run tests
uv add requests             # add runtime dep
uv add --dev pytest-cov     # add dev dep
uv run videoannotator --help
```

If any discrepancy between an active shell environment and `uv run` behavior is suspected, treat the `uv run` result as canonical.


