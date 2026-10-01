# Research: Core Dependency Clean-up and Tooling

## R1. Baseline core install (SC-001)

Core only (`uv sync --frozen --no-dev --no-install-project`, Python 3.13, before this spec):
**73 packages, 746 MB.** Largest: llvmlite 128 MB + numba 17 MB (audio-only), scipy 114 MB and
networkx 8 MB (via scikit-image, unused), av 114 MB (unused), imageio-ffmpeg 77 MB (unused),
matplotlib + fontTools 43 MB (unused), pandas 44 MB (see R2), cryptography 13 MB (unused).

## R2. What is really unused

Checked `src/`, `scripts/`, `examples/`, `tests/` and the docs for imports, plus runtime-by-name
needs:

| Package | Finding | Decision |
|---|---|---|
| moviepy | only `visualization/video_utils.py`, via `moviepy.editor` (removed in moviepy 2) | drop with the dead module |
| matplotlib, tqdm | only listed in `version.py`'s dependency report | drop; report must tolerate absence |
| openpyxl, pandas | only `main.py` (`read_excel`, DataFrames) | drop with the dead module (R3) |
| imageio, imageio-ffmpeg, av, alembic, rich, scikit-image, cryptography | imported nowhere | drop |
| click | not imported; typer depends on it | drop the direct declaration |
| python-multipart | not imported; FastAPI requires it for form/file uploads | **keep** |
| numba | only `whisper_base_pipeline.py` (audio) | move to `audio` |
| imutils (`face`), supervision (`person`) | imported nowhere | drop |

## R3. Dead modules

- `src/videoannotator/visualization/`: imports `src.visualization.render` (pre-src-layout path),
  IPython (not a dependency), `moviepy.editor` (gone), and ultralytics at module level. Not
  importable; nothing imports it. Remove.
- `src/videoannotator/main.py`: "legacy batch processing entry point", imports `src.config`,
  `src.processors...`, `src.utils...` (none exist). Not importable; nothing imports it. Remove.
- `exporters/__init__.py` and `native_formats.py` mention `src.exporters` only in docstring
  examples: fix the examples to `videoannotator.exporters`.

## R4. Upgrading core

- **Decision**: `uv lock --upgrade-package` for each remaining core dependency and the dev tools
  (not torch, pyannote, transformers, opencv, pipeline libraries: later specs), then the full
  suite on 3.12 and 3.13 and the output comparison. Majors in scope: SQLAlchemy 2.0 → 2.1, mypy 1
  → 2, pytest-cov 6 → 7. A major that fails gets an upper bound with the reason beside it.
- pandas 3 is no longer a core question once pandas leaves core (it arrives only through extras
  that need it).

## R5. Tooling

- **mypy hook**: replace `mirrors-mypy` (isolated env, mypy 1.5.1, excludes pipelines/, storage/,
  utils/, version.py, exporters/, schemas/) with a `local` hook running `uv run mypy
  src/videoannotator` (`pass_filenames: false`): exactly CI's command, version and config.
- **Drop**: `pydocstyle` (its `files: ^src/api/` matches nothing since the src-layout move; ruff's
  `D` rules are the maintained replacement if docstring linting is wanted later) and
  `mirrors-prettier` (no stable release since v3; manual stage only).
- **Upgrade**: pre-commit-hooks v6.0.0, ruff-pre-commit to match the dev `ruff`, bandit 1.9.4,
  shellcheck-py v0.11.0.1, hadolint v2.15.1, commitizen v4.19.0, safety hook v1.4.2.
- **Dev tools once**: `dev` exists both as an extra and as `[dependency-groups].dev`. Keep the
  dependency group (uv installs it by default; CI uses `--dev`); remove the `dev` extra and its
  mention in `all`.
- **Actions**: checkout v4 → v7, setup-python v5 → v7, setup-uv v5 → v10, upload-artifact v4 → v7,
  codecov-action v4 → v7, docker actions to current majors; pin `openjournals-draft-action` to a
  commit instead of `@master`.

## R6. Findings during implementation

- **Hidden dependencies of `face-openface3`**: with `scikit-image` and `pandas` gone from core,
  OpenFace 3 failed ("No module named 'skimage'"). Installing each extra alone in a clean
  environment showed `face-openface3` had never worked on its own: `openface-test` imports torch,
  torchvision, timm, scikit-image, pandas, huggingface-hub, tqdm, matplotlib, seaborn and
  tensorboardX (the last three at module level in its STAR landmark detector) without declaring
  them. All now declared in the extra. Every other extra (`face`, `audio`, `scene`, `person`)
  runs its pipelines with only core plus itself; `llm` needs a running Ollama;
  `face-laion` / `audio-laion` timed out downloading their models (dropped in spec 4).
- **Outputs before/after** (demo video, full install, Python 3.13): `speech_recognition`,
  `speaker_diarization`, `scene_detection`, `person_tracking`, `face_analysis` identical.
  `face_openface3_embedding` within run-to-run variation (landmarks ≤ 0.075 px, yaw ≤ 0.018, AU
  intensity ≤ 0.0073).
- **OpenFace's run-to-run sensitivity**: a 1-pixel change in a RetinaFace box between runs moves
  that face's gaze and AU outputs by up to ~2.4 (seen comparing a core+`face-openface3`
  environment with the full one; the same probe gives the same AU01 in both). Not caused by this
  spec; roadmap item for a deterministic mode.
- **mypy coverage**: `exclude` and seven `ignore_errors` overrides kept the pipelines, storage,
  utils, exporters, schemas and `version.py` out of the type check. With them removed, five errors
  (two `None` vs `list[int]` assignments, three stale `type: ignore`); fixed, and mypy now checks
  117 files instead of 86.
