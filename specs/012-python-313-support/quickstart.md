# Quickstart: verifying Python 3.13 support

Manual checks for the parts CI can't cover. Run from the repository root.

## 1. Install on 3.13 (User Story 1)

```bash
uv sync --python 3.13 --all-extras          # or just `uv sync --all-extras`: .python-version says 3.13
uv run python --version                     # Python 3.13.x
uv run videoannotator pipelines list        # every installed pipeline: available
```

## 2. Run every pipeline on a real video, and compare with 3.12 (SC-003)

`speaker_diarization` needs `HUGGINGFACE_TOKEN` (in `.env` is fine) for an account that accepted
the pyannote model licences.

```bash
VIDEO=viewer/demo-assets/2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4
PIPES=speech_recognition,speaker_diarization,scene_detection,person_tracking,face_analysis,face_openface3_embedding

# 3.13
uv run python scripts/compare_pipeline_outputs.py dump "$VIDEO" --pipelines "$PIPES" -o /tmp/out313.json

# 3.12, in a separate environment
UV_PROJECT_ENVIRONMENT=.venv-312 uv sync --python 3.12 --all-extras
UV_PROJECT_ENVIRONMENT=.venv-312 uv run python scripts/compare_pipeline_outputs.py dump "$VIDEO" --pipelines "$PIPES" -o /tmp/out312.json

uv run python scripts/compare_pipeline_outputs.py compare /tmp/out312.json /tmp/out313.json
```

Expected: every pipeline `completed` in both; `compare` reports no differences apart from run
identifiers and timestamps. To measure run-to-run variation (the spec's allowance), dump twice on
3.12 and compare those first.

## 3. Unsupported Python (SC-004)

```bash
uv sync --python 3.14         # refuses: incompatible with the project's Python requirement
uv venv /tmp/v314 --python 3.14 && uv pip install --python /tmp/v314/bin/python --no-deps .
/tmp/v314/bin/videoannotator version     # runs, with one warning naming 3.12 and 3.13
```

## 4. Containers (User Story 4)

```bash
docker build -f Dockerfile.cpu -t va:cpu-313 .
docker run --rm va:cpu-313 uv run python --version            # 3.13.x
```

Dev container: rebuild, then `uv run python --version` (3.13.x) and `uv run pytest -q -m "not real_models"`.

## 5. CI (User Story 3)

Open the pull request's checks: `test` appears for each of ubuntu / macOS / Windows × 3.12 / 3.13.
