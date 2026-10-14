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

## Results (2026-10-01)

Linux x86_64, RTX 4060 Laptop GPU, demo video `2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4`;
3.12.3 (system Python) and 3.13.15 (uv-managed), same lock.

| Pipeline | Annotations | 3.12 run vs 3.12 run | 3.12 vs 3.13 |
|---|---|---|---|
| speech_recognition | 1 transcript ("Ready, baby girl? Good morning, girl. …") | identical | identical |
| speaker_diarization | 4 segments | identical | identical |
| scene_detection | 1 scene | identical | identical |
| person_tracking | 22 | identical | identical |
| face_analysis | 0 (finds no faces in this video on either version) | identical | identical |
| face_openface3_embedding | 1 record | differs: landmarks ≤ 0.095 px, yaw ≤ 0.017, AU intensity ≤ 0.0064 | differs **less**: ≤ 0.040 px, ≤ 0.017, ≤ 0.0045 |

OpenFace's differences are GPU run-to-run nondeterminism, present on 3.12 alone; 3.13 stays
within it (SC-003). Test suite on 3.13: 1335 passed, 0 failed (CI covers both versions).

Found and fixed on the way: a Triton cache shared between Python versions made
`speech_recognition` return nothing on 3.12 after a 3.13 run (research R10).

Dev container: the first rebuild after this change recreates `.venv` on 3.13 (its post-create
`uv sync` follows `.python-version`); model files in `models/` are untouched.

Not run here: Docker image builds (no Docker in the dev container); CI's `docker-build` job and
the maintainer cover them.
