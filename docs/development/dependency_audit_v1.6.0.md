# Dependency audit for v1.6.0

**Status**: draft, 2026-10-01. Part of v1.6.0 Phase 1 ([`roadmap_v1.6.0.md`](roadmap_v1.6.0.md)).
Scope follows the [pipeline review](pipeline_review_v1.6.0.md): where a recommendation depends on
a review decision still open, it says so. Each **Recommendation** is a proposal; the specs decide.

Versions: *locked* is `uv.lock` on `1.6-dev` (2026-09-30); *latest* is PyPI on 2026-09-30.

## 1. Summary

**Update (spec 015, 2026-10-01)**: torch moved to **2.11**, not 2.14: pyannote.audio 4 imports
torchaudio, discontinued at 2.11 and compiled against torch 2.11. torch ≥ 2.12 waits for
pyannote.audio to drop torchaudio. pyannote.audio 4 also turned out to need the
`speaker-diarization-community-1` licence and to send telemetry by default (now off).

The lockfile is about a year old. Most of it upgrades freely. Five things don't, and they're
connected:

1. **torch is held at 2.6.0 by pyannote.audio 3.** torchaudio ≥ 2.9 removed `AudioMetaData`,
   which pyannote.audio 3 reads audio through (`3ac2034`). Our own code doesn't use torchaudio. So
   the torch upgrade *is* the pyannote.audio 4 migration (which reads audio through torchcodec and
   FFmpeg's shared libraries instead).
2. **opencv-python is held below 5 by `face_analysis`**: its fallback detector and DeepFace's
   default `opencv` detector use the Haar cascades that 5.0 removed. If the review replaces
   `face_analysis`, the cap goes with it.
3. **TensorFlow** comes only from DeepFace (`face_analysis`). It's the one thing holding back
   Python 3.14.
4. **transformers 5 / huggingface-hub 2**: the `<5` / `<1` caps were defensive (`d9c64e8`). They
   only matter to `face_laion_clip` and `laion_voice`, which the review recommends dropping.
   pyannote.audio 4 and the HF downloads in readiness still use huggingface-hub, so its upgrade
   needs testing either way.
5. **`openface-test` stays `==0.1.13`** (`f4cdae7`): later releases pin Pillow 9.4, numpy 1.26 and
   scipy 1.13, which must be built from source on Python ≥ 3.12.

## 2. Python

- **Now**: `>=3.12,<3.13` since v1.2.0 (`ca5c422`), with no recorded reason beyond what was
  tested. 3.12 gets security fixes only, until October 2028.
- **3.13**: see §7 for the trial. With every other constraint unchanged, the lock resolves for 3.12
  and 3.13 together, adding only five backports of standard-library modules 3.13 removed
  (`audioop-lts`, `standard-aifc`, `standard-chunk`, `standard-sunau`, plus `pyyaml-ft`), which the
  audio stack needs.
- **3.14**: blocked by TensorFlow (no 3.14 release) through DeepFace, and `openai-whisper` and
  `ultralytics` don't list 3.14 yet. Also `open-clip-torch` lists only up to 3.12 and
  `tf-keras` up to 3.12, though both install on 3.13 (classifiers lag).
- **Also**: VTC 2 (review §3) requires Python ≥ 3.13.
- **Recommendation**: `requires-python = ">=3.12,<3.14"` in the first spec (keeps 3.12 users
  working), CI on 3.12 and 3.13, the dev container and Docker images on 3.13. Add a non-blocking
  3.14 CI job once TensorFlow is gone or ships for 3.14. Move the floor to 3.13 at v1.7.0.

## 3. Core dependencies

**Done in spec 013** (2026-10-01): unused dependencies removed, numba moved to `audio`, core
upgraded; core-only install 73 → 40 packages, 746 → 135 MB.

| Package | Constraint | Locked | Latest | Notes | Recommendation |
|---|---|---|---|---|---|
| fastapi | `>=0.115.0` | 0.116.1 | 0.142.2 | | Upgrade; raise floor |
| uvicorn[standard] | `>=0.30.0` | 0.35.0 | 0.54.0 | | Upgrade |
| python-multipart | `>=0.0.9` | 0.0.20 | 0.0.32 | Not imported, but FastAPI needs it for uploads | Keep, upgrade |
| pydantic | `>=2.0.0` | 2.11.7 | 2.13.5 | | Upgrade |
| sqlalchemy | `>=2.0.0` | 2.0.43 | 2.1.1 | 2.1 is a minor with deprecation removals; two `Job` models share one table (see CI fixes on `1.6-dev`), worth fixing first | Upgrade after the `Job` model cleanup |
| alembic | `>=1.12.0` | 1.16.4 | 1.20.0 | **Not imported anywhere.** Migrations are hand-written | Drop, or adopt for real migrations (spec decides) |
| numpy | `>=1.24.0` | 2.2.6 | 2.5.3 | Ceiling set by numba, by design | Upgrade; raise floor to 2.0 |
| pandas | `>=2.2.2` | 2.3.2 | 3.0.6 | **Major**: copy-on-write and a string dtype by default. Used in one module | Upgrade with a test of that module |
| numba | `>=0.60.0` | 0.61.2 | 0.68.0 | Used only by the audio pipeline (and librosa) | **Move to `audio`** |
| Pillow | `>=10.4.0` | 11.3.0 | 12.3.0 | | Upgrade |
| pycocotools | `>=2.0.10` | 2.0.10 | 2.0.11 | | Upgrade |
| webvtt-py | `>=0.5.1` | 0.5.1 | 0.5.1 | Last release 2024; classifiers stop at 3.12 | Keep; small, pure Python |
| praatio | `>=6.2.0` | 6.2.0 | 6.2.2 | | Upgrade |
| pyjwt | `>=2.10.1` | 2.10.1 | 2.15.1 | | Upgrade |
| cryptography | `>=45.0.6` | 45.0.6 | 50.0.2 | Not imported; pyjwt only needs it for RSA/EC keys, which we don't use | Drop unless an import turns up |
| requests, pyyaml, packaging, psutil, python-dotenv, typer | various | | | | Upgrade |
| moviepy | `>=1.0.3` | 2.2.1 | 2.2.1 | Only `visualization/video_utils.py` uses it, via `moviepy.editor`, **which moviepy 2 removed**, so that code is already broken; nothing imports the module | Drop with the dead module |
| matplotlib | `>=3.9.2` | 3.10.5 | 3.11.2 | Only listed in `version.py`'s version report | Drop |
| tqdm | `>=4.65.0` | 4.67.1 | 4.70.1 | Only in `version.py` | Drop (still arrives transitively where needed) |
| openpyxl, imageio, imageio-ffmpeg, av, rich, click, scikit-image | various | | | **Not imported anywhere in `src/`**; click comes with typer anyway | Drop, after checking `scripts/` and the docs' examples |

Dropping the unused ones makes the core install smaller, which is v1.5.0's "slim by default" goal.

## 4. Extras groups

| Group | Pipelines | Key constraints | Recommendation |
|---|---|---|---|
| `face` | `face_analysis` | `deepface`, `tf-keras`, `opencv-python<5`, `imutils` (not imported: drop) | Depends on the review. If replaced: remove the group's TensorFlow stack and the opencv cap. If kept: keep `<5`, set the DeepFace detector default to `retinaface` |
| `face-laion` | `face_laion_clip` | `transformers<5`, `huggingface-hub<1`, torch | Remove if the pipeline is dropped |
| `face-openface3` | `face_openface3_embedding` | `openface-test==0.1.13`, scipy, opencv | Keep the pin. Show the non-commercial licence before install |
| `audio` | speech, diarization, `audio_processing` | torch/torchaudio `==2.6.0`, `pyannote.*<next major`, librosa, openai-whisper | **pyannote.audio 4 migration** together with the torch upgrade (one spec). librosa 1.0 is a major: test. Consider faster-whisper (review) |
| `audio-laion` | `laion_voice` | as `face-laion` | Remove if the pipeline is dropped |
| `scene` | `scene_detection` | open-clip-torch, scenedetect, opencv | Upgrade (scenedetect 0.7, open-clip 3.3) |
| `person` | `person_tracking` | ultralytics (AGPL-3.0), supervision (not imported: drop), `lap` (added 2026-10-01: ByteTrack needs it, ultralytics doesn't declare it) | Upgrade; licence notice; replacement is Phase 5 |
| `llm` | `vlm_annotation` | ollama | Upgrade |
| `dev` | | ruff, mypy, pre-commit, jupyter, jupyterlab, pytest-cov | Upgrade; mypy 2 is a major. Move to `[dependency-groups]` only (it's in both) |
| `annotation` | | empty (commented out) | Remove |

**torch everywhere**: `torch==2.6.0` is repeated in five groups so they combine. After the
pyannote migration, use one range (`torch>=2.14,<2.15`) in every group, so combinations still
resolve to one torch and upgrades are deliberate.

## 5. CUDA, drivers, containers

- **Wheel index**: `pytorch-cu124` for Linux. torch 2.14 isn't built for cu124; its oldest CUDA
  build is **cu126**, which matches the `nvidia/cuda:12.6.0-runtime-ubuntu24.04` Docker base.
  Recommendation: move to `cu126` with the torch upgrade. That raises the minimum NVIDIA driver
  from the CUDA 12.4 level to the 12.6 level (NVIDIA's release notes give the exact numbers; state
  them in the install docs).
- **Docker**: `ubuntu:24.04` (CPU) and `nvidia/cuda:12.6.0-runtime-ubuntu24.04` (GPU, dev). Ubuntu
  24.04 ships Python 3.12, so 3.13 comes from uv (`uv python install`), not apt. Image size against
  the v1.4.3 baseline is a separate roadmap item.
- **pyannote.audio 4 needs FFmpeg's shared libraries** at runtime (torchcodec). The images have the
  `ffmpeg` CLI; check that the libraries torchcodec loads are there too.

## 6. Tooling

**Done in spec 013** (2026-10-01), except the viewer's packages and Bun (Phase 0).

| Item | Now | Latest | Recommendation |
|---|---|---|---|
| pre-commit `mirrors-mypy` | v1.5.1 | v2.3.1 | **Upgrade, and match the project's mypy.** v1.5.1 is why pre-commit passed what CI's mypy failed (`6d6c347`) |
| `ruff-pre-commit` | v0.14.0 | v0.16.9 | Upgrade with the `ruff` dev dependency |
| `pre-commit-hooks` | v4.4.0 | v6.0.0 | Upgrade |
| `bandit` | 1.7.5 | 1.9.4 | Upgrade |
| `hadolint` | v2.12.0 | v2.15.1 | Upgrade |
| `shellcheck-py` | v0.9.0.5 | v0.11.0.1 | Upgrade |
| `commitizen` | v3.7.1 | v4.19.0 | Upgrade, or drop if unused |
| `pydocstyle` | 6.3.0 | 6.3.0 (2023, unmaintained) | Replace with ruff's `D` rules |
| `mirrors-prettier` | v3.0.3 | only v4 alphas | Drop, or use the viewer's own formatter for YAML/JSON/Markdown |
| `pre-commit-hooks-safety` | v1.3.2 | v1.4.2 | Upgrade, or replace with `uv`'s audit / pip-audit |
| `actions/checkout` | v4 | v7 | Upgrade: v4, `setup-python@v5` and `setup-uv@v5` run on Node 20, which CI now warns is deprecated |
| `actions/setup-python` | v5 | v7 | Upgrade (or drop: setup-uv can install Python) |
| `astral-sh/setup-uv` | v5 | v10 | Upgrade |
| `actions/upload-artifact`, `codecov-action`, docker actions | v4 / v4 / v3–v6 | v7 / v7 / v7 | Upgrade |
| `openjournals-draft-action` | `@master` | | Pin to a commit |
| JS: Bun | 1.x in CI, 1.4.2 in the dev container | | Keep Bun; drop `viewer/package-lock.json` (Phase 0 item) |
| Viewer packages | | | Separate audit of `viewer/package.json` (not done in this draft) |

## 7. Python 3.13 trial

Run 2026-10-01 in a separate worktree (branch `py313-trial`), CPython 3.13.15, Linux x86_64 with
CUDA.

**A. Python 3.13, every other constraint unchanged** (`requires-python = ">=3.12,<3.14"`):
- `uv lock` resolves 311 packages for 3.12 and 3.13 together; the only additions are the five
  standard-library backports in §2.
- `uv sync --all-extras` installs everything on 3.13, including the source builds
  (`openai-whisper`) and `openface-test==0.1.13`: 641 packages, about 20 minutes, mostly CUDA wheel
  downloads.
- All ten pipeline classes import.
- Test suite (`-m "not real_models"`, as CI runs it): **1335 passed, 33 skipped, 0 failed** in
  2 min 20 s. CI on 3.12 for the same commit: 1332 passed (fewer extras installed there).
- Real pipelines on the viewer's demo video (`2UWdXP…Peekaboo_h265.mp4`, run through the server's
  `JobProcessor`): `speech_recognition`, `scene_detection`, `face_analysis`,
  `face_openface3_embedding`, `person_tracking` and `speaker_diarization` all complete on 3.13.
  `speaker_diarization`'s four segments are identical on 3.12 and 3.13 (start, end and speaker,
  to every digit).
  - `person_tracking` first failed with `No module named 'lap'`. ByteTrack imports `lap`;
    ultralytics doesn't declare it and pip-installs it at runtime instead, which is how the 3.12
    dev environment got it. A fresh or offline install would fail. **Fixed on `1.6-dev`**: `lap`
    is now in the `person` extra. Not a 3.13 issue.
  - `face_analysis` finds 0 faces in this video on both 3.12 and 3.13 (same result, so not a
    regression; noted in the pipeline review).

So the Python upgrade doesn't depend on any library upgrade and can be its own small spec.

**B. Python 3.13 with every upper bound and exact pin lifted, cu126 index** (resolution only):
resolves 318 packages to torch 2.14.1+cu126, torchaudio 2.11.0, torchvision 0.29.1, pyannote.audio
4.0.7, transformers 5.18.0, opencv-python 5.0.0.93, TensorFlow 2.21.0, numpy 2.5.3, pandas 3.0.6,
librosa 1.0.0, ultralytics 8.4.170, with `openface-test` still at 0.1.13. Two packages stay below
their latest, both for reasons outside our constraints:
- Pillow 11.3 (latest 12.3): `moviepy` 2.2.1 requires `pillow<12`. Dropping moviepy (§3) lifts it.
- huggingface-hub 1.33 (latest 2.0): `tokenizers` (via transformers) requires `<2.0`. Upstream;
  1.x is current.

Resolution says nothing about behaviour: B's majors (pyannote.audio 4, opencv 5, pandas 3,
librosa 1.0, transformers 5) need the specs and their tests.

## 8. Diarization and the torch cap (2026-10-02)

Not an immediate priority; recorded so Phase 5 starts from it.

**Why torch is held at 2.11.** pyannote.audio 4 imports torchaudio (audio I/O fallbacks and the
speaker-embedding code). torchaudio is discontinued: its last release, 2.11, is compiled against
torch 2.11 and fails to load on anything newer (`OSError: Could not load this library:
.../_torchaudio.abi3.so`). It declares no torch requirement, so a resolver pairs it with torch 2.14
and the failure only shows at runtime. Every extras group pins the same torch so combinations such
as `[all]` resolve, so the whole stack waits. Our own code never imports torchaudio. Cost today:
three minor torch releases, nothing broken (outputs unchanged, see CHANGELOG). The cost grows when
a future model or library needs torch ≥ 2.12; whether torch 2.11 has builds for the newest GPUs
(RTX 50-series) is unchecked.

**Upstream state.** pyannote.audio 4.0.7 (2026-06-30) still requires `torchaudio>=2.8`. No open
work to drop it: a PR relaxing the torch/torchaudio pins (#1977) was closed unmerged in February,
and there have been no commits since 2026-06-30. Open issues worth knowing: #2019 "Remote Code
Execution via Attacker-Controlled config.yaml Class Loading" (low risk for us: we load the
official models) and #1963 (4.x uses ~6× the VRAM of 3.3). The company's effort appears to be on
its hosted models.

**Alternatives checked.**

| Option | Removes torchaudio? | Notes |
|---|---|---|
| **VTC 2** (LAAC-LSCP): adult female, adult male, key child, other child | **No**: depends on `pyannote-audio>=3.4` | The tool the field wants (pipeline review: top Phase 5 priority). No LICENSE file in the repository. Installed by git clone + git-lfs, not from PyPI; Python ≥ 3.13; no Windows support |
| **NVIDIA Streaming Sortformer** (`diar_streaming_sortformer_4spk-v2.1`) via NeMo | **Yes** | NeMo 3.0.0 (2026-08, Apache-2.0); model not gated (no HF-token hurdle), NVIDIA Open Model License, updated 2026-09. But NeMo's `asr` extra brings 23 packages, among them wandb, hydra and `nv_one_logger_*` telemetry packages to vet against Local-First. It caps `lightning<=2.4.0` while pyannote needs `>=2.4`: both together only at exactly 2.4.0. Up to 4 speakers |
| **DiariZen** (BUT) | No | Vendors a pyannote fork pinned to torch 2.1.1, numpy 1.26.4, Python 3.10. Weights CC BY-NC. A step back |

**Decision (2026-10-02): decide now, swap later.**
1. Keep pyannote through v1.6.0 Phase 1: nothing is broken, and no default changes without a
   benchmark score.
2. Phase 5's diarization benchmark scores pyannote community-1, Sortformer and VTC 2, with
   dependency health as a criterion alongside accuracy.
3. Before Phase 5 relies on it, a half-day spike: Sortformer in a scratch environment on torch
   2.14 without torchaudio; what the telemetry does and whether it can be switched off; install
   size; output on the demo clip.
4. Settle VTC 2's licence with LAAC-LSCP early (maintainer to email). If it stays unlicensed it
   can't ship, and the voice-type plan changes.

Sources: https://github.com/pyannote/pyannote-audio/releases ,
https://github.com/LAAC-LSCP/VTC , https://huggingface.co/coml/VTC-2.0 ,
https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1 ,
https://github.com/BUTSpeechFIT/DiariZen ,
https://neosophie.com/en/blog/20260223-diarization (2026 comparison: Sortformer v2-streaming and
DiariZen among the best open models).

## Specs this suggests

In order, each one small enough to review. The pipeline review's recommendations were all accepted
(2026-10-01).

1. **Python 3.13**: `requires-python <3.14`, CI on 3.12 and 3.13, the dev container and Docker
   images on 3.13. Nothing else: §7 shows it needs no library change.
2. **Core clean-up and tooling**: drop unused core dependencies, move numba to `audio`, upgrade the
   core, pre-commit hooks and GitHub Actions.
3. **torch 2.14 + pyannote.audio 4 + cu126**: one change, since each forces the others.
4. **Drop pipelines**: `laion_voice`, `face_laion_clip`, `audio_processing`, with their extras.
5. **Face stack**: replace `face_analysis`; with it go DeepFace, TensorFlow and tf-keras, and the
   opencv `<5` cap. This is the path to Python 3.14, so do it before rc1 if Phase 5's benchmark
   allows, and add the 3.14 CI job with it.
6. **Viewer dependencies** (separate audit).
