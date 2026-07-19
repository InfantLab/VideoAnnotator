# Quickstart: Verifying the Extras-Based Install

Manual/CI verification steps mapped to spec 004's acceptance scenarios.

## Two environments

Every section below uses one of these two. Each section header says which.

- **DEV VENV** — this repo's own `.venv`, managed by `uv` (`uv sync --dev --all-extras`, `uv run
  ...`). Has the test suite's tooling (`pytest-asyncio` etc.) installed, which real end users never
  get — those live in `[dependency-groups] dev`, a mechanism `pip install package[extra]` doesn't
  touch. Use this whenever a section runs `pytest`.

- **SCRATCH VENV** — a throwaway venv you create fresh (`python -m venv /tmp/va-scene`), where you
  `pip install` this project the way an actual end user would, to prove real extras-isolation
  behaviour. Not on PyPI yet, so install from the local checkout: `pip install
  /workspaces/VideoAnnotator[scene]` (or `pip install .[scene]` if your shell is already inside the
  repo).

  **Recreate it fresh for each full pass through §1–§4 below** — don't keep adding extras to the
  same one across sessions. A stale scratch venv can carry forward corruption that a `pyproject.toml`
  fix alone won't clean up (e.g. `opencv-python`/`opencv-python-headless` having been installed
  together at some point — removing one afterwards leaves the other's files partially broken,
  since they share the same `cv2` install path). `pip install --force-reinstall` does **not** fix
  this either — pass `--no-deps` only if you specifically want to skip dependency changes, which
  means any `pyproject.toml` fix (extras pins, new dependencies) won't take effect until you drop
  `--no-deps` or start over.

`videoannotator pipelines` and `videoannotator job submit` are both HTTP clients against the API
server (default `http://localhost:18011`) — start `videoannotator server --dev` in a second
terminal (same scratch venv activated) before either command does anything useful.

## §0 — Clear old install..

```
rm -rf /tmp/va-scene
```

---

## §1 — Slim single-family install (User Story 1) — SCRATCH VENV

```bash
python -m venv /tmp/va-scene && source /tmp/va-scene/bin/activate
pip install /workspaces/VideoAnnotator[scene]

pip show torch open-clip-torch scenedetect     # present — scene needs them (research.md §1)
pip show pyannote.audio ultralytics deepface   # MUST be absent
```

In a second terminal, same venv:

```bash
source /tmp/va-scene/bin/activate
videoannotator server --dev
```

Back in the first terminal:

```bash
videoannotator pipelines   # only scene_detection shown

videoannotator job submit /workspaces/VideoAnnotator/tests/fixtures/videos/test.mp4 \
  --pipelines scene_detection   # succeeds

videoannotator job submit /workspaces/VideoAnnotator/tests/fixtures/videos/test.mp4 \
  --pipelines face_analysis     # actionable error, no traceback
```

## §2 — Additive install (User Story 1, scenario 4) — SCRATCH VENV

Same venv as §1, extending it (don't recreate).

```bash
pip install /workspaces/VideoAnnotator[face]
```

Restart the server (Ctrl-C the §1 one, run `videoannotator server --dev` again — extras are
resolved at process start, not picked up live):

```bash
videoannotator pipelines   # scene_detection AND face_analysis now shown; nothing reset
```

## §3 — Full-parity install (User Story 2) — DEV VENV

```bash
cd /workspaces/VideoAnnotator
uv sync --dev --all-extras
uv run pytest tests/integration/test_v144_parity.py -v
# v1.4.4 fixtures: byte-identical or within documented tolerance.
# Currently mostly SKIPPED — no v1.4.4 golden fixtures captured yet
# (see the test file's own module docstring for how to capture them
# from the v1.4.4 git tag). Not a regression from this session's work.
```

## §4 — LAION migration message (User Story 2, scenario 2) — SCRATCH VENV

Fresh scratch venv (or reuse §1/§2's if it only has `face` installed, not `face-laion`).

```bash
pip install /workspaces/VideoAnnotator[face]   # face, not face-laion
# server already running from §1/§2, or start it fresh — see §1
videoannotator job submit /workspaces/VideoAnnotator/tests/fixtures/videos/test.mp4 \
  --pipelines face_laion_clip
# expect: "no longer installed by default as of v1.5.0" message naming
# `face-laion`, not a raw ImportError
```

## §5 — NumPy 2.x — DEV VENV

Don't `pip install "numpy>=2.0"` directly — that bypasses the resolver and grabs the newest
release on PyPI, which can be newer than `numba` (a core dependency) actually supports and breaks
audio/scene imports (`ImportError: Numba needs NumPy 2.x or less`; see `tasks.md` T030). Let the
resolver pick the newest numpy `numba` supports instead:

```bash
cd /workspaces/VideoAnnotator
uv lock --upgrade-package numpy
uv sync --dev --all-extras
uv run pytest -x
```

## §6 — Forward-compatibility gate (User Story 3 / SC-007) — DEV VENV

```bash
cd /workspaces/VideoAnnotator
cp tests/fixtures/stub_pipeline.yaml src/videoannotator/registry/metadata/  # or a test-only registry dir
uv run python -c "from videoannotator.registry import get_registry; get_registry().load(force=True); print('ok')"
```

Author a throwaway stub pipeline YAML with `requires_extras: []` and a `module_path` pointing at a
`NotImplementedError` stub; confirms the registry loads it without code changes. (Already covered
by an automated test: `tests/unit/registry/test_forward_compat_stub.py`.)

---

## Known pitfalls (found via real runs of §1/§2/§4)

- **`videoannotator pipelines list` doesn't exist** — `pipelines` is a flat command, not a group;
  the correct invocation is just `videoannotator pipelines`. (`job` *is* a group: `job submit`,
  `job status`, `job list`, etc.)
- **`ModuleNotFoundError: No module named 'librosa'` on any non-audio install** — was a real bug
  (`videoannotator/utils/audio.py` imported `librosa` at module level, pulled in by the CLI at
  startup regardless of which extras were installed); fixed by making that import lazy.
- **`[OK] Pipelines: 0 found` even with the right extras installed** — was a real packaging bug
  (`registry/metadata/*.yaml` was never declared in `package-data`, so no pipeline metadata shipped
  in a real install at all); fixed by adding it to `[tool.setuptools.package-data]`.
- **`WARNING: scenedetect 0.7 does not provide the extra 'opencv'`** on `pip install .[scene]` —
  harmless, but fixed anyway: dropped the `[opencv]` extra from the `scene` group's scenedetect
  pin (see next item for why we don't rely on any package's own `[opencv]`/`[opencv-headless]`
  extra at all now).
- **`job submit --pipelines scene_detection` failed with `No module named 'librosa'`, server-side**
  — the biggest bug of this phase. `videoannotator/pipelines/__init__.py` (and, one level down,
  `audio_processing/__init__.py` and `face_analysis/__init__.py`) unconditionally imported *every*
  pipeline family, so loading any single pipeline forced every other family's heavy deps to import
  too — no pipeline could ever load except under an `[all]` install, regardless of which extras
  were actually present. Fixed by making those three `__init__.py` files resolve their pipeline
  classes lazily (PEP 562 `__getattr__`) instead of importing eagerly. `face-laion` had a related,
  genuine (not just packaging) bug: `LAIONFacePipeline` composes `FaceAnalysisPipeline` as its
  detector backend, so it was never functional alone even before this fix — `face-laion` now
  depends on `videoannotator[face]`.
- **`AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`** when running
  `face_laion_clip` (or any `face` pipeline using the "opencv" detector backend) — two stacked
  bugs. First: core previously declared `opencv-python-headless` unconditionally while `face`
  (deepface) and `person` (ultralytics/supervision) transitively force plain `opencv-python` —
  installing both into the same venv corrupts the shared `cv2` install (a documented upstream
  footgun, github.com/opencv/opencv-python#note-1), and **removing one afterwards doesn't fix an
  already-corrupted venv** — the leftover `cv2/` directory stays broken until you `rm -rf` it and
  reinstall clean. Fixed by never declaring `opencv-python-headless` anywhere and standardizing on
  plain `opencv-python` in every extras group that needs it. Second, independent bug found while
  fixing the first: `opencv-python` 5.0 (released after this project's original `>=4.11.0.86` pin
  was written) **removed `cv2.CascadeClassifier` and all Haar-cascade data entirely**, replaced by
  the DNN-based `FaceDetectorYN` — `face_pipeline.py`'s "opencv" backend still uses the legacy API,
  so an unpinned install breaks at runtime, not install time. Pinned `opencv-python<5.0` everywhere
  it's declared. Migrating to `FaceDetectorYN` is real follow-up work, not a pyproject fix.
- **`Scene classification failed: too many values to unpack (expected 2)`** on `scene_detection`
  jobs — `scene_pipeline.py` called `open_clip`'s `model(image, text)` expecting the legacy 2-tuple
  `(logits_per_image, logits_per_text)`; `open-clip-torch` 3.1.0 returns
  `(image_features, text_features, logit_scale)` instead. Fixed by computing the similarity logits
  directly (`logit_scale * image_features @ text_features.T`).
- **`No DNN in stream executor` warnings from DeepFace's "emotion" action, on GPU installs** — torch
  hard-pins `nvidia-cudnn-cu12==9.1.0.70`, `tensorflow` (deepface's dependency) wants cuDNN
  `>=9.3.0.75` for GPU use; no single cuDNN version satisfies both pins, so this can't be fixed by
  hunting for a compatible tensorflow build. Fixed by forcing TensorFlow onto CPU before deepface
  touches the GPU (`tf.config.set_visible_devices([], "GPU")` in `face_pipeline.py`) — torch's own
  GPU usage (scene/person/face-laion) is untouched, since this uses TF's own device-visibility API,
  not the shared `CUDA_VISIBLE_DEVICES` env var. Set `VIDEOANNOTATOR_DEEPFACE_GPU=1` to opt back
  into TF-on-GPU if you've resolved the cuDNN mismatch yourself.
