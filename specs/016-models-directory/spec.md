# Feature Specification: One Models Directory

**Feature Branch**: `1.6-dev` (v1.6.0 development branch)
**Created**: 2026-10-01
**Status**: Implemented 2026-10-01 on `1.6-dev`
**Input**: v1.6.0 Phase 1, "one place for model weights" ([`roadmap_v1.6.0.md`](../../docs/development/roadmap_v1.6.0.md)).
Today each library keeps weights where it likes: Whisper and YOLO under `./models/...` relative to
wherever the server was started (a different directory means another download), Hugging Face
under `~/.cache/huggingface`, pyannote under `~/.cache/torch/pyannote`, torch hub under
`~/.cache/torch`, DeepFace under `~/.deepface`. Users can't find them; containers lose them on every
rebuild. Only the dev container points them all into `models/`, with four environment variables.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Weights download once, to one place I can find (Priority: P1)

A researcher runs pipelines, restarts the server from a different directory, or upgrades
VideoAnnotator, and models are not downloaded again. They can see where the weights are and how much
space they take.

**Why this priority**: a first run downloads gigabytes; doing it twice, or not knowing what to delete
when space runs out, is the most common friction after install.

**Independent Test**: run every pipeline once; start the server from another directory; run again;
nothing downloads. `videoannotator diagnose` shows the directory and its size.

**Acceptance Scenarios**:

1. **Given** a fresh install, **When** pipelines run, **Then** every pipeline's weights land under one
   directory, one subdirectory per source (`huggingface/`, `pyannote/`, `torch/`, `deepface/`,
   `whisper/`, `yolo/`).
2. **Given** weights already downloaded, **When** the server starts from a different working
   directory, **Then** nothing is downloaded again.
3. **Given** any install, **When** the user runs `videoannotator diagnose`, **Then** it shows the
   models directory, its size, and the size per source.

---

### User Story 2 - I choose where models go (Priority: P1)

A lab with a shared data disk, or a container with a mounted volume, sets one setting and every
library uses it.

**Independent Test**: set `VIDEOANNOTATOR_MODELS_DIR`; run pipelines; everything lands there.

**Acceptance Scenarios**:

1. **Given** `VIDEOANNOTATOR_MODELS_DIR` is set, **Then** every library's weights go there.
2. **Given** the user has set a library's own variable (e.g. `HF_HUB_CACHE`, `TORCH_HOME`), **Then**
   that one is respected for that library.
3. **Given** the dev container or the Docker images, **Then** models persist across rebuilds and
   container restarts (a mounted directory or named volume).

---

### User Story 3 - Readiness tells the truth about downloads (Priority: P2)

The viewer's "first run downloads about N MB" never claims a model is missing when it's present, or
present when it's missing, because it looks where the pipelines will look.

**Independent Test**: with weights present in the models directory, readiness reports nothing to
download; remove one, it reports that one.

---

### User Story 4 - Upgrading users aren't surprised (Priority: P2)

A lab upgrading from v1.5 learns once that models now live elsewhere, and where the old copies are,
instead of silently re-downloading gigabytes and filling the disk with duplicates.

**Acceptance Scenarios**:

1. **Given** weights in the old default locations and an empty models directory, **When** the server
   starts, **Then** it logs, once, the new location and the old locations that hold weights, and how
   to move them.

### Edge Cases

- The Hugging Face *token* must not move: `huggingface-cli login` stores it under `HF_HOME`, so the
  models directory sets the hub cache only (`HF_HUB_CACHE`), never `HF_HOME`.
- Configs that name a YOLO model by an explicit path keep working; a bare model name or the old
  default `models/yolo/...` resolves into the models directory.
- OpenFace's `./weights/` (where `openface download` puts files) is still honoured if present.
- Ollama keeps its own model store (it's a separate server); out of scope.
- Two users on one machine get separate per-user directories by default.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: One setting, `VIDEOANNOTATOR_MODELS_DIR`, MUST decide where every pipeline's weights
  go; it MUST be resolved to an absolute path once, at import.
- **FR-002**: Default: the platform's per-user data directory (`$XDG_DATA_HOME` or
  `~/.local/share/videoannotator/models` on Linux, `~/Library/Application Support/videoannotator/models`
  on macOS, `%LOCALAPPDATA%\videoannotator\models` on Windows).
- **FR-003**: VideoAnnotator MUST set `HF_HUB_CACHE`, `TORCH_HOME`, `PYANNOTE_CACHE` and
  `DEEPFACE_HOME` from it before any pipeline library is imported, unless the user set them; it MUST
  NOT set `HF_HOME`.
- **FR-004**: Whisper's and YOLO's default locations MUST come from the same directory.
- **FR-005**: Readiness MUST find weights through the same resolver.
- **FR-006**: On server start, if the models directory is empty and old default locations hold
  weights, VideoAnnotator MUST log the old and new locations once.
- **FR-007**: `videoannotator diagnose` MUST show the models directory and its size, per source.
- **FR-008**: The dev container MUST use one variable (`VIDEOANNOTATOR_MODELS_DIR=<repo>/models`) in
  place of its four, with the same on-disk layout, so nothing is re-downloaded; the Docker images and
  docker-compose MUST keep models in a volume.
- **FR-009**: Install docs and CHANGELOG MUST explain the location, the setting, and the one-time
  re-download for upgraded installs.

### Key Entities

- **Models directory**: root path; per-source subdirectories; size.

## Success Criteria *(mandatory)*

- **SC-001**: After one run of every pipeline, a second run from a different working directory
  downloads nothing.
- **SC-002**: `diagnose` reports the models directory and a size within 1% of `du`.
- **SC-003**: Rebuilding the dev container downloads no model weights.
- **SC-004**: Pipeline outputs are unchanged (same weights, new location).

## Assumptions

- A one-time re-download for upgraded non-container installs is acceptable when announced
  (roadmap); moving files automatically is not attempted (too easy to get wrong across libraries).
- Weight prefetch ("download before the first job") is a separate spec built on this directory.
