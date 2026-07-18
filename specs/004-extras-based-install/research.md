# Phase 0 Research: Extras-Based Modular Install

## 1. Actual per-pipeline dependency footprint

**Question**: Spec 003/roadmap assumed extras groups are `face`, `audio`, `scene`, `person`,
`embedding`. Is that grouping accurate to what each pipeline's code actually imports?

**Method**: Traced module-level and lazy (in-method) imports in each pipeline's source
(`src/videoannotator/pipelines/*/`) rather than trusting the metadata YAML's informal
`requirements.packages` list, since that list predates this refactor and isn't necessarily complete.

**Decision**: Use these five extras groups, matching actual `pipeline_family` values already in
the metadata (`face`, `audio`, `scene`, `person`) plus two LAION carve-outs — **not** a separate
`embedding` group, because no pipeline in the current registry declares `pipeline_family:
embedding`; `face_openface3_embedding` is `pipeline_family: face, variant: openface3-embedding`.
Inventing an `embedding` group not backed by the taxonomy would create a group with nothing in it.

| Extras group | Pipelines (variant) | Heavy deps actually imported |
|---|---|---|
| `face` | `face_analysis` (deepface) | `deepface`, `opencv-python`, `imutils` — **no torch** |
| `face-laion` | `face_laion_clip` | `torch`, `transformers`, `huggingface-hub` |
| `face-openface3` | `face_openface3_embedding` | `openface-test`, `scipy`, `torch` (lazy, only on embedding-similarity path) |
| `audio` | `audio_processing`, `speech_recognition`, `speaker_diarization` | `torch`, `torchaudio`, `librosa`, `openai-whisper`, `pyannote.audio`+`core`+`database`+`metrics`+`pipeline` |
| `audio-laion` | `laion_voice` | `torch`, `transformers`, `huggingface-hub`, `librosa` |
| `scene` | `scene_detection` | `torch`, `open-clip-torch`, `scenedetect[opencv]` |
| `person` | `person_tracking` | `torch` (via ultralytics), `ultralytics`, `supervision` |

**Rationale**: `face` (DeepFace-only) is the one genuinely torch-free group — worth calling out
explicitly in docs/SC-001 examples ("I want only scene labelling" pulls torch; "I want only
standard face analysis" does not). `scene` and `person` both need torch even though a naive
reading of the roadmap implied otherwise; this doesn't block FR-001 (no extras = no torch), but it
does mean `scene` alone is not as lightweight as `face` alone. Document this plainly rather than
imply all five groups are equally slim.

**Alternatives considered**:
- A single `embedding` group holding CLIP/openface3/sentence-transformers shared code: rejected —
  no such shared module exists today; would require a real code split, out of scope for this phase
  (FR-001–FR-009 are about dependency *declaration*, not module *extraction*, which is v2.0 scope
  per `roadmap_v1.7_to_v2.0.md`).
- Merging `face-laion` and `face-openface3` into one `face-extended` group: rejected — they have
  different dependency sets (`transformers`+`huggingface-hub` vs `openface-test`+`scipy`) and
  different licenses to track per constitution Engineering Standards; keeping them separate matches
  FR-002's requirement that LAION not be bundled into anything non-opt-in, and doesn't block a user
  who wants openface3 embeddings but not LAION.

## 2. `requires_extras` semantics vs. `backends`

**Question**: `PipelineMetadata` already has a `backends: list[str]` field (currently populated
with values like `tensorflow`, `opencv`). FR-011 requires `requires_extras: []` to be valid, for a
future non-local pipeline. Do these two fields overlap?

**Decision**: They're orthogonal and both stay. `requires_extras` answers "what `pip install
videoannotator[...]` do I need" (a packaging-time question); `backends` answers "what inference
backend does this pipeline variant use at runtime" (already used descriptively, e.g.
`tensorflow`/`opencv` for `face_analysis`). A future Ollama-backed pipeline (v1.6.0) would have
`requires_extras: [llm]` (a small `httpx`-only extra) and `backends: [ollama]` — both populated,
neither empty, but neither heavy. This confirms FR-011's premise: `requires_extras: []` is reserved
for a pipeline with literally no extra to install (rare, maybe never used pre-v1.6.0), while a
lightweight non-ML extra (`llm`) is the more common future shape. Spec's User Story 3 acceptance
scenario 1 (`requires_extras: []` stub) still stands as the schema-flexibility test; it doesn't
predict that v1.6.0's actual Ollama pipeline will use an empty list.

## 3. Registry graceful-degradation mechanism

**Question**: How should `PipelineLoader`/`PipelineRegistry` detect "extras not installed" without
attempting (and catching an exception from) every pipeline's heavy import at every registry load?

**Decision**: Check `importlib.util.find_spec()` for each package name mapped from
`requires_extras` (e.g. `face-laion` → `["torch", "transformers"]`) before attempting
`importlib.import_module()` on the pipeline's `module_path`. `find_spec` is cheap (no execution of
the target module) and works even for packages that are slow or side-effect-heavy to import.
Package-name-per-extras-group mapping lives alongside the extras definitions in `pyproject.toml`
metadata read via `importlib.metadata`, avoiding a second hand-maintained mapping that could drift.

**Alternatives considered**: try/except around the real import (rejected — heavy ML imports can
have expensive or noisy failure modes, e.g. partial CUDA init, that we don't want happening for
every unavailable pipeline on every registry load) using `importlib.metadata.PackageNotFoundError`
lookups per declared extra requirement instead of `find_spec` per underlying module (viable
alternative, roughly equivalent cost/accuracy — implementation detail for `/speckit-tasks`, not a
blocking research question).

## 4. Migration-message design for demoted pipelines (LAION)

**Decision**: A v1.4.x config referencing `face_laion_clip` or `laion_voice` on a v1.5.0 install
without the corresponding extras produces a distinct message (not the generic "pipeline not
found"): `Pipeline 'face_laion_clip' requires the 'face-laion' extras group, which is no longer
installed by default as of v1.5.0. Install it with: pip install videoannotator[face-laion]`. This
satisfies FR-006 and Edge Cases' distinguishability requirement. Implementation detail (exact
string, i18n if ever needed) belongs in `/speckit-tasks`; the *shape* — name the extra, give the
exact command, explain the "no longer default" context — is the Phase-1 design decision.

## 5. NumPy 2.x verification approach

**Decision**: Add a CI job variant that runs the full suite with NumPy 2.x pinned, gated the same
as the existing 3-OS matrix, before removing the `<2.0` pin from the default dependency set. Keep
the NumPy-1.x job running in parallel until the removal PR merges (belt-and-braces given
`pyannote`/`scipy`/`matplotlib` binary-wheel history noted in the existing pyproject comment at
line 42), then delete the 1.x job once 2.x is confirmed default. This is a CI-config task for
`/speckit-tasks`, not a design decision requiring further research here.

## 6. Docker image slimming approach

**Decision**: `Dockerfile.cpu`/`Dockerfile.gpu` build with no extras by default (`pip install .`,
no `[all]`); a documented `--build-arg EXTRAS=all` (or a separate `Dockerfile.cpu.all` — exact
mechanism is a `/speckit-tasks` implementation choice) reproduces the current image for users who
want everything. SC-002's ≥80% reduction target is measured against the current `[all]`-equivalent
baseline image size recorded at v1.4.4 release.
