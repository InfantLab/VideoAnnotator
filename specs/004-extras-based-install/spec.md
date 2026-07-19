# Feature Specification: Extras-Based Modular Install & Registry Refactor (v1.5.0 Phase 1)

**Feature Branch**: `004-extras-based-install`
**Created**: 2026-07-18
**Status**: Draft
**Input**: User description: "Now that the JOSS resubmission is in review, resume the v1.5.0 modularity work. Implement the extras-based install and metadata-driven registry loading that Phase 1 of the v1.5.0 roadmap scopes (moving heavy ML deps to per-pipeline optional-dependencies groups, replacing LEGACY_MAPPINGS with metadata-driven module loading, registry graceful-degradation, migration messaging, dropping the numpy<2.0 pin). The resulting extras/metadata design must not foreclose the already-roadmapped follow-on work: v1.6.0's Ollama/llama.cpp local-LLM pipeline backend, and v1.7-v2.0's HTTPDispatcher, SlurmDispatcher/HPC dispatch, and secure remote/cloud pipeline execution — this modularity effort exists specifically to make local Ollama models, HPC offload, and remote/cloud offload possible in later releases without re-architecting the registry or install system again."

## Relationship to Existing Specs

This is the **implementation-ready Phase 1 slice** of the architecture already specified in
[`specs/003-modular-pipeline-architecture/spec.md`](../003-modular-pipeline-architecture/spec.md)
("spec 003"). Spec 003 covers the full modular-architecture vision, including plugin discovery,
the dispatcher abstraction, and the cross-cutting-utilities split, and explicitly defers
remote/HPC/VLM *implementation* to v1.7+ (see spec 003's Assumptions section) while requiring
(FR-008) that job dispatch be abstracted so those targets can be added later without touching
pipeline code. This spec does not re-derive that architecture; it narrows spec 003's User Stories
1 and 2 (FR-001–FR-005, FR-009, FR-011, FR-014, FR-015) into a concrete, buildable unit of work —
what [`docs/development/roadmap_v1.5.0.md`](../../docs/development/roadmap_v1.5.0.md) calls
"Phase 1: Extras-Based Install" — and adds one new requirement category: forward-compatibility
guarantees that this phase's schema choices must satisfy, so that v1.6.0 (Ollama), v1.7.0
(HTTPDispatcher, remote pipelines), and v1.8.0 (SlurmDispatcher/HPC) can build directly on it.

**Why this phase, now**: as of this spec's creation, no part of Phase 1 has been implemented —
all runtime ML dependencies remain in `[project.dependencies]` (unconditional), the registry
still resolves module paths through the hardcoded `LEGACY_MAPPINGS` dict in
`src/videoannotator/registry/pipeline_loader.py`, and `numpy<2.0` is still pinned in
`pyproject.toml`. Phase 2 of v1.5.0 (bundling Video Annotation Viewer at `/viewer`) already
shipped in v1.4.4. This spec covers the remaining, unstarted half of the v1.5.0 release.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Slim, targeted install for a single research need (Priority: P1)

A researcher who only needs scene labelling on their lecture-recording corpus installs the
toolkit with just the scene-labelling pipeline. They do not download multi-gigabyte models or
libraries unrelated to their work.

**Why this priority**: Directly answers the JOSS pre-review's flagged blocker (~30 GB Docker
image, every pipeline's dependencies mandatory). This is the single most-cited adoption barrier.

**Independent Test**: `pip install videoannotator[scene]` on a clean environment; run scene
labelling on a sample video; confirm the install pulled no audio/face/person-tracking deps and
produced correct output.

**Acceptance Scenarios**:

1. **Given** a clean Python environment, **When** the user installs with only the `scene` extra,
   **Then** the install completes without downloading torch-dependent audio/face/person packages,
   and scene labelling runs successfully.
2. **Given** the toolkit installed with only `scene`, **When** the user lists available pipelines
   (CLI or `GET /api/v1/pipelines`), **Then** only scene-labelling pipelines appear as available;
   others are omitted, not erroring.
3. **Given** the toolkit installed with only `scene`, **When** the user requests a face-emotion
   pipeline, **Then** they get an actionable message with the exact `pip install
   videoannotator[face]` command, never a Python traceback.
4. **Given** an install with `scene` only, **When** the user runs `pip install
   videoannotator[face]` afterward, **Then** both families work; nothing about the existing
   install needs to be reinstalled or reset.

---

### User Story 2 - Backward-compatible upgrade for existing studies (Priority: P1)

A researcher mid-study on v1.4.4 upgrades to v1.5.0 using `pip install videoannotator[all]` and
keeps running their existing configuration files and scripts unchanged.

**Why this priority**: Studies run for months; breaking a running analysis mid-study violates
Constitution Principle V (Backward Compatibility by Default). Without this, the modularity work
is a hostile change, not a foundation.

**Independent Test**: Run the v1.4.4 acceptance-test fixtures against a v1.5.0 `[all]` install;
outputs match v1.4.4 within documented tolerance; no config file needs editing.

**Acceptance Scenarios**:

1. **Given** an existing v1.4.4 config file and video corpus, **When** upgraded to v1.5.0 `[all]`,
   **Then** the same CLI invocations produce equivalent annotations with zero modifications.
2. **Given** a v1.4.4 config that references LAION (now excluded from the default `face`/`audio`
   extras), **When** run unchanged against v1.5.0, **Then** the user gets a clear migration
   message naming the extras group to add, not a silent failure or raw ImportError.
3. **Given** a v1.4.4 model cache on disk, **When** v1.5.0 runs the same pipeline, **Then** it
   reuses the cache rather than re-downloading.

---

### User Story 3 - Metadata schema has room for non-local execution (Priority: P2)

A maintainer implementing next release's Ollama backend (v1.6.0) or a later release's remote/HPC
dispatch (v1.7.0+) needs to add a pipeline whose actual compute doesn't happen via a local Python
import at all — it happens by calling a local `ollama serve` process, an HTTP endpoint, or a Slurm
job. This spec's registry/metadata changes must accommodate that without another breaking schema
migration.

**Why this priority**: This is the explicit reason modularity is being prioritized right now
(ahead of other JOSS-response work): the extras/registry refactor exists to unblock Ollama,
HPC-offload, and secure remote/cloud execution in the releases immediately following this one. If
Phase 1's schema has to be redesigned again in v1.6.0 to fit Ollama, or again in v1.7.0 to fit
HTTP/Slurm dispatch, this phase has failed its actual purpose even if User Stories 1 and 2 pass.
P2 (not P1) because no non-local backend ships in *this* release — only the room for one.

**Independent Test**: Author a throwaway pipeline metadata YAML entry that declares
`requires_extras: []` (no local ML dependency) and a `module_path` pointing at a stub class that
raises `NotImplementedError` in `process()`. Confirm the registry loads it, lists it as available
with no extras required, and that nothing in the extras/dependency system assumes every pipeline
must ship a heavy local dependency. This proves the schema doesn't hard-code "every pipeline has a
local ML extra" as an assumption.

**Acceptance Scenarios**:

1. **Given** a pipeline metadata YAML with `requires_extras: []` and a `backends` entry naming a
   non-Python-import execution target (e.g. `ollama`, `http`), **When** the registry loads
   metadata, **Then** it loads successfully without requiring changes to the registry loader code.
2. **Given** the Phase 1 extras-group naming scheme, **When** a future release adds an `llm` extras
   group (v1.6.0, for an HTTP client only — no torch/transformers) or reserves names for HPC/remote
   dispatch configuration, **Then** the naming scheme already documented by this phase has room for
   those groups without colliding with or renaming existing groups (`face`, `audio`, `scene`,
   `person`, `embedding`, `all`).
3. **Given** the `PipelineMetadata` dataclass extended by this phase (`requires_extras` field
   added), **When** a later release adds fields for backend connection details (base URL, auth
   reference, dispatch target), **Then** those are additive optional fields, not modifications to
   fields this phase introduces.

---

### Edge Cases

- A user requests a pipeline whose extras are not installed: MUST get an actionable install
  command, never a Python import traceback surfaced to the CLI/API caller.
- A v1.4.4 config references a pipeline demoted from the default install (LAION): MUST be detected
  and produce a migration message, not a first-import crash mid-batch-job.
- Two installed extras groups pin different versions of the same underlying library: MUST resolve
  at install time or surface the conflict at install time, not at runtime.
- A third-party or future-reserved extras-group name collides with an official group: registry MUST
  detect the collision deterministically (full plugin *discovery* is v1.6.0 scope per spec 003 —
  this phase only needs the naming scheme to not preclude that check later).
- A model cache from v1.4.4 exists on disk: v1.5.0 MUST reuse it, not force re-download.
- The environment uses NumPy 2.x: full pipeline suite MUST pass without a downgrade.
- A pipeline's `requires_extras` is empty (no local ML dependency, e.g. a future Ollama/HTTP-backed
  pipeline) but its extras group is still meaningful for a lightweight client dependency (e.g.
  `httpx`): the schema MUST distinguish "no extras needed" from "a small non-ML extra is needed" —
  both are valid and different from "heavy ML extra needed."

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Heavy ML dependencies (torch, torchvision, torchaudio, ultralytics, pyannote.*,
  transformers, sentence-transformers, open-clip-torch, openai-whisper, timm) MUST move from
  `[project.dependencies]` to named `[project.optional-dependencies]` groups: `face`, `audio`,
  `scene`, `person`, `embedding`. `pip install videoannotator` with no extras MUST NOT pull any of
  them in.
- **FR-002**: LAION Empathic-Insight (face and voice) MUST have its own extras group(s), separate
  from the base `face`/`audio` groups, so standard face/audio pipelines don't pull LAION's stack.
- **FR-003**: An `all` meta-extra MUST exist that reproduces v1.4.4's install footprint exactly.
- **FR-004**: `PipelineMetadata` MUST gain a `requires_extras: list[str]` field (additive; existing
  fields, including `backends` which already exists, are untouched). The registry's `pipeline_loader.py`
  MUST resolve `module_path` from this metadata-driven field instead of the hardcoded
  `LEGACY_MAPPINGS` dict, which MUST be removed once every pipeline YAML carries `module_path`.
- **FR-005**: When a pipeline's `requires_extras` are not installed, the registry MUST omit it from
  listings (CLI, `GET /api/v1/pipelines`) rather than raising ImportError during registry load; any
  user-facing surface that names an unavailable pipeline MUST include the exact
  `pip install videoannotator[<group>]` command.
- **FR-006**: A v1.4.x config file that references a pipeline now outside the default install
  (e.g. LAION) MUST produce a specific, actionable migration message identifying the missing
  extras group, distinguishable from a generic "pipeline not found" error.
- **FR-007**: The `numpy<2.0` pin (`pyproject.toml`, currently line 43) MUST be dropped once the
  full pipeline suite is verified against NumPy 2.x; CI MUST run the suite on NumPy 2.x before this
  requirement is considered met.
- **FR-008**: `Dockerfile.cpu`/`Dockerfile.gpu` base images MUST build with no extras (slim); an
  `[all]`-equivalent image MUST remain available and documented separately.
- **FR-009**: Output artifacts produced by v1.5.0 `[all]` MUST match v1.4.4 byte-for-byte for
  deterministic pipelines, and within documented numerical tolerance for non-deterministic ones.
- **FR-010 (forward-compatibility)**: The extras-group naming scheme introduced by this phase
  (`face`, `audio`, `scene`, `person`, `embedding`, `all`, plus LAION's group(s)) MUST be documented
  as a reserved namespace with room left for names known to be needed next: an `llm` group
  (v1.6.0 Ollama/llama.cpp client, no ML weights) and names for remote/HPC dispatch configuration
  (v1.7.0+). This phase does not create those groups — it must not choose names or a registry
  design that blocks creating them later without renaming existing ones.
- **FR-011 (forward-compatibility)**: `requires_extras: []` (empty list) MUST be a valid, supported
  state in the metadata schema, representing a pipeline with no local heavy-ML dependency — the
  shape a future Ollama-backed or HTTP-remote pipeline will use. The registry and pipeline loader
  MUST NOT assume every pipeline entry has a non-empty `requires_extras`.
- **FR-012 (forward-compatibility)**: This phase's registry/metadata changes MUST NOT require the
  job-execution paths (`api/job_processor.py`, `batch/batch_orchestrator.py`) to change how they
  invoke a pipeline based on which extras group it belongs to — invocation MUST stay
  extras-agnostic, so that a future Dispatcher abstraction (spec 003 FR-008, planned v1.6.0+) can
  be introduced without this phase's work needing rework.

### Key Entities

- **Extras group**: A named, documented, pip-installable dependency bundle. Reserved at the project
  level. This phase defines the ML-pipeline groups; the namespace is designed to also hold the
  `llm` group (v1.6.0) and remote/HPC-related groups (v1.7.0+) without collision.
- **Pipeline metadata record**: Per-pipeline YAML under `src/videoannotator/registry/metadata/`.
  This phase adds `requires_extras`; the existing `backends` field (already present, e.g.
  `tensorflow`, `opencv` today) is the same field later releases extend with non-local values
  (`ollama`, `http`).
- **Migration message**: User-facing text mapping "pipeline referenced in config but unavailable"
  to the specific extras-install command or, for demoted pipelines (LAION), an explicit note that
  it moved out of the default install.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `pip install videoannotator[scene]` (or any single-family extra) completes and runs
  that family's pipeline on a sample video in under 5 minutes on a representative laptop with
  broadband, without downloading unrelated families' dependencies.
- **SC-002**: Default (no-extras) Docker image footprint is reduced by ≥ 80% vs. the v1.4.4
  baseline.
- **SC-003**: 100% of v1.4.4 acceptance-test fixtures pass against v1.5.0 `[all]` with zero
  modification to configs, CLI invocations, or downstream analysis code.
- **SC-004**: 100% of pipelines available in v1.4.4 remain installable in v1.5.0 (bundled in `[all]`
  or as a documented extra) — no functionality lost.
- **SC-005**: When a user requests an unavailable pipeline, the error message includes the exact
  pip install command in 100% of cases (measured by test fixtures covering every pipeline family).
- **SC-006**: Full automated test suite passes on NumPy 2.x on Linux, macOS, and Windows.
- **SC-007 (forward-compatibility gate)**: A design review — before this phase is marked done —
  confirms that adding the v1.6.0 Ollama/llama.cpp `llm` extras group and pipeline entries requires
  zero changes to the `PipelineMetadata` schema, the registry loader, or the extras-naming scheme
  introduced here (only additive YAML entries and a new `[project.optional-dependencies]` group).
  This is the concrete test of "modularity supports Ollama/HPC/remote-cloud going forward."

## Assumptions

- v1.4.4 is the baseline for behaviour-parity and reproducibility comparisons (superseding v1.4.2,
  which spec 003's original text used as its baseline before v1.4.3/v1.4.4 shipped).
- This phase does **not** implement the Ollama/llama.cpp backend, HTTPDispatcher, SlurmDispatcher,
  RemotePipelineProxy, entry-point third-party plugin discovery, or the `videoannotator-utils`
  package split. Those remain scheduled in
  [`docs/development/roadmap_v1.6.0.md`](../../docs/development/roadmap_v1.6.0.md) (plugin ecosystem
  + Ollama/llama.cpp local backend) and the v1.7.0–v2.0 arc (remote pipelines, HTTPDispatcher, VLM
  plugin, SlurmDispatcher/HPC, slim core) — see
  [`docs/development/roadmap_v1.7_to_v2.0.md`](../../docs/development/roadmap_v1.7_to_v2.0.md).
  This phase's job is narrower and concrete: make sure nothing built now has to be undone to reach
  those.
- **Security of remote/cloud dispatch** (the "secure" in "remote (secure) cloud services") is out
  of scope for this phase — there is no remote dispatch yet to secure. It is flagged here as a
  requirement that MUST be treated as first-class (not bolted on) when the v1.7.0 HTTPDispatcher /
  RemotePipelineProxy design begins: authentication, transport encryption, and credential handling
  for any pipeline that sends video/frame data off the local machine. The current v1.7–v2.0 roadmap
  draft mentions an `auth` config field on `RemotePipelineProxy` but does not yet specify a
  transport-security or secrets-handling model; that gap should be closed in the v1.7.0 spec, not
  deferred silently.
- The Ultralytics AGPL-3.0 licence isolation (person-tracking as its own plugin) remains deferred to
  v2.0 per spec 003; person-tracking stays a top-level `person` extras group in this phase.
- The model swaps in the v1.5.0 roadmap (faster-whisper, SigLIP-2, pyannote community-1, YOLO12,
  face-stack consolidation) are tracked separately and are not required by this spec's acceptance
  criteria, though they may land in the same release.
