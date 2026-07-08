# Feature Specification: Modular Pipeline Architecture

**Feature Branch**: `v1.5-modularity`
**Created**: 2026-05-06
**Status**: Draft
**Input**: User description: "Refactor VideoAnnotator's package structure to support modular installation, optional pipeline plugins, and out-of-process pipeline dispatch. Move heavy ML deps (torch, ultralytics, pyannote, whisper, transformers, etc.) from required [project] dependencies to per-pipeline [project.optional-dependencies] extras (face, audio, scene, person, embedding). Strip eager pipeline imports from pipelines/__init__.py. Replace the hardcoded LEGACY_MAPPINGS dict in the registry with metadata-driven loading via per-pipeline YAML module_path + requires_extras keys, unioned with importlib.metadata.entry_points discovery for future third-party plugins. Consolidate the duplicate job-execution paths (api/job_processor.py and batch/batch_orchestrator._process_single_job) into a single function. Introduce a Dispatcher ABC with LocalThreadDispatcher preserving current behaviour, as the seam for future HTTPDispatcher/SlurmDispatcher. Migrate cross-cutting utils/{person_identity,automatic_labeling,model_loader,size_based_person_analysis} to a sibling videoannotator-utils package in the monorepo. Drop the numpy<2.0 pin. Slim core install must run FastAPI/CLI without any heavy ML dep; videoannotator[all] reproduces v1.4.2 behaviour exactly."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Slim, targeted install for a single research need (Priority: P1)

A researcher who only needs scene labelling on their lecture-recording corpus installs the toolkit with just the scene-labelling pipeline. They do not download multi-gigabyte models or libraries unrelated to their work, and the install fits comfortably on their laptop.

**Why this priority**: The current default install footprint (~30 GB Docker image / multi-gigabyte pip install) is the largest single barrier to adoption flagged during JOSS review. Most researchers use only a subset of the toolkit's pipelines but pay the cost of all of them. Solving this directly removes the most-cited friction.

**Independent Test**: A user runs `pip install videoannotator[scene]` on a clean environment, executes scene labelling on a sample video, and verifies that the resulting install footprint is materially smaller than today's full install while producing equivalent scene-labelling output.

**Acceptance Scenarios**:

1. **Given** a clean Python environment, **When** the user installs the toolkit with only the scene extras group, **Then** the install completes without downloading audio, face, or person-tracking dependencies, and the toolkit runs scene labelling successfully.
2. **Given** the toolkit installed with only the scene extras, **When** the user lists available pipelines, **Then** only scene-labelling pipelines appear and the others are absent without raising errors.
3. **Given** the toolkit installed with only the scene extras, **When** the user attempts to invoke a face-emotion pipeline, **Then** the user receives a clear, actionable message explaining how to add the required extras, with no Python traceback.
4. **Given** an existing install with the scene extras, **When** the user adds the face extras with `pip install videoannotator[face]`, **Then** both pipeline families are available and previously usable functionality continues to work.

---

### User Story 2 - Backward-compatible upgrade for existing studies (Priority: P1)

A researcher who is mid-study on v1.4.2 upgrades to v1.5 and continues running their existing analysis pipeline configuration files unchanged. Their outputs are identical and citable as part of the same study; their existing scripts and notebooks keep working.

**Why this priority**: Studies in behavioural research run for months. A toolkit that breaks running analyses mid-study fails Principle V (Backward Compatibility by Default) of the project constitution. This is also P1 because without it, the modularity work is a hostile change rather than a foundation.

**Independent Test**: Run the v1.4.2 acceptance test fixtures against the v1.5 install with the meta-extra group enabled. Verify that pipeline outputs match v1.4.2 byte-for-byte (or to a documented numerical tolerance for non-deterministic pipelines), and that no configuration file requires modification.

**Acceptance Scenarios**:

1. **Given** an existing v1.4.2 user with a working configuration file and a corpus of in-flight study videos, **When** they upgrade to v1.5 by installing the toolkit with the meta-extra (`pip install videoannotator[all]`), **Then** their existing CLI invocations and configuration files produce equivalent annotations to v1.4.2 with no modifications.
2. **Given** a published study citing VideoAnnotator v1.4.2 outputs, **When** a reviewer reproduces the analysis with v1.5 `[all]`, **Then** the outputs match within documented tolerances.
3. **Given** a user upgrading from v1.4.2 without explicitly choosing the `[all]` meta-extra, **When** they run their existing scripts, **Then** they receive a clear migration message directing them to the appropriate install command, rather than silent failure.

---

### User Story 3 - Third-party pipeline as a separate package (Priority: P3)

A researcher in another lab develops a custom pipeline (for example, an infant-cry detector specific to their study). They publish it as a standalone Python package on PyPI. Other users install it alongside VideoAnnotator and the pipeline appears in the toolkit's registry without anyone modifying VideoAnnotator's source code.

**Why this priority**: P3 because it is not required for the v1.5 release to provide value to existing users (P1 and P2 deliver that). However, the architectural seam for plugins must be in place for v1.5 so that v1.6+ plugin extractions and the future remote/HPC/VLM plugins all rely on a single discovery mechanism.

**Independent Test**: Author a minimal third-party plugin package (≤ 50 lines of code) declaring a single dummy pipeline. Install it into a v1.5 environment alongside VideoAnnotator core. Verify that the dummy pipeline appears in `GET /api/v1/pipelines`, can be invoked through the standard API/CLI, and produces output without core modifications.

**Acceptance Scenarios**:

1. **Given** a v1.5 install, **When** a separately published plugin package is installed via pip, **Then** the plugin's pipeline(s) appear in the registry alongside the bundled pipelines.
2. **Given** an installed third-party plugin, **When** the user invokes its pipeline via the toolkit's standard CLI or API, **Then** the pipeline executes and produces annotations using the same job-execution path as bundled pipelines.
3. **Given** an installed third-party plugin that declares its required extras, **When** the user lacks those extras, **Then** the plugin is shown in the registry but flagged as unavailable with an actionable install hint.

---

### Edge Cases

- A user requests a pipeline whose extras are not installed: the toolkit MUST guide the user to the right install command rather than fail with a Python import traceback.
- A user upgrades from v1.4.2 with a configuration file that references a pipeline now demoted from the default install (e.g. LAION Empathic-Insight): the toolkit MUST detect this and direct the user to install the relevant extras group.
- Two installed extras groups depend on different versions of the same underlying ML library: the toolkit MUST either resolve to compatible versions at install time, or surface the conflict at install time rather than at runtime.
- A third-party plugin author declares an extras group that collides with a name reserved for an official pipeline: the registry MUST detect the collision and produce a deterministic error.
- A model cache from a v1.4.2 install exists on disk: v1.5 MUST reuse it rather than re-download.
- The user's Python environment uses NumPy 2.x: the toolkit MUST run correctly without requiring users to downgrade NumPy.
- A pipeline's extras are installed but its model weights are not yet cached: first run MUST complete the download with progress indication, not fail silently.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The toolkit MUST install successfully on a clean Python environment without pulling in any heavy ML library when no extras are specified.
- **FR-002**: Each pipeline modality (face, audio, scene, person, embedding-shared) MUST be installable as a named extras group, with names reserved and documented.
- **FR-003**: A meta-extras group (`all`) MUST exist that pulls every official pipeline group, providing a one-step install equivalent to v1.4.2's default footprint.
- **FR-004**: Adding extras to an existing install MUST be additive; previously installed pipelines MUST continue to function without re-installation.
- **FR-005**: When a pipeline's extras are not installed, the registry MUST omit that pipeline from public listings rather than crash; user-facing surfaces (CLI, API, error messages) MUST direct the user to the install command for that pipeline.
- **FR-006**: External plugin packages installed alongside the toolkit MUST be discovered automatically via standard Python entry-point mechanisms, with no edits to core source code required to enable them.
- **FR-007**: The toolkit MUST run a single canonical code path for executing a job; the CLI batch mode and the API job-submission mode MUST both invoke the same per-job logic with the same defaults.
- **FR-008**: Job dispatch MUST be abstracted behind an interface so that future deployment targets (remote workers, HPC schedulers, alternative concurrency models) can be added without modifying pipeline implementations or the per-job logic.
- **FR-009**: Output artifacts (annotation files, provenance metadata, native-format files) produced by v1.5 with the meta-extras group MUST match those produced by v1.4.2 byte-identically for deterministic pipelines, and within documented numerical tolerances for non-deterministic pipelines.
- **FR-010**: Cross-cutting utility code (person-identity tracking, automatic labelling helpers, model-loading helpers, size-based person analysis) MUST be importable as a separately versioned package, independently of any specific pipeline.
- **FR-011**: The toolkit MUST run on the current major Python numerical stack (NumPy 2.x), and its full test suite MUST pass without requiring a downgrade.
- **FR-012**: The user-facing pipeline registry surface (CLI listing, API endpoint, documentation index) MUST clearly distinguish between (a) pipelines available now, (b) pipelines whose extras are installable, and (c) pipelines not present in this install at all.
- **FR-013**: Installation documentation MUST include a per-pipeline install matrix indexable by user use case (e.g. "I want only scene labelling", "I want everything", "I want a slim API server").
- **FR-014**: When a user upgrades from v1.4.2 to v1.5 and runs an existing config file that requires extras now no longer in the default install (e.g. LAION emotion), the toolkit MUST detect the gap and produce an actionable migration message rather than failing on the first import.
- **FR-015**: Cached model weights from a previous install MUST be reused by v1.5 without forced re-download, regardless of which extras combination is installed.

### Key Entities

- **Pipeline metadata record**: The declarative description of a pipeline family — its public name, output schema, required extras group, and discovery identifier. Travels with the pipeline's package, not in the core registry source.
- **Extras group**: A named, documented bundle of dependencies installable via pip's extras syntax. Each group corresponds to one pipeline modality or to a meta-collection. Names are reserved at the project level to prevent third-party plugins from claiming them.
- **Plugin package**: An external Python package that contributes one or more pipelines to the registry without modifying core. Discoverable through standard entry-point mechanisms.
- **Cross-cutting utilities package**: A separately versioned package containing logic shared between pipeline families (identity tracking, automatic labelling, model loading, size analysis). Pinned at known versions by core and by each official plugin.
- **Job dispatch context**: An abstraction representing the place a single job runs (current default: local thread pool). Encapsulates the runtime decisions (concurrency, retry, queueing) so that the per-job logic stays unchanged when new dispatch targets are added in later releases.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user installing only one pipeline family completes `install + first successful run on a sample video` in under 5 minutes from a clean environment, on a representative laptop with broadband internet.
- **SC-002**: The default Docker image footprint (no extras) is reduced by at least 80 % compared to the v1.4.2 baseline.
- **SC-003**: 100 % of v1.4.2 acceptance-test fixtures pass against a v1.5 `[all]` install without any modification to user-side configuration files, CLI invocations, or downstream analysis code.
- **SC-004**: A toy third-party plugin (≤ 50 lines of Python plus a `pyproject.toml`) can be authored, installed, and successfully invoked via the toolkit's standard interfaces with zero changes to VideoAnnotator core code.
- **SC-005**: 100 % of pipelines available in the v1.4.2 release are installable in v1.5 (either bundled in `[all]` or available as documented extras), so no functionality is lost in the refactor.
- **SC-006**: A user reading the installation documentation can identify the right install command for their use case within one paragraph of reading.
- **SC-007**: When a user requests an unavailable pipeline, the resulting error message includes the exact pip install command needed to enable it, in 100 % of cases.
- **SC-008**: The toolkit's full automated test suite passes on the current major Python numerical stack (NumPy 2.x) on Linux, macOS, and Windows.
- **SC-009**: A reproduction of any published v1.4.2 study (using the same configuration and inputs) under v1.5 `[all]` produces outputs matching v1.4.2 within documented tolerances, supporting Principle V (Backward Compatibility) and Principle III (Provenance & Reproducibility).

## Assumptions

- The monorepo organisation decided in the v1.5 roadmap is locked in; the cross-cutting utilities package and any future plugin packages live as subdirectories of this repository, with one release cadence per version.
- v1.4.2 is the baseline against which "behaviour parity" is measured; reproducibility and migration apply specifically to that release.
- The current major Python numerical stack (NumPy 2.x) is mature enough that the existing pipeline ecosystem can run against it without forks; any blocker is fixable within the v1.5 timeline.
- The Ultralytics AGPL-3.0 licence isolation (separate plugin) is deferred to v2.0; for v1.5, person-tracking remains a top-level extras group within the main package.
- The cross-cutting utilities package is published to the same PyPI release cadence as core in v1.5; later versions may decouple them.
- LAION Empathic-Insight (face and voice emotion) remains optionally installable in v1.5 but is no longer part of the default install; this aligns with the model-currency decision in the v1.5 roadmap (see [`docs/development/roadmap_v1.5.0.md`](../../docs/development/roadmap_v1.5.0.md)).
- The model swaps documented in the v1.5 roadmap (faster-whisper, SigLIP-2, pyannote community-1, YOLO12, face-stack consolidation) ride alongside this refactor in the same release but are tracked as a separate concern; this specification covers the architectural changes only.
- The remote-pipeline proxy, HTTP dispatcher, HPC dispatcher, and VLM plugin documented in the v1.5–v2.0 roadmap series are NOT part of this specification's scope; this specification provides the architectural seam (the dispatcher abstraction and entry-point discovery) on which they will be built in v1.7+.
