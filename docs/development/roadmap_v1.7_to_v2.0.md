# VideoAnnotator v1.7 — v2.0 Roadmap

## Theme: Remote Pipelines, HPC Dispatch, Multimodal VLM, Slim Core

This document covers the arc from v1.7 through v2.0 — the period when VideoAnnotator stops being a single-machine monolith and becomes a toolkit that can dispatch work to GPU servers, HPC clusters, and remote multimodal models.

The detail here is intentionally lighter than [v1.5](roadmap_v1.5.0.md) and [v1.6](roadmap_v1.6.0.md). These versions are 6-12+ months out; specifics will firm up as v1.5/v1.6 land and field experience informs the design.

**Prerequisite:** v1.5.0's extras-based install and metadata-driven registry loading (see [`specs/004-extras-based-install/spec.md`](../../specs/004-extras-based-install/spec.md)), plus v1.6.0's entry-point plugin discovery, `Dispatcher` ABC seam, and Ollama/llama.cpp local-LLM backend (which establishes the `backends` metadata field's non-local-execution pattern this document extends to remote/HPC targets).

---

## v1.7 — Remote pipelines and the first VLM plugin

**Theme:** Pipelines no longer have to run in-process. Adds the proxy layer that v2.0 needs and ships the first plugin that uses it.

**Target:** Q2-Q3 2027
**Estimated duration:** ~10 weeks

### Deliverables

- [ ] **`RemotePipelineProxy(BasePipeline)`** in core. Reads `endpoint`, `model`, `auth`, sampling params from config. In `process()`, serialises args as JSON, POSTs to the endpoint, parses the response back to `list[dict]`. The pipeline class is locally installable and locally instantiable; only the heavy compute lives elsewhere.
- [ ] **`HTTPDispatcher`** as a third `Dispatcher` implementation alongside `LocalThreadDispatcher`. Submits jobs to a remote VideoAnnotator instance over the existing FastAPI v1. Useful for "I have a GPU server in the next room and a laptop in front of me" workflows.
- [ ] **Transport security and credential handling for `HTTPDispatcher`/`RemotePipelineProxy`.** TLS-only endpoints by default (reject plaintext `http://` unless explicitly opted into for local-network use), and a defined credential-storage model for the `auth` config field (e.g. environment variable or OS keychain reference, never a literal secret in a checked-in config file). This was flagged as an open gap in earlier drafts of this roadmap and must be a first-class requirement of the v1.7.0 spec, not an afterthought — see [`specs/004-extras-based-install/spec.md`](../../specs/004-extras-based-install/spec.md) Assumptions section.
- [ ] **Reference pipeline-runner microserver.** A 30-50 line FastAPI app at `packages/videoannotator-pipeline-runner/` that exposes a single `BasePipeline` instance over HTTP using the same JSON contract as `RemotePipelineProxy`. Production-deployable as a Docker image.
- [ ] **`videoannotator-vlm` plugin (first VLM pipeline).** New sibling package. Default backend: Qwen2.5-VL-7B via Ollama (`qwen2.5vl:7b`). Optional backend: vLLM for batch throughput. Pipeline config:
  ```yaml
  model_id: ollama:qwen2.5vl:7b
  backend: ollama  # or vllm, lmdeploy
  prompt_template: |
    Describe what is happening in this video segment.
    Focus on: {focus_areas}
    Respond with JSON matching: {output_schema}
  output_schema:
    type: object
    properties:
      activity: {type: string}
      participants: {type: integer}
      confidence: {type: number}
  frames_per_segment: 8
  frame_sampling: uniform
  temperature: 0.0
  max_new_tokens: 512
  ```
  Output: WebVTT cues with parsed JSON sidecars (mirrors v1.5's native-format doctrine). Source: [survey notes](#vlm-survey-summary).
- [ ] **Reproducibility plumbing.** Extend `create_annotation_metadata` with `revision_sha` (HF model commit), `quantisation` (Q4_K_M / Q8_0 / F16 for Ollama), `prompt_sha256`, `sampling_params` echoed verbatim. VLM output without these is research-grade unusable.

### Out of scope for v1.7

- HPC dispatcher (deferred to v1.8 unless someone needs it sooner).
- New specialist pipelines (pose, hand, motion) — deferred to v1.8.
- Splitting face/scene/person plugins — that's v2.0.

---

## v1.8 — HPC dispatch and new pipeline categories

**Theme:** Lab-scale to cluster-scale. Adds Slurm and the pipelines researchers have been asking for that aren't VLMs.

**Target:** Q4 2027
**Estimated duration:** ~8-10 weeks

### Deliverables

- [ ] **`SlurmDispatcher`.** Fourth `Dispatcher` implementation. `submit(job)` serialises the `BatchJob` to JSON in shared storage, generates an `sbatch` script that runs `videoannotator process-one --job-file <path>` (CLI subcommand from v1.5), calls `subprocess.run(["sbatch", ...])`, returns a `Future` that resolves by polling `sacct`/`squeue`. Configurable partition, account, time, memory, GPU count via `~/.videoannotator/slurm.yaml`.
- [ ] **`PBSDispatcher` and `SGEDispatcher` skeletons.** Same shape as Slurm; lower priority since fewer behavioural-research clusters use these. Ship if someone needs them; otherwise document the extension point.
- [ ] **`videoannotator-pose` plugin.** RTMPose-m / RTMW from MMPose for whole-body keypoints (133 points incl. hands+face). Apache-2.0. Real-time on GPU. Wraps as a standard `BasePipeline` with COCO output. Strong fit for infant/parent posture analysis. Source: [github.com/open-mmlab/mmpose](https://github.com/open-mmlab/mmpose).
- [ ] **`videoannotator-hand` plugin.** MediaPipe Hand Landmarker. 21 keypoints/hand on CPU, trivial CC-licensed dependency. Useful for caregiver-infant gesture coding. Source: [github.com/google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe).
- [ ] **`videoannotator-motion` plugin (optional).** SEA-RAFT optical flow as an opt-in for "how much movement happened in this clip" features. BSD-3. Source: [github.com/princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT).

### Out of scope for v1.8

- Distributed multi-node within a single job (still single-machine per pipeline call).
- Audio events (BEATs) — defer unless requested.
- Lip reading — research-grade, no maintained Python package.

---

## v2.0 — Slim core, all plugins extracted

**Theme:** The core install is a thin orchestrator. Every pipeline lives in its own plugin package. Default Docker image targets ~2 GB.

**Target:** 2028
**Estimated duration:** ~12 weeks

### Deliverables

- [ ] **Extract `videoannotator-face`** (DeepFace + OpenFace 3) as a sibling package.
- [ ] **Extract `videoannotator-scene`** (PySceneDetect + open_clip/SigLIP-2) as a sibling package.
- [ ] **Extract `videoannotator-person`** (Ultralytics YOLO + ByteTrack/BoT-SORT) as a sibling package.
- [ ] **Extract `videoannotator-laion`** (Empathic-Insight, if upstream still exists by 2028 — otherwise drop entirely).
- [ ] **Slim core image.** Base: `python:3.12-slim` + FastAPI + SQLAlchemy + numpy/pandas/opencv-headless + the registry/dispatcher/storage layers. Target: ~2 GB. No PyTorch in the core image.
- [ ] **Per-plugin Docker images.** Each plugin published as `videoannotator/<plugin>:v2.0` extending the slim core. Compose-friendly: `docker-compose up videoannotator-core videoannotator-audio videoannotator-vlm` for a multi-pipeline deployment.
- [ ] **`videoannotator[all]` meta-package.** Depends on every official plugin. Reproduces v1.4.x install footprint for users who want everything.
- [ ] **Plugin authoring guide.** Documentation and a cookiecutter template so third-party labs can publish their own pipelines without forking.
- [ ] **Stable plugin contract.** `BasePipeline`, the YAML metadata schema, the entry-point group name, and `videoannotator-utils` are versioned semver-strict from v2.0 onwards.

### Out of scope for v2.0

- Distributed multi-node orchestration within a single pipeline call (researchers who need this should run one job per node).
- Cloud provider integrations (AWS/Azure/GCP) — orthogonal; the HTTPDispatcher already covers "pipeline runs somewhere else".
- Real-time streaming — different architectural problem, not on this arc.

---

## Cross-cutting concerns for v1.7-v2.0

### Reproducibility

VLM outputs vary with sampling, model version, prompt phrasing. Standard practice for research papers: pin the HF revision SHA, set `temperature=0` and a fixed seed, store the exact prompt verbatim, report inference framework + version + quantisation. The reproducibility plumbing landed in v1.7 (above) makes this enforceable: every annotation carries the full reproducibility envelope.

For paper-grade reproducibility, recommend the **vLLM or transformers backend with `torch.use_deterministic_algorithms(True)`** rather than Ollama — Ollama/llama.cpp can produce non-deterministic outputs across hardware due to floating-point reductions even at temperature 0.

### Hallucination and content-policy risk

VLMs confabulate counts above ~5 people in frame, invent emotions for ambiguous expressions, and invent text on low-resolution faces. For child / vulnerable-population footage:

- **Llama-3.2-Vision and Gemma-3 intermittently refuse to describe minors** (vendor RLHF policies). **Hard blocker for behavioural research.**
- **Qwen2.5-VL, Qwen3-VL, InternVL3.5, MiniCPM-o have substantially weaker refusal behaviour around minors** and are practically usable. The maintainer remains responsible for IRB/ethics approval; document this prominently.
- **Pixtral has no built-in moderation**, which is a feature here (but document the lack of safety rails for users who want them).

### Licensing matrix at v2.0

| Component | Licence | Notes |
|---|---|---|
| `videoannotator` (core) | MIT | Unchanged |
| `videoannotator-utils` | MIT | Unchanged |
| `videoannotator-audio` | MIT | Wraps faster-whisper (MIT) + pyannote weights (CC-BY-4.0, gated) |
| `videoannotator-face` | MIT | Wraps DeepFace (MIT) + OpenFace 3 (Boost Software Licence) |
| `videoannotator-scene` | MIT | Wraps PySceneDetect (BSD-3) + open_clip + SigLIP-2 (Apache-2.0) |
| `videoannotator-person` | MIT | Wraps Ultralytics YOLO (AGPL-3.0 or Enterprise) — flag clearly |
| `videoannotator-pose` | MIT | Wraps MMPose (Apache-2.0) |
| `videoannotator-hand` | MIT | Wraps MediaPipe (Apache-2.0) |
| `videoannotator-motion` | MIT | Wraps SEA-RAFT (BSD-3) |
| `videoannotator-vlm` | MIT | Wraps Qwen2.5-VL/Qwen3-VL (Apache-2.0); InternVL3.5 (Apache-2.0); MiniCPM-o (Apache-2.0) |
| `videoannotator-laion` | MIT | Wraps LAION Empathic-Insight (CC-BY-4.0) — drop if upstream gone |

The Ultralytics AGPL-3.0 issue is the only real licensing wrinkle. Ship `videoannotator-person` as a separate AGPL-3.0 plugin so the rest of the toolkit stays MIT-clean.

### VLM survey summary

[Full survey on file with maintainer.] Headline recommendation for v1.7: **Qwen2.5-VL-7B-Instruct via Ollama as default, vLLM as power-user backend**. Apache-2.0, native video, 32K context, fits 24 GB VRAM, weak refusal behaviour around children. Runner-up: **InternVL3.5-8B via lmdeploy** (different vision encoder, useful as a second rater for reproducibility cross-checks). Future upgrade: Qwen3-VL-8B (256K native context, October 2025 weights) once Ollama publishes a first-party tag.

Pipelines a VLM can **replace**: scene labelling, coarse action description, video-segment summaries, person/object enumeration, simple emotion description. Pipelines a VLM **augments but doesn't replace**: face emotion (use as independent rater), behaviour description on top of person tracks. Pipelines a VLM **never replaces**: pose keypoints, AUs, diarisation timestamps, frame-accurate scene cuts.

---

## Risks and open questions

- **Ollama video support.** Ollama currently has no video parameter on `/api/generate`; video is sent as multi-image at extracted frames. If Ollama adds native video before v1.7 lands, the plugin can use it; otherwise the multi-image pattern is fine.
- **Cluster heterogeneity.** Slurm clusters vary wildly (partitions, GRES syntax, container runtimes). The `SlurmDispatcher` ships a sensible default but expect lab-specific config files in practice.
- **Plugin release coupling.** Once we have 7+ sibling packages, version-resolution between core and plugins becomes a real maintenance task. Mitigation: lock-step semver-major releases of core + all official plugins until v3.0 if needed.
- **Long-tail of specialist models.** Field will keep moving; v2.0+ should include a clear cadence ("review specialist defaults each minor release") rather than letting them drift.

---

**Last updated:** 2026-07-18
**Authors:** Caspar Addyman (with Claude Code review)
