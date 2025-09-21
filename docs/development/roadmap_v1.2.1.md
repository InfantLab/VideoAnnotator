# VideoAnnotator v1.2.1 Roadmap - Polish, Pipeline Specification & Documentation

## 🎯 **Release Overview**
**Target Release**: 1-2 weeks after v1.2.0  
**Focus**: Quick wins, documentation updates, example modernization, and foundational pipeline specification & discovery layer.  
**Scope**: Non-breaking improvements; establish the single-source-of-truth (SSOT) pipeline registry needed for future (v1.3.0) advanced features.

## 🔄 Scope Adjustment (Compared to Original Draft)
This release now explicitly includes ONLY the following foundational and quick-win pillars:
1. Pipeline Specification & Discovery (registry + metadata + basic generation)
2. Documentation & Example Modernization
3. CI / Logging / CLI UX polish

All advanced AI, multi-modal correlation, plugin architecture, advanced analytics, streaming, RBAC/auth, and enterprise feature work are deferred to v1.3.0. See "Deferred to v1.3.0" section below.

## 🧱 Foundational Principle for v1.2.1
Introduce a maintainable, incremental Pipeline Registry that is minimal today but extensible for v1.3.0 (capabilities, resources, modality fusion, plugin injection points). Keep implementation intentionally lean (YAML + lightweight loader + adapters) to avoid blocking the polish timeline.

## 📚 **Documentation & Examples Modernization**

### **Priority 1: Pipeline Specification & Documentation System (Foundational Slice)** 🎯
**Goal (v1.2.1 scope)**: Introduce a minimal, functional single source of truth (SSOT) for pipeline metadata powering: CLI listing, API `/pipelines`, and generated Markdown docs. Advanced metadata (multi-modal correlation, resource modeling, plugin contracts) is intentionally deferred to v1.3.0.

#### **Unified Pipeline Registry System (Minimal v1.2.1 Deliverable)**
- [x] **Create Minimal Pipeline Registry**
  - Load YAML metadata files under `src/registry/metadata/`
  - Provide `list_pipelines()` + `get_pipeline(name)` functions
  - Return core fields: `name`, `display_name`, `description`, `category`, `config_schema`, `outputs`, `examples` (optional), `version` (schema version)
  - Implement graceful fallback if metadata missing (warn, exclude, or mark experimental)
- [x] **Lightweight Metadata Schema (YAML)**
  ```yaml
  # Example: src/registry/metadata/person_tracking.yaml
  name: person_tracking
  display_name: Person Tracking & Pose
  category: tracking
  description: Track persons with YOLO11 + ByteTrack and generate COCO pose annotations.
  outputs:
    - format: COCO
      types: [person_detection, keypoints]
  config_schema:
    model:
      type: string
      default: models/yolo/yolo11n-pose.pt
      description: YOLO pose model path or alias
    conf_threshold:
      type: float
      default: 0.4
      description: Detection confidence threshold
  examples:
    - cli: videoannotator job submit demo.mp4 --pipelines person_tracking
      description: Submit tracking job
  version: 1
  ```
- [x] **Validation Layer (Basic)**: Ensure required keys present; warn if `outputs` missing.
- [x] **Extensibility Hooks (Document-only)**: Note future fields (resources, modalities, capabilities) reserved for v1.3.0.

#### **Auto-Generated Documentation (Lean Slice)**
- [x] **CLI Pipeline Listing**: `videoannotator pipelines --detailed` surfaces registry fields
- [x] **API Endpoint Backed by Registry**: Replace mock `/api/v1/pipelines` implementation with dynamic registry data
- [x] **Markdown Generation Script**: `scripts/generate_pipeline_docs.py` creates `docs/usage/pipeline_specs.md` (fails CI if drift)
- [x] **Config Validation (Basic)**: `videoannotator config --validate` references `config_schema` types (presence + type check only in 1.2.1)
- [ ] **Schema Version Tagging**: Include `schema_version` in generated markdown for future migration tracking (DEFERRED to v1.3.0)

#### **Complete Pipeline Coverage**
- [x] **Audit Current Pipelines** - Document ALL available pipelines
  - scene_detection (PySceneDetect + CLIP)
  - person_tracking (YOLO11 + ByteTrack) 
  - face_analysis (OpenFace 3.0, LAION Face, OpenCV variants)
  - audio_processing (Whisper + pyannote)
  - Any experimental or development pipelines

- [x] **Pipeline Categories** - Organize pipelines into logical groups
  - **Detection**: person_tracking, face_detection
  - **Analysis**: face_analysis, audio_processing, emotion_detection
  - **Segmentation**: scene_detection, temporal_segmentation
  - **Preprocessing**: video_preprocessing, audio_extraction

#### **Implementation Strategy (Incremental)**
Phase 1 (Day 1–2): Registry loader + 2 metadata files + API swap.

Phase 2 (Day 3): Add markdown generator + CLI integration + drift CI check.

Phase 3 (Day 4): Basic config validation + tests (registry integrity, endpoint parity, doc generation).

Phase 4 (Polish): Add examples linkage + schema version annotation + warnings for missing fields.

### **Priority 2: CLI Documentation & Examples Updates**
- [x] **examples/README.md** - Update all CLI usage patterns to new `videoannotator` syntax (COMPLETED)
- [x] **docs/usage/GETTING_STARTED.md** - Verify all CLI examples are current (COMPLETED)
- [x] **docs/usage/demo_commands.md** - Update command examples throughout (COMPLETED)

### **Priority 2: Example Script Modernization**
#### **Update Existing Examples**
- [ ] **examples/basic_video_processing.py** - Add API integration alternative (IN PROGRESS / PARTIAL)
- [ ] **examples/batch_processing.py** - Show API-based batch processing (PENDING)
- [x] **examples/test_individual_pipelines.py** - Demonstrate CLI pipeline testing (COMPLETED)
- [ ] **examples/custom_pipeline_config.py** - Show new config validation (PENDING)

#### **Add New API-First Examples**
- [x] **examples/api_job_submission.py** - Complete API workflow example (COMPLETED)
- [ ] **examples/api_batch_processing.py** - Multi-video API processing (PENDING)
- [x] **examples/cli_workflow_example.py** - Modern CLI usage patterns (COMPLETED)

### **Priority 3: Configuration & Validation Updates**
- [x] **configs/README.md** - Update for v1.2.0 API server usage (COMPLETED)
- [ ] **Legacy config cleanup** - Document which configs are legacy vs. modern (PENDING)

## 🐛 **Minor Bug Fixes**

### **CI/CD & Testing Issues**
- [ ] **🚨 GitHub Actions CI/CD Failure** - Investigate and fix CI/CD pipeline failure from v1.2.0 release (PENDING)
  - **Issue**: GitHub Actions run failed with git operation errors (exit code 128)
  - **URL**: https://github.com/InfantLab/VideoAnnotator/actions/runs/17261547713
  - **Impact**: 19 errors across multiple OS/Python version combinations
  - **Investigation needed**: Git repository configuration, dependency compatibility, test environment setup
  - **Priority**: High - affects automated testing and release validation

### **Logging & Diagnostics**
- [x] **(Reclassify) Logging directory creation** - CONFIRM & document existing behavior (already auto-created)
- [ ] **Improve CUDA detection** - More accurate GPU availability reporting (DEFERRED)
- [x] **Enhanced health checks (Foundational)** - Add: pipeline count, registry load status, GPU availability (COMPLETED)

### **CLI Improvements** 
- [x] **Better error messages** - Standardized helper + consistent prefixing (COMPLETED)
- [x] **Config validation feedback** - Use registry schema field types in messages (BASIC COMPLETED)
- [ ] **Progress indicators (Optional)** - Lightweight status output (DEFERRED)
- [x] **Optional JSON output mode** - `--json` for scripting (COMPLETED: pipelines command)

## 🔧 **Developer Experience**

### **Testing Updates**
- [ ] **Integration test documentation** - How to run and interpret tests (PENDING)
- [ ] **Example test coverage** - Ensure all examples have basic tests (PARTIAL)
- [ ] **CLI testing** - Add tests for new CLI commands (IN PROGRESS)

### **Code Quality**
- [x] **Remove deprecated imports** - Clean up old import patterns (COMPLETED via absolute import normalization)
- [ ] **Update type hints** - Ensure all new CLI code is properly typed (PENDING)
- [ ] **Docstring updates** - Complete docstring coverage for new functions (PENDING)

## 📊 **Performance & Polish**

### **API Enhancements (Minimal in 1.2.1)**
- [x] **Error response standardization (Foundational)** - Introduce simple error envelope pattern (code, message) (COMPLETED BASELINE)
- [x] **Pipeline endpoint dynamic data** - Registry-backed (COMPLETED)
- [x] **Health endpoint enrichment** - See diagnostics above (COMPLETED)
- [ ] (Deferred) Response time profiling → v1.3.0
- [ ] (Deferred) Advanced request validation → v1.3.0

### **CLI Polish**
- [ ] **Output formatting consistency** - Shared formatter helper (PARTIAL)
- [ ] **Configuration precedence doc** - Document order (PENDING)
- [ ] (Deferred) Color support → v1.3.0

## 📋 **Success Criteria**

### **Pipeline Specification System (v1.2.1 Slice)**
- ✅ **SSOT Introduced** - Registry exists & used by CLI + API + docs
- ✅ **Drift Detection** - CI fails on markdown mismatch
- ✅ **Coverage Threshold** - ≥ 3 core pipelines (person_tracking, scene_detection, audio_processing) documented
- ✅ **Extensibility Reserved** - Future fields documented (resources, modalities, capabilities)
- ✅ **Validation Minimum** - Basic type/presence validation supported

### **Documentation**
- ✅ All CLI examples use modern `videoannotator` syntax
- ✅ New users can follow examples without confusion
- ✅ Legacy vs. modern patterns are clearly documented

### **Developer Experience**
- ✅ All CLI commands have helpful error messages
- ✅ Configuration validation provides clear feedback
- ✅ Examples run without modification on fresh installs

### **Code Quality (Right-Sized for v1.2.1)**
- ✅ Examples free of deprecated patterns
- ✅ New registry + generator code typed & documented
- ✅ Tests for: registry load, endpoint parity, doc generation
- (Informational) Overall coverage target unchanged; registry code must be ≥90% local coverage

## ⚡ **Fast-Track Items (1 week)**
These can be completed quickly for immediate release:

1. **🚨 CI/CD Pipeline Fix** (4-6 hours) - **PRIORITY**: Fix GitHub Actions failure from v1.2.0
2. **examples/README.md CLI syntax updates** (2 hours)
3. **Basic logging directory fix** (1 hour)  
4. **CLI error message improvements** (3 hours)
5. **Add 1-2 new API examples** (4 hours)

## ♻️ **Deferred to v1.3.0 (Out of 1.2.1 Scope)**
Advanced items intentionally postponed:
- Multi-modal correlation modeling (temporal relationships, behavioral analysis)
- Active learning & quality assessment pipelines
- Confidence scoring & ensemble systems
- Plugin framework & extension sandboxing
- Resource/capability scheduling & adaptive pipeline selection
- Streaming/WebRTC + real-time low-latency pathways
- Advanced request validation + performance profiling automation
- Colorized + interactive CLI UI / progress bars
- Rich config templating & precedence engine
- RBAC, SSO, multi-tenancy, audit logging
- Analytics dashboards & usage metrics

All appear in the v1.3.0 roadmap under appropriate phases.

## 🎯 **Release Timeline**

### **Updated Release Timeline (Lean)**
Week 1:
- Registry + 2 metadata files + API endpoint swap
- Examples/README CLI syntax modernization
- CI pipeline fix & logging confirmation

Week 2:
- Markdown generator + drift check
- Add config validation + enriched health endpoint
- Third metadata file + tests + polish (error helper, JSON mode)

**Total engineering effort**: ~3 focused dev days spread over 2 weeks (parallelizable)

## 🔗 Mapping Table (v1.2.1 → v1.3.0)
| v1.2.1 Deliverable | Enables v1.3.0 Feature |
| ------------------ | ----------------------- |
| Registry (minimal) | Plugin system, adaptive processing |
| Basic config schema | Advanced validation, templates |
| Markdown generator | Documentation automation & marketplace entries |
| Health enrichment | Performance profiling & monitoring foundation |
| Error envelope pattern | Standardized API + GraphQL alignment |

---
*Last Updated*: September 2025