# VideoAnnotator v1.2.1 Roadmap - Polish & Documentation

## 🎯 **Release Overview**
**Target Release**: 1-2 weeks after v1.2.0  
**Focus**: Documentation updates, example modernization, and minor polish  
**Scope**: Non-breaking improvements and user experience enhancements

## 📚 **Documentation & Examples Modernization**

### **Priority 1: Comprehensive Pipeline Documentation System** 🎯
**Goal**: Create a unified, complete, and consistent pipeline documentation system that automatically generates documentation for CLI, API, and Markdown docs from a single source.

#### **Unified Pipeline Registry System**
- [ ] **Create Pipeline Registry** - Single source of truth for all pipeline information
  - Pipeline names, descriptions, and capabilities
  - Input/output schemas and examples
  - Configuration parameters and validation
  - Dependencies and requirements
  - Performance characteristics
  
- [ ] **Pipeline Metadata Schema** - Structured data format for complete pipeline information
  ```python
  # Example: src/schemas/pipeline_registry.py
  @dataclass
  class PipelineMetadata:
      name: str
      display_name: str  
      description: str
      category: str  # "detection", "tracking", "analysis", "preprocessing"
      inputs: Dict[str, Any]
      outputs: Dict[str, Any]
      config_schema: Dict[str, Any]
      requirements: List[str]  # GPU, CPU, external dependencies
      examples: List[Dict[str, Any]]
      performance_notes: str
  ```

#### **Auto-Generated Documentation**
- [ ] **CLI Help Generation** - Generate `--help` text from pipeline registry
  ```bash
  uv run videoannotator pipelines list --detailed  # Generated from registry
  uv run videoannotator pipeline info scene_detection  # Complete pipeline docs
  ```

- [ ] **API Documentation** - Auto-generate FastAPI schema descriptions from registry
  ```python
  # Auto-generated endpoint docs with examples
  @router.get("/pipelines/{pipeline_name}")
  def get_pipeline_info(pipeline_name: str):
      # Documentation auto-generated from PipelineMetadata
  ```

- [ ] **Markdown Documentation** - Generate pipeline_specs.md from registry
  ```python
  # Script to generate docs/usage/pipeline_specs.md from registry data
  # Ensures CLI, API, and docs are always in sync
  ```

#### **Complete Pipeline Coverage**
- [ ] **Audit Current Pipelines** - Document ALL available pipelines
  - scene_detection (PySceneDetect + CLIP)
  - person_tracking (YOLO11 + ByteTrack) 
  - face_analysis (OpenFace 3.0, LAION Face, OpenCV variants)
  - audio_processing (Whisper + pyannote)
  - Any experimental or development pipelines

- [ ] **Pipeline Categories** - Organize pipelines into logical groups
  - **Detection**: person_tracking, face_detection
  - **Analysis**: face_analysis, audio_processing, emotion_detection
  - **Segmentation**: scene_detection, temporal_segmentation
  - **Preprocessing**: video_preprocessing, audio_extraction

#### **Implementation Strategy**
```python
# Phase 1: Create registry system
src/registry/
├── pipeline_registry.py      # Core registry class
├── metadata/                 # Individual pipeline metadata files
│   ├── scene_detection.yaml
│   ├── person_tracking.yaml
│   └── face_analysis.yaml
└── generators/               # Auto-documentation generators
    ├── cli_help_generator.py
    ├── api_doc_generator.py
    └── markdown_generator.py

# Phase 2: Integration
# Update CLI commands to use registry
# Update API endpoints to use registry  
# Generate markdown docs from registry

# Phase 3: Validation
# CI checks to ensure docs stay in sync
# Tests to validate registry completeness
```

### **Priority 2: CLI Documentation Updates**
- [ ] **examples/README.md** - Update all CLI usage patterns to new `videoannotator` syntax
- [ ] **docs/usage/GETTING_STARTED.md** - Verify all CLI examples are current  
- [ ] **docs/usage/demo_commands.md** - Update command examples throughout

### **Priority 2: Example Script Modernization**
#### **Update Existing Examples**
- [ ] **examples/basic_video_processing.py** - Add API integration alternative
- [ ] **examples/batch_processing.py** - Show API-based batch processing
- [ ] **examples/test_individual_pipelines.py** - Demonstrate CLI pipeline testing
- [ ] **examples/custom_pipeline_config.py** - Show new config validation

#### **Add New API-First Examples**
- [ ] **examples/api_job_submission.py** - Complete API workflow example
- [ ] **examples/api_batch_processing.py** - Multi-video API processing
- [ ] **examples/cli_workflow_example.py** - Modern CLI usage patterns

### **Priority 3: Configuration Updates**
- [ ] **configs/README.md** - Update for v1.2.0 API server usage
- [ ] **Legacy config cleanup** - Document which configs are legacy vs. modern

## 🐛 **Minor Bug Fixes**

### **Logging & Diagnostics**
- [ ] **Fix logging directory creation** - API server should create logs/ directory
- [ ] **Improve CUDA detection** - More accurate GPU availability reporting
- [ ] **Enhanced health checks** - Better system diagnostics in health endpoints

### **CLI Improvements** 
- [ ] **Better error messages** - More user-friendly CLI error reporting
- [ ] **Config validation feedback** - Clearer config file validation messages
- [ ] **Progress indicators** - Visual feedback for long-running CLI operations

## 🔧 **Developer Experience**

### **Testing Updates**
- [ ] **Integration test documentation** - How to run and interpret tests
- [ ] **Example test coverage** - Ensure all examples have basic tests
- [ ] **CLI testing** - Add tests for new CLI commands

### **Code Quality**
- [ ] **Remove deprecated imports** - Clean up old import patterns
- [ ] **Update type hints** - Ensure all new CLI code is properly typed
- [ ] **Docstring updates** - Complete docstring coverage for new functions

## 📊 **Performance & Polish**

### **API Enhancements**
- [ ] **Response time optimization** - Profile and optimize slow endpoints
- [ ] **Error response standardization** - Consistent error format across all endpoints
- [ ] **Request validation improvements** - Better input validation messages

### **CLI Polish**
- [ ] **Output formatting** - Consistent formatting across CLI commands
- [ ] **Color support** - Add optional color output for better UX
- [ ] **Configuration precedence** - Clear config file vs. CLI argument handling

## 📋 **Success Criteria**

### **Pipeline Documentation System**
- ✅ **Single Source of Truth** - All pipeline info comes from registry
- ✅ **Auto-Generated Consistency** - CLI help, API docs, and Markdown all match
- ✅ **Complete Coverage** - Every available pipeline is fully documented
- ✅ **User-Friendly** - Clear examples and use cases for each pipeline
- ✅ **Maintainable** - New pipelines automatically get consistent documentation

### **Documentation**
- ✅ All CLI examples use modern `videoannotator` syntax
- ✅ New users can follow examples without confusion
- ✅ Legacy vs. modern patterns are clearly documented

### **Developer Experience**
- ✅ All CLI commands have helpful error messages
- ✅ Configuration validation provides clear feedback
- ✅ Examples run without modification on fresh installs

### **Code Quality**
- ✅ No deprecated code patterns in examples
- ✅ Test coverage maintained above 90%
- ✅ All new functions have proper documentation

## ⚡ **Fast-Track Items (1 week)**
These can be completed quickly for immediate release:

1. **examples/README.md CLI syntax updates** (2 hours)
2. **Basic logging directory fix** (1 hour)  
3. **CLI error message improvements** (3 hours)
4. **Add 1-2 new API examples** (4 hours)

## 🎯 **Pipeline Documentation System (Priority)**
This comprehensive system should be implemented as the cornerstone of v1.2.1:

### **Phase 1: Registry Foundation** (3-4 days)
- Create pipeline registry infrastructure
- Define metadata schema for all pipeline information
- Implement registry loading and validation system

### **Phase 2: Documentation Generation** (2-3 days) 
- Build CLI help text generator from registry
- Create API documentation auto-generator
- Implement Markdown documentation generator

### **Phase 3: Integration & Testing** (2-3 days)
- Integrate registry with existing CLI commands
- Update API endpoints to use registry data
- Add tests to ensure documentation consistency

## 🎯 **Release Timeline**

### **Week 1: Core Updates**
- Documentation syntax updates
- Basic example modernization
- Logging fixes

### **Week 2: Polish & Testing**
- New API examples
- CLI improvements
- Final testing and validation

**Total effort**: ~2-3 developer days spread over 2 weeks