# VideoAnnotator Test Suite Organization Plan

## Executive Summary
**Current State**: 28 test files with ~695 tests across different categories
**Goal**: Reorganize into logical, maintainable structure with clear execution tiers

## Current Analysis (August 19, 2025)

### Test File Categories
- **Unit Tests**: 6 files (batch, storage, recovery, types)
- **Integration Tests**: 3 files (batch_integration, phase2_integration, integration_simple)  
- **Modern Pipeline Tests**: 3 files (*_modern.py)
- **Real/Live Tests**: 4 files (*_real.py)
- **Legacy Tests**: 12 files (older pipeline tests, individual components)

### Key Issues Identified
1. **Scattered organization**: Related tests in different files
2. **Import conflicts**: Some tests fail to collect due to dependencies
3. **Mixed test types**: Unit, integration, and end-to-end tests intermingled
4. **Unclear execution strategy**: No clear fast/slow/experimental tiers

## Proposed New Structure

```
tests/
├── unit/                     # Fast, isolated tests (<30 seconds total)
│   ├── batch/
│   │   ├── test_types.py     # BatchJob, PipelineResult, etc.
│   │   ├── test_orchestrator.py
│   │   ├── test_recovery.py
│   │   └── test_progress_tracker.py
│   ├── storage/
│   │   ├── test_backends.py
│   │   └── test_validation.py
│   ├── pipelines/
│   │   ├── test_base_pipeline.py
│   │   ├── test_configuration.py
│   │   └── test_schemas.py
│   └── utils/
│       ├── test_person_identity.py
│       ├── test_automatic_labeling.py
│       └── test_size_analysis.py
│
├── integration/              # Cross-component tests (<5 minutes total)
│   ├── test_batch_storage.py
│   ├── test_pipeline_coordination.py
│   ├── test_person_identity_integration.py
│   └── test_export_formats.py
│
├── pipelines/               # Full pipeline tests (<15 minutes total)
│   ├── test_person_tracking.py
│   ├── test_face_analysis.py
│   ├── test_audio_processing.py
│   ├── test_scene_detection.py
│   └── test_openface3.py
│
├── performance/             # Benchmarking and regression tests
│   ├── test_batch_performance.py
│   ├── test_pipeline_benchmarks.py
│   └── test_memory_usage.py
│
├── experimental/            # Prototype and research tests
│   ├── test_new_features.py
│   └── test_research_pipelines.py
│
└── fixtures/                # Shared test data and utilities
    ├── __init__.py
    ├── mock_data.py
    ├── test_videos/
    ├── expected_outputs/
    └── conftest.py          # pytest fixtures
```

## Implementation Strategy

### Phase 1: Foundation (Week 1)
1. **Create new directory structure**
2. **Move unit tests** (batch, storage, utils) - these are most stable
3. **Create shared fixtures** and conftest.py files
4. **Update import paths** and ensure tests still pass

### Phase 2: Pipeline Reorganization (Week 2)  
1. **Consolidate pipeline tests** from multiple files into single focused files
2. **Separate unit vs integration vs full pipeline tests**
3. **Create performance benchmarking structure**
4. **Remove duplicate/obsolete tests**

### Phase 3: Execution Tiers (Week 3)
1. **Create execution scripts** for different test tiers
2. **Configure CI/CD integration** with appropriate test selection
3. **Add pytest markers** for test categorization
4. **Performance and timeout configuration**

### Phase 4: Documentation and Maintenance (Week 4)
1. **Update test documentation** and contribution guidelines
2. **Create test maintenance procedures**
3. **Establish test coverage targets** per category
4. **Monitor and refine organization**

## Test Execution Tiers

### Tier 1: Fast Unit Tests (Target: <30 seconds)
```bash
pytest tests/unit/ -m "not slow" --maxfail=5
```
- Core data structures
- Configuration validation  
- Individual component logic
- Mock-heavy, no heavy dependencies

### Tier 2: Integration Tests (Target: <5 minutes)
```bash
pytest tests/integration/ -m "not slow" 
```
- Cross-component interactions
- Storage system integration
- Lightweight pipeline coordination
- Smart mocking of heavy ML models

### Tier 3: Full Pipeline Tests (Target: <15 minutes)
```bash
pytest tests/pipelines/ --timeout=300
```
- Complete pipeline processing
- Real model loading (with caching)
- End-to-end workflows
- GPU acceleration testing

### Tier 4: Performance & Experimental (As needed)
```bash
pytest tests/performance/ tests/experimental/ --benchmark-only
```
- Performance regression testing
- Research and prototype validation
- Long-running stability tests

## Pytest Configuration

### pyproject.toml markers
```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (cross-component)", 
    "pipeline: Full pipeline tests (slow)",
    "performance: Performance and benchmark tests",
    "experimental: Experimental and research tests",
    "slow: Tests that take >10 seconds",
    "gpu: Tests requiring GPU acceleration",
    "real_models: Tests using real ML models"
]
```

### Execution Scripts
```bash
# scripts/test_fast.sh - Development workflow
pytest tests/unit/ tests/integration/ -m "not slow" --maxfail=3

# scripts/test_full.sh - Pre-commit validation  
pytest tests/unit/ tests/integration/ tests/pipelines/ --timeout=600

# scripts/test_ci.sh - CI/CD pipeline
pytest tests/unit/ tests/integration/ -m "not gpu and not real_models"
```

## Migration Plan

### File Mapping
- `test_batch_*.py` → `tests/unit/batch/`
- `test_storage_*.py` → `tests/unit/storage/`  
- `test_*_pipeline_modern.py` → `tests/pipelines/`
- `test_*_integration.py` → `tests/integration/`
- `test_*_real.py` → Merge into appropriate tier based on what they test

### Test Consolidation
- **Combine duplicate tests**: Many files test similar functionality
- **Remove placeholder tests**: Convert skipped tests to real implementations or remove
- **Update imports**: Ensure all tests can import dependencies correctly
- **Add missing tests**: Fill gaps identified during reorganization

## Success Metrics

### Immediate (Phase 1)
- [ ] New directory structure created
- [ ] 100% of unit tests moved and passing
- [ ] Shared fixtures working across test files

### Short-term (Phase 2-3)
- [ ] <300 total tests (down from 695 through consolidation)
- [ ] All 4 execution tiers working reliably
- [ ] <30 second fast test execution
- [ ] CI/CD integration functional

### Long-term (Phase 4+)
- [ ] >90% test pass rate for each tier
- [ ] Clear test maintenance procedures
- [ ] New contributor onboarding simplified
- [ ] Test execution time regression monitoring

## Risk Mitigation

### Import Dependencies
- **Problem**: Some tests fail to import due to heavy ML dependencies
- **Solution**: Smart mocking strategy, dependency isolation per tier

### Test Data Management  
- **Problem**: Tests need video files and expected outputs
- **Solution**: Lightweight test fixtures, shared test data repository

### CI/CD Performance
- **Problem**: Full test suite too slow for continuous integration
- **Solution**: Tiered execution with appropriate test selection per context

This organization plan transforms the test suite from a collection of scattered files into a **maintainable, efficient testing system** that supports both rapid development and comprehensive validation.