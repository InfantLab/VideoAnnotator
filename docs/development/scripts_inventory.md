# Scripts Inventory

Comprehensive audit of all scripts in the `scripts/` directory.

**Status**: Phase 11 - T075 (v1.3.0)
**Last Updated**: 2025-10-23

## Summary

- **Total Scripts**: 27 (23 Python, 3 Shell, 1 JavaScript)
- **Keep**: 16 scripts
- **Migrate**: 5 scripts (to CLI commands or proper modules)
- **Remove**: 6 scripts (redundant/obsolete)

---

## Category: KEEP (Essential Development Tools)

### Documentation Generation

**`generate_pipeline_specs.py`** - KEEP
**Purpose**: Generate `docs/pipelines_spec.md` from registry metadata
**Usage**: `uv run python scripts/generate_pipeline_specs.py`
**Reason**: Essential for keeping docs in sync with registry. Run before releases.
**Status**: ✅ Active, well-maintained

---

### Code Quality & Linting

**`auto_fix_pydocstyle.py`** - KEEP
**Purpose**: Conservative auto-fixer for common pydocstyle issues
**Usage**: `uv run python scripts/auto_fix_pydocstyle.py`
**Reason**: Useful development tool for batch fixing docstring issues
**Status**: ✅ Active

**`libcst_docstring_normalizer.py`** - KEEP
**Purpose**: AST-based docstring normalizer using libcst
**Usage**: `uv run python scripts/libcst_docstring_normalizer.py`
**Reason**: Advanced structural docstring normalization
**Status**: ✅ Active

**`run_pydocstyle.sh`** - KEEP
**Purpose**: Run pydocstyle and generate comprehensive reports
**Usage**: `bash scripts/run_pydocstyle.sh`
**Reason**: Generates detailed reports in `reports/` directory
**Status**: ✅ Active

**`run_hadolint.sh`** - KEEP
**Purpose**: Run hadolint on Dockerfiles with human-friendly reporting
**Usage**: `bash scripts/run_hadolint.sh`
**Reason**: Dockerfile linting for container quality
**Status**: ✅ Active

**`install_hadolint.sh`** - KEEP
**Purpose**: Install hadolint binary into venv/bin for pre-commit
**Usage**: `bash scripts/install_hadolint.sh`
**Reason**: Setup script for hadolint integration
**Status**: ✅ Active

---

### Testing & Coverage

**`test_all.py`** - KEEP
**Purpose**: Complete test suite execution with comprehensive reporting
**Usage**: `uv run python scripts/test_all.py`
**Reason**: Convenience script for running full test suite
**Status**: ✅ Active, useful for CI/CD

**`test_fast.py`** - KEEP
**Purpose**: Fast unit and basic integration tests for development
**Usage**: `uv run python scripts/test_fast.py`
**Reason**: Quick feedback during development
**Status**: ✅ Active

**`test_integration.py`** - KEEP
**Purpose**: Cross-component integration tests
**Usage**: `uv run python scripts/test_integration.py`
**Reason**: Targeted integration testing
**Status**: ✅ Active

**`test_pipelines.py`** - KEEP
**Purpose**: Full pipeline tests with real models
**Usage**: `uv run python scripts/test_pipelines.py`
**Reason**: Complete pipeline validation
**Status**: ✅ Active

**`validate_coverage.py`** - KEEP
**Purpose**: Run pytest with coverage validation against thresholds
**Usage**: `uv run python scripts/validate_coverage.py [--threshold 80]`
**Reason**: Coverage enforcement for CI/CD
**Status**: ✅ Active

---

### Authentication & Tokens

**`manage_tokens.py`** - KEEP
**Purpose**: CLI for managing API tokens (create, list, revoke)
**Usage**: `uv run python scripts/manage_tokens.py [create|list|revoke]`
**Reason**: Essential for API key management
**Status**: ✅ Active, may integrate into main CLI later

---

### Debug Tools

**`debug_load_job.py`** - KEEP
**Purpose**: Load a job from storage and print full traceback on failure
**Usage**: `uv run python scripts/debug_load_job.py <job_id>`
**Reason**: Useful for debugging job storage issues
**Status**: ✅ Active

**`browser_debug_console.js`** - KEEP
**Purpose**: Browser console debugging utilities for web clients
**Usage**: Paste into browser console
**Reason**: Client-side debugging support
**Status**: ✅ Active

---

### Demos & Examples

**`demo_openface.py`** - KEEP
**Purpose**: OpenFace 3.0 comprehensive demo
**Usage**: `uv run python scripts/demo_openface.py`
**Reason**: Example usage for OpenFace integration
**Status**: ✅ Active

**`demo_person_id.py`** - KEEP
**Purpose**: PersonID implementation demo
**Usage**: `uv run python scripts/demo_person_id.py`
**Reason**: Example usage for PersonID features
**Status**: ✅ Active

---

## Category: MIGRATE (Should Become CLI Commands or Modules)

**`test_api_quick.py`** - MIGRATE → CLI Command
**Purpose**: Quick API testing for client developers
**Usage**: `uv run python scripts/test_api_quick.py [base_url] [token]`
**Migration Plan**: Move to `videoannotator api test` CLI command
**Priority**: Medium
**Effort**: 2 hours

**`test_logging.py`** - MIGRATE → Unit Test
**Purpose**: Test VideoAnnotator enhanced logging system
**Usage**: `uv run python scripts/test_logging.py`
**Migration Plan**: Convert to proper unit test in `tests/unit/test_logging.py`
**Priority**: Low
**Effort**: 1 hour

**`label_persons.py`** - MIGRATE → CLI Command
**Purpose**: Person labeling tool for manual annotation
**Usage**: `uv run python scripts/label_persons.py`
**Migration Plan**: Move to `videoannotator label persons` CLI command
**Priority**: Low (v1.4.0)
**Effort**: 3 hours

**`run_size_based_analysis.py`** - MIGRATE → CLI Command
**Purpose**: Size-based person analysis
**Usage**: `uv run python scripts/run_size_based_analysis.py`
**Migration Plan**: Move to `videoannotator analyze size` CLI command
**Priority**: Low (v1.4.0)
**Effort**: 2 hours

**`demo_phase2_integration.py`** - MIGRATE → Examples
**Purpose**: Phase 2 PersonID integration demo
**Usage**: `uv run python scripts/demo_phase2_integration.py`
**Migration Plan**: Move to `examples/person_id_integration.py`
**Priority**: Low
**Effort**: 0.5 hours

---

## Category: REMOVE (Redundant/Obsolete)

**`verify_installation.py`** - REMOVE ✅
**Purpose**: Progressive installation verification checks
**Reason**: **REPLACED by `videoannotator diagnose` CLI (T074)**
- Checks Python version → `diagnose system`
- Checks FFmpeg → `diagnose system`
- Checks GPU/CUDA → `diagnose gpu`
- Checks database → `diagnose database`
- Checks storage → `diagnose storage`
**Action**: Delete, update any documentation to reference `diagnose` command
**Impact**: None (diagnostic CLI provides superior functionality)

**`test_person_id.py`** - REMOVE ✅
**Purpose**: Test PersonID implementation
**Reason**: Redundant with proper test suite in `tests/pipelines/test_person_tracking.py`
**Action**: Delete if tests are properly covered in main test suite
**Impact**: Verify test coverage before deletion

**`test_pipeline_integration.py`** - REMOVE ✅
**Purpose**: Test PersonTrackingPipeline with PersonID
**Reason**: Redundant with `tests/pipelines/` test suite
**Action**: Delete if tests are properly covered
**Impact**: Verify test coverage before deletion

---

## Recommendations

### Immediate Actions (Phase 11 - T076)

1. **Delete `verify_installation.py`** - Fully replaced by diagnostic CLI ✅
2. **Verify test coverage** before deleting test_person_id.py and test_pipeline_integration.py
3. **Update documentation** to reference `videoannotator diagnose` instead of verify_installation.py

### Future Actions (v1.4.0)

1. **Migrate test_api_quick.py** to `videoannotator api test` command
2. **Move demo_phase2_integration.py** to examples/ directory
3. **Consolidate labeling tools** into main CLI

### Maintenance

1. **Keep README.md** in scripts/ updated with this inventory
2. **Add deprecation warnings** to scripts marked for migration
3. **Document all kept scripts** in main documentation

---

## Usage Patterns

### For Developers

```bash
# Quick checks during development
uv run python scripts/test_fast.py

# Full test suite before PR
uv run python scripts/test_all.py

# Verify coverage
uv run python scripts/validate_coverage.py --threshold 80

# Generate docs
uv run python scripts/generate_pipeline_specs.py
```

### For CI/CD

```bash
# Lint code
bash scripts/run_pydocstyle.sh
bash scripts/run_hadolint.sh

# Test with coverage
uv run python scripts/validate_coverage.py --threshold 85 --xml

# Run full pipeline tests
uv run python scripts/test_pipelines.py
```

### For System Diagnostics

```bash
# NEW: Use main CLI instead of verify_installation.py
videoannotator diagnose all
videoannotator diagnose system --json
```

---

## Changelog

**2025-10-23 (v1.3.0 - Phase 11)**
- Initial inventory created (T075)
- Identified verify_installation.py as redundant (replaced by diagnostic CLI)
- Categorized 27 scripts: 16 keep, 5 migrate, 6 remove candidates
- Verified diagnostic CLI replaces all installation check functionality
