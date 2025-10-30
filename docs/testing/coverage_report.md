# Test Coverage Validation

This document describes the test coverage requirements, validation process, and best practices for VideoAnnotator.

## Coverage Requirements

VideoAnnotator maintains high test coverage standards to ensure code quality and reliability for JOSS publication and production use.

### Module-Specific Thresholds

| Module | Threshold | Rationale |
|--------|-----------|-----------|
| `src/api` | **90%** | API endpoints are critical user-facing components with predictable behavior |
| `src/registry` | **90%** | Registry system is core infrastructure for pipeline discovery |
| `src/validation` | **90%** | Validation logic must be thoroughly tested for correctness |
| `src/database` | **85%** | Database operations should have high coverage for data integrity |
| `src/storage` | **85%** | Storage layer handles persistent state and file operations |
| `src/worker` | **85%** | Worker execution must be reliable for job processing |
| `src/pipelines` | **80%** | Pipeline modules include ML code with some environment-dependent paths |
| `src/batch` | **80%** | Batch processing includes various execution scenarios |
| `src/utils` | **80%** | Utility functions should be well-tested |
| `src/exporters` | **75%** | Exporters have complex I/O patterns and format variations |

### Global Threshold

- **Overall Project**: **80%** minimum coverage across all source code

### Excluded from Coverage

The following directories and files are excluded from coverage requirements:

- `tests/` - Test code itself
- `scripts/` - Utility scripts
- `examples/` - Example code
- `__pycache__/` - Python bytecode
- `site-packages/` - Third-party dependencies

## Running Coverage Validation

### Quick Start

```bash
# Run coverage validation with default settings
uv run python scripts/validate_coverage.py

# Generate HTML report for detailed analysis
uv run python scripts/validate_coverage.py --html

# Verbose output with both HTML and XML reports
uv run python scripts/validate_coverage.py -v --html --xml
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Show detailed coverage output including files below threshold |
| `--html` | Generate HTML coverage report (saved to `htmlcov/`) |
| `--xml` | Generate XML coverage report for CI (saved to `coverage.xml`) |
| `--fail-under PERCENT` | Override global threshold (default: 80) |
| `--skip-tests` | Analyze existing `.coverage` file without running tests |
| `--module MODULE` | Check coverage for specific module only (e.g., `src/api`) |

### Examples

```bash
# Custom global threshold
uv run python scripts/validate_coverage.py --fail-under 85

# Check API module only
uv run python scripts/validate_coverage.py --module src/api

# CI mode: XML report with verbose output
uv run python scripts/validate_coverage.py -v --xml --fail-under 80

# Quick check of existing coverage
uv run python scripts/validate_coverage.py --skip-tests
```

## Interpreting Results

### Exit Codes

- **0**: All coverage thresholds met ✅
- **1**: Coverage thresholds not met ❌
- **2**: Test execution failed ⚠️

### Sample Output

```
======================================================================
COVERAGE VALIDATION REPORT
======================================================================

[OK] Total Coverage: 84.23%
[OK] Global Threshold: 80.00%

[OK] Total coverage meets global threshold

----------------------------------------------------------------------
MODULE-SPECIFIC COVERAGE
----------------------------------------------------------------------

[OK] src/api: 92.15% (threshold: 90.00%)
[OK] src/pipelines: 83.67% (threshold: 80.00%)
[OK] src/database: 87.42% (threshold: 85.00%)
[FAIL] src/exporters: 72.18% (threshold: 75.00%)

======================================================================
[FAIL] Some coverage thresholds not met
======================================================================

[SUGGESTION] To improve coverage:
  1. Run: uv run pytest --cov=src --cov-report=html
  2. Open: htmlcov/index.html
  3. Focus on modules/files with low coverage
  4. Add unit tests for uncovered code paths
```

## Improving Coverage

### Step 1: Identify Low Coverage Areas

Generate an HTML report to see exactly which lines are not covered:

```bash
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html  # or xdg-open on Linux
```

The HTML report provides:
- **File-level coverage** - Which files have low coverage
- **Line-level highlighting** - Which specific lines are not executed
- **Branch coverage** - Which conditional branches are untested

### Step 2: Analyze Uncovered Code

Common reasons for uncovered code:

| Reason | Action |
|--------|--------|
| **Error handling paths** | Add tests for exception scenarios |
| **Edge cases** | Add boundary condition tests |
| **Platform-specific code** | Add conditional tests or exclude with `# pragma: no cover` |
| **Dead code** | Remove or refactor unused code |
| **Interactive code** | Exclude with `# pragma: no cover` if not testable |

### Step 3: Write Targeted Tests

Focus on:

1. **Unit tests** for isolated functions/classes
2. **Integration tests** for cross-component interactions
3. **Error path tests** for exception handling
4. **Edge case tests** for boundary conditions

### Step 4: Re-validate

```bash
uv run python scripts/validate_coverage.py --html
```

## Coverage in CI/CD

### GitHub Actions Integration

```yaml
- name: Run tests with coverage
  run: |
    uv run python scripts/validate_coverage.py --xml --fail-under 80

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
    fail_ci_if_error: true
```

### Pre-commit Hook (Optional)

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: coverage-check
      name: Check test coverage
      entry: uv run python scripts/validate_coverage.py
      language: system
      pass_filenames: false
      stages: [push]
```

## Configuration

### pyproject.toml

Coverage is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/examples/*", "*/scripts/*"]
branch = true

[tool.coverage.report]
precision = 2
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]

[tool.coverage.html]
directory = "htmlcov"
```

### Updating Thresholds

To modify module-specific thresholds, edit `scripts/validate_coverage.py`:

```python
THRESHOLDS = {
    "src/api": 90.0,
    "src/pipelines": 80.0,
    # Add or modify thresholds here
}
```

## Best Practices

### DO ✅

- **Write tests alongside new code** - Maintain coverage as you develop
- **Test error paths** - Don't just test happy paths
- **Use fixtures for setup** - Keep tests DRY and maintainable
- **Mock external dependencies** - Test in isolation
- **Run coverage locally** - Catch issues before CI

### DON'T ❌

- **Chase 100% coverage blindly** - Focus on meaningful tests
- **Skip tests for "untestable" code** - Usually there's a way
- **Test implementation details** - Test behavior, not internals
- **Write tests just to increase numbers** - Write meaningful tests
- **Ignore failing coverage** - Fix or understand why it's failing

## Troubleshooting

### Coverage Not Collected

**Problem**: `.coverage` file not created or empty

**Solution**:
```bash
# Ensure pytest-cov is installed
uv sync

# Run with explicit coverage
uv run pytest --cov=src --cov-report=term
```

### Module Coverage Shows 0%

**Problem**: Module shows 0% coverage despite having tests

**Solution**:
- Check that module path in `THRESHOLDS` matches actual path
- Verify tests are in correct location (e.g., `tests/api/` for `src/api/`)
- Ensure imports are correct in tests

### Tests Pass but Coverage Fails

**Problem**: All tests pass but coverage validation fails

**Solution**:
- Review the HTML report to identify specific uncovered lines
- Add tests for untested code paths
- Consider if code should be excluded with `# pragma: no cover`

### HTML Report Not Generated

**Problem**: `htmlcov/` directory not created

**Solution**:
```bash
# Generate HTML explicitly
uv run pytest --cov=src --cov-report=html

# Or use validation script
uv run python scripts/validate_coverage.py --html
```

## JOSS Requirements

For Journal of Open Source Software (JOSS) publication, we maintain:

- ✅ **>80% overall coverage** - Demonstrates thorough testing
- ✅ **Module-specific thresholds** - Ensures critical components are well-tested
- ✅ **Automated validation** - Reproducible coverage checks
- ✅ **HTML reports** - Easy review for editors/reviewers
- ✅ **Documentation** - Clear coverage policies and practices

JOSS reviewers can validate coverage by running:

```bash
# After installation
uv run python scripts/validate_coverage.py --html --verbose
```

## Additional Resources

- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](./README.md)
- [JOSS Review Guidelines](https://joss.readthedocs.io/en/latest/review_criteria.html)

## Questions?

If you have questions about coverage requirements or validation:

1. Check this documentation first
2. Review the HTML coverage report for details
3. Open an issue on GitHub with specific questions
4. Include validation output and coverage reports when reporting issues
