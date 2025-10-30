# Pre-Commit Hooks Guide for VideoAnnotator

This guide helps you work smoothly with VideoAnnotator's pre-commit hooks, especially when working on the API layer which has strict type checking.

## Quick Start

### Before First Commit

```bash
# Install pre-commit hooks (one-time)
pre-commit install

# Optionally run on all files to see current state
pre-commit run --all-files
```

### Before Each Commit (Recommended)

Run these locally to catch issues before pre-commit:

```bash
# Option 1: Use Makefile shortcuts
make format          # Auto-format code
make type-check      # Check types (API modules)
make lint            # Lint check

# Option 2: Direct uv commands
uv run ruff format src/api/v1/
uv run mypy src/api/
uv run ruff check src/api/
```

## Understanding the Hook Pipeline

Pre-commit runs hooks in this order:

1. **trailing-whitespace** ✨ (auto-fixes)
2. **end-of-file-fixer** ✨ (auto-fixes)
3. **ruff** ⚠️ (lints, may auto-fix with --fix)
4. **ruff-format** ✨ (auto-formats code)
5. **mypy** ❌ (type checks, NO auto-fix)

**✨ = Auto-fixes and requires re-staging**
**⚠️ = May auto-fix**
**❌ = Reports errors only**

## Type Checking Strictness

### Strict Modules (Full Type Safety)

These modules have **strict mypy checking** enabled:

- `src/api/*` - All API code
- `tests/api/*` - API tests
- `tests/integration/test_api_integration.py`

Requirements for strict modules:
- All function parameters must have type hints
- All function return types must be declared
- No implicit optionals
- Complete function definitions only

### Relaxed Modules

These modules have **relaxed checking** (excluded from strict rules):

- `src/pipelines/*` - ML pipeline code (noisy with ML library types)
- `src/storage/`, `src/utils/`, `src/exporters/`, `src/schemas/`
- `examples/`, `scripts/`

## Common Type Issues & Solutions

### 1. Pydantic Config Classes (RUF012)

**Issue**: Mutable class attribute warning

```python
# ❌ Wrong
class MyModel(BaseModel):
    class Config:
        json_schema_extra = {"example": {...}}
```

**Solution**: Use `ClassVar` annotation

```python
# ✅ Correct
from typing import ClassVar, Any

class MyModel(BaseModel):
    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {"example": {...}}
```

### 2. Exception Subclasses

**Issue**: Type mismatch in exception attribute assignments

```python
# ❌ Wrong (mypy can't infer attribute types)
class MyException(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
```

**Solution**: Use inline type annotations

```python
# ✅ Correct
class MyException(Exception):
    def __init__(self, code: str, message: str):
        self.code: str = code
        self.message: str = message
```

### 3. Mixed-Type Dictionaries

**Issue**: Mypy infers dict type from first assignment

```python
# ❌ Wrong (mypy thinks detail values are all int after first assignment)
detail = {}
detail["count"] = 42        # mypy infers dict[str, int]
detail["name"] = "error"    # ❌ Type error: str incompatible with int
```

**Solution**: Declare explicit union type upfront

```python
# ✅ Correct
detail: dict[str, str | int] = {}
detail["count"] = 42
detail["name"] = "error"    # ✅ OK
```

### 4. Mypy Cache Staleness

**Symptom**: Errors persist after fixing code, or errors on seemingly correct lines

**Solution**: Clear mypy cache

```bash
# Clear all mypy caches
rm -rf .mypy_cache ~/.cache/pre-commit/mypy*

# Or use Makefile
make clean
```

## Pre-Commit Workflow

### Typical Success Flow

```bash
git add src/api/v1/my_feature.py
git commit -m "feat: Add my feature"
# ✅ All hooks pass
# → Commit succeeds
```

### When Hooks Auto-Fix Files

```bash
git add src/api/v1/my_feature.py
git commit -m "feat: Add my feature"
# ✨ trailing-whitespace: Fixed
# ✨ ruff-format: 1 file reformatted
# ❌ Commit rejected (files modified)

# Re-stage modified files
git add src/api/v1/my_feature.py
git commit -m "feat: Add my feature"
# ✅ All hooks pass
# → Commit succeeds
```

### When Mypy Fails

```bash
git add src/api/v1/my_feature.py
git commit -m "feat: Add my feature"
# ❌ mypy: Found 2 errors
# src/api/v1/my_feature.py:45: error: ...

# Option 1: Fix the issues (preferred)
# Edit the file to fix type errors
git add src/api/v1/my_feature.py
git commit -m "feat: Add my feature"

# Option 2: Clear cache if errors seem wrong
rm -rf .mypy_cache
git commit -m "feat: Add my feature"

# Option 3: Skip hooks (LAST RESORT - use sparingly)
git commit --no-verify -m "feat: Add my feature"
```

## Testing Type Safety Locally

### Check Specific File

```bash
uv run mypy src/api/v1/errors.py
```

### Check Entire API Module

```bash
uv run mypy src/api/
```

### Check With Explicit Config

```bash
uv run mypy --config-file=pyproject.toml src/api/
```

### Check What Pre-Commit Will See

```bash
pre-commit run mypy --files src/api/v1/errors.py
```

## API Module Checklist

Before committing new API modules, verify:

- [ ] All function parameters have type hints
- [ ] All function return types declared (or `-> None`)
- [ ] Pydantic Config classes use `ClassVar` for class attributes
- [ ] Exception classes have inline type annotations in `__init__`
- [ ] Dictionaries with mixed value types have explicit hints
- [ ] Necessary imports: `from typing import ClassVar, Any` etc.
- [ ] Ran `uv run mypy src/api/` locally and it passed
- [ ] Ran `uv run ruff format src/api/` to pre-format

## Troubleshooting

### "But the code looks correct!"

If mypy complains about seemingly correct code:

1. **Check strict mode settings** - API modules have stricter rules
2. **Clear cache** - Stale cache can show phantom errors
3. **Check inferred types** - Use `reveal_type(variable)` to debug
4. **Verify imports** - Missing `ClassVar`, `Any`, etc.

### "Pre-commit is too slow"

```bash
# Skip hooks for WIP commits (use sparingly)
git commit --no-verify -m "WIP: work in progress"

# Run only specific hooks
pre-commit run ruff --files src/api/v1/errors.py
```

### "I want to update pre-commit hook versions"

```bash
pre-commit autoupdate
git add .pre-commit-config.yaml
git commit -m "chore: Update pre-commit hooks"
```

## Configuration Reference

### Mypy Configuration (pyproject.toml)

```toml
[tool.mypy]
python_version = "3.12"
strict = false  # Default is relaxed
# ... base settings ...

[[tool.mypy.overrides]]
module = [
    "src.api.*",
    "tests.api.*",
]
check_untyped_defs = true          # Strict for API
disallow_incomplete_defs = true    # Strict for API
no_implicit_optional = true        # Strict for API
```

### Pre-Commit Mypy Hook

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.5.1
  hooks:
    - id: mypy
      args: [--config-file=pyproject.toml, --namespace-packages]
      additional_dependencies: [types-PyYAML, types-requests]
      exclude: ^(src/pipelines/|src/storage/|...)  # Relaxed modules
```

## Best Practices

1. **Run formatting before type checking** - Formatting issues are easier to fix
2. **Fix type errors incrementally** - Don't accumulate many type errors
3. **Use local checks before committing** - Faster feedback loop
4. **Clear cache when stuck** - Stale cache causes confusion
5. **Use --no-verify sparingly** - Only after confirming fixes are correct
6. **Update AGENTS.md** - Document new patterns you discover

## Additional Resources

- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pydantic Type Hints](https://docs.pydantic.dev/latest/concepts/types/)
- [PEP 526 - Variable Annotations](https://peps.python.org/pep-0526/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)

## Quick Reference Card

| Task | Command |
|------|---------|
| Format code | `make format` or `uv run ruff format src/api/` |
| Check types | `make type-check` or `uv run mypy src/api/` |
| Lint code | `make lint` or `uv run ruff check src/api/` |
| All checks | `make quality-check` |
| Clear cache | `make clean` or `rm -rf .mypy_cache` |
| Run pre-commit manually | `pre-commit run --all-files` |
| Skip hooks | `git commit --no-verify` (use sparingly) |
