# Upgrading to VideoAnnotator v1.3.0

This guide helps you upgrade from VideoAnnotator v1.2.x to v1.3.0, which includes a major refactoring to follow Python packaging best practices.

## What Changed

### Directory Structure (src Layout)

The source code has been reorganized to follow the **src layout** pattern, which is the modern Python packaging standard:

**Before (v1.2.x):**
```
src/
├── __init__.py
├── api/
├── pipelines/
├── storage/
└── ...
```

**After (v1.3.0):**
```
src/
└── videoannotator/
    ├── __init__.py
    ├── api/
    ├── pipelines/
    ├── storage/
    └── ...
```

### Import Changes

All imports now use **relative imports** within the package and support **namespace imports** for external users.

## Migration Guide

### Step 1: Reinstall the Package

If you have VideoAnnotator installed in development mode, reinstall it:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

This ensures Python recognizes the new package structure.

### Step 2: Update Your Import Statements

#### **For External Users (Applications Using VideoAnnotator)**

**Old way (v1.2.x) - No longer works:**
```python
# These imports will fail in v1.3.0
from api.database import get_storage_backend
from pipelines.scene_detection import SceneDetectionPipeline
from registry.pipeline_registry import get_registry
```

**New way (v1.3.0) - Use videoannotator namespace:**
```python
# Method 1: Import from videoannotator package
from videoannotator.api.database import get_storage_backend
from videoannotator.pipelines.scene_detection import SceneDetectionPipeline
from videoannotator.registry.pipeline_registry import get_registry

# Method 2: Access modules via namespace
import videoannotator
api = videoannotator.api
registry = videoannotator.registry
storage = videoannotator.storage
```

#### **For Internal Development (VideoAnnotator Contributors)**

If you're working on VideoAnnotator itself, all imports within the codebase use **relative imports**:

```python
# Inside src/videoannotator/api/v1/jobs.py
from ..database import get_storage_backend  # Import from parent
from ...batch.types import BatchJob         # Import from grandparent
from .exceptions import JobNotFoundException # Import from same directory
```

### Step 3: Update CLI Usage

The CLI remains unchanged - no changes needed:

```bash
# All these commands still work exactly the same
videoannotator --version
videoannotator job submit video.mp4
videoannotator server --port 18011
```

### Step 4: Update API Server Usage

The API server usage is unchanged:

```python
# Still works the same
from videoannotator.api.main import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=18011)
```

## Common Import Patterns

### Pattern 1: Import Specific Functions/Classes

```python
# Import directly from the full path
from videoannotator.api.database import check_database_health
from videoannotator.registry.pipeline_registry import get_registry
from videoannotator.storage.base import StorageBackend
from videoannotator.validation.validator import ConfigValidator
```

### Pattern 2: Import Modules via Namespace

```python
# Import the package, then access modules
import videoannotator

# Access modules dynamically (uses __getattr__)
api_module = videoannotator.api
pipelines = videoannotator.pipelines
registry = videoannotator.registry

# Then use them
from videoannotator.api.database import get_storage_backend
```

### Pattern 3: Version Information

```python
# Version info is available at package level
import videoannotator

print(videoannotator.__version__)
print(videoannotator.__author__)
print(videoannotator.get_version_info())
```

## Benefits of the New Structure

### 1. **Better Test Isolation**
The src layout prevents accidentally importing from the source directory instead of the installed package, ensuring tests run against the installed code.

### 2. **Industry Standard**
Follows PEP 517/518 recommendations and matches the structure used by major Python projects (pytest, setuptools, pip).

### 3. **Cleaner Namespace**
The `videoannotator.*` namespace makes it clear where imports come from and prevents naming conflicts.

### 4. **Future-Proof**
Prepares the codebase for advanced features like namespace packages and plugin systems.

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'api'`

**Cause:** You're using old import style from v1.2.x

**Solution:** Update imports to use the `videoannotator.` prefix:

```python
# Old (v1.2.x)
from api.database import get_storage_backend

# New (v1.3.0)
from videoannotator.api.database import get_storage_backend
```

### Issue: `ModuleNotFoundError: No module named 'videoannotator'`

**Cause:** Package not installed properly

**Solution:** Reinstall in development mode:

```bash
uv sync
# or
pip install -e .
```

### Issue: Imports work in Python but not in tests

**Cause:** The test environment might not have the package installed

**Solution:** Always run tests with `uv run`:

```bash
# Correct way
uv run pytest tests/

# Incorrect (may use wrong environment)
pytest tests/
```

## Migration Checklist

Use this checklist to ensure a smooth upgrade:

- [ ] **Reinstall package**: `uv sync` or `pip install -e .`
- [ ] **Update import statements**: Add `videoannotator.` prefix to all imports
- [ ] **Test your code**: Run your application to verify imports work
- [ ] **Update documentation**: Update any code examples in your docs
- [ ] **Update CI/CD**: Ensure automated tests use `uv run` for test execution
- [ ] **Review custom scripts**: Update any scripts that import VideoAnnotator

## Estimated Migration Time

- **Small projects** (< 10 imports): **5-10 minutes**
- **Medium projects** (10-50 imports): **15-30 minutes**
- **Large projects** (> 50 imports): **30-60 minutes**

Most of the time is spent on find-and-replace operations. The actual changes are mechanical and straightforward.

## Automated Migration Script

For projects with many imports, you can use this simple script to update them automatically:

```python
#!/usr/bin/env python3
"""Migrate imports from v1.2.x to v1.3.0 style."""

import re
from pathlib import Path

# Modules that need videoannotator prefix
MODULES = [
    "api", "auth", "batch", "database", "exporters", "pipelines",
    "registry", "schemas", "storage", "utils", "validation",
    "visualization", "worker", "config", "main", "cli"
]

def migrate_file(file_path: Path) -> bool:
    """Migrate imports in a single file."""
    content = file_path.read_text()
    original = content

    for module in MODULES:
        # Pattern: from MODULE import X
        pattern = rf'^from {module}(\.[a-zA-Z0-9_.]+)? import '
        replacement = rf'from videoannotator.{module}\1 import '
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if content != original:
        file_path.write_text(content)
        return True
    return False

def main():
    """Migrate all Python files in the current directory."""
    changed = []
    for py_file in Path(".").rglob("*.py"):
        if "venv" in str(py_file) or ".venv" in str(py_file):
            continue  # Skip virtual environments
        if migrate_file(py_file):
            changed.append(py_file)
            print(f"Migrated: {py_file}")

    print(f"\nTotal files migrated: {len(changed)}")

if __name__ == "__main__":
    main()
```

Save this as `migrate_imports.py` and run it in your project directory:

```bash
python migrate_imports.py
```

**Note:** Always review the changes and test thoroughly before committing!

## Getting Help

If you encounter issues during migration:

1. **Check the documentation**: See `docs/installation/troubleshooting.md`
2. **Review examples**: Check `examples/` directory for updated usage patterns
3. **Run diagnostics**: Use `videoannotator --version` to verify installation
4. **File an issue**: https://github.com/InfantLab/VideoAnnotator/issues

Include this information when reporting issues:
- VideoAnnotator version (`videoannotator --version`)
- Python version (`python --version`)
- Installation method (uv, pip, conda)
- Full error traceback
- Minimal code example that reproduces the issue

## Summary

The v1.3.0 upgrade primarily involves:

1. **Reinstalling** the package: `uv sync`
2. **Updating imports**: Add `videoannotator.` prefix
3. **Testing**: Verify everything works

The changes are mechanical and straightforward. The new structure provides better maintainability and follows Python best practices.

**Estimated total time: 10-30 minutes for most projects**
