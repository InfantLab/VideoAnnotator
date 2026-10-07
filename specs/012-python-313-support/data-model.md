# Data Model: Python 3.13 Support

No stored data changes. One configuration entity:

## Supported Python range

The Python versions VideoAnnotator declares, tests and documents.

| Field | Value | Where it lives |
|---|---|---|
| Oldest supported | 3.12 | `pyproject.toml` `requires-python` lower bound; ruff `target-version`; mypy `python_version` |
| Newest supported | 3.13 | `requires-python` upper bound (`<3.14`); classifiers |
| Default for development and images | 3.13 | `.python-version` |
| Runtime copy | `SUPPORTED_PYTHON = ((3, 12), (3, 13))` | `src/videoannotator/version.py` |

**Validation**: a unit test asserts the runtime copy matches `requires-python` and the classifiers,
so the four places can't drift apart.

**Lifecycle**: 3.12 → {3.12, 3.13} (this spec) → {3.13, 3.14} (face-stack spec adds 3.14; v1.7.0
drops 3.12).
