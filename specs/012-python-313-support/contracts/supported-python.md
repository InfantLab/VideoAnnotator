# Contract: supported Python versions

## Install time

| Installer | Python 3.12 / 3.13 | Any other version |
|---|---|---|
| `pip install videoannotator[...]` | installs | refuses in the first step: `requires a different Python: X not in '<3.14,>=3.12'` |
| `uv sync` in a checkout | installs (3.13 by default, from `.python-version`) | refuses: interpreter incompatible with the project's Python requirement |
| `uv pip install .` from a checkout | installs | **installs anyway** (uv doesn't check the local project's `requires-python`); see runtime |

## Runtime

When the CLI (`videoannotator ...`) or the API server starts on a Python outside the supported
range, it logs one warning, and continues:

```
[WARNING] Python 3.14 is not supported by VideoAnnotator 1.6.0 (supported: 3.12, 3.13).
Some pipelines may fail to install or run.
```

- Emitted once per process, before any pipeline is loaded.
- Never emitted on 3.12 or 3.13.
- A warning, not an error: nothing is blocked.

## Metadata

- `requires-python = ">=3.12,<3.14"`.
- Classifiers list `Programming Language :: Python :: 3.12` and `:: 3.13`.
