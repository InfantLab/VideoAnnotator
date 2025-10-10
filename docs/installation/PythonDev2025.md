# Modern Python Project Setup (CUDA + PyTorch) — Short Guide for Coding Agents

**Objective:** Lowest-friction local dev, reproducible containers (CPU/GPU), agent-friendly conventions.
**Defaults:** `uv` + `pyproject.toml` + Docker. Use `micromamba` only if native deps are painful (geo/ffmpeg/etc.).
**Why:** `uv` is a fast, standards-aligned project & package manager with a universal lockfile; NVIDIA Container Toolkit is the GPU runtime; PyTorch provides pip/Conda CUDA builds. ([Astral Docs][1], [NVIDIA Docs][2], [PyTorch][3])

---

## 0) Repository hygiene (apply to new or existing repos)

**Agent, do this:**

- Ensure a clean tree: `src/` layout, tests in `tests/`, no code at top level.
- Add `.gitignore`, `.editorconfig`, `README.md` with _How to run_.
- Keep secrets in `.env` (never commit); ship `.env.example`.

```
repo/
  src/<package_name>/...
  tests/
  pyproject.toml
  uv.lock
  .editorconfig
  .gitignore
  .pre-commit-config.yaml
  README.md
  Dockerfile.cpu
  Dockerfile.gpu
  .devcontainer/devcontainer.json
  .env.example
```

---

## 1) Project metadata & tooling (pyproject + uv)

**Agent, create/update: `pyproject.toml`**

```toml
[project]
name = "your-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115",
  "httpx>=0.27",
  "pydantic>=2.7",
  # Add your core deps here (avoid pinning too tightly except for CUDA libs)
]

[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[tool.uv]  # uv-native config if needed
# empty is fine; uv uses pyproject + uv.lock

[tool.ruff]
line-length = 100
lint.select = ["E","F","I","UP","B","SIM","RUF"]
lint.ignore = ["E501"]
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = false
warn_unused_ignores = true
warn_redundant_casts = true

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
```

**Agent, initialize & lock:**

```bash
uv sync        # creates venv and installs deps
uv add ruff mypy pytest pre-commit
uv lock
```

`uv` is the fast, standards-aligned default; it works with `pyproject.toml` and creates a reproducible `uv.lock`. ([Astral Docs][1], [packaging.python.org][4])

---

## 2) Pre-commit quality gates

**Agent, add `.pre-commit-config.yaml`:**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
```

```bash
pre-commit install
```

Ruff = fast lints/format; pre-commit enforces on commit. ([Astral Docs][5], [Astral][6], [pre-commit.com][7])

---

## 3) Local run commands (simple Makefile)

**Agent, create `Makefile`:**

```makefile
PY=uv run
install:
	uv sync

lint:
	uv run ruff check .
format:
	uv run ruff format .
typecheck:
	uv run mypy src
test:
	uv run pytest

run:
	$(PY) python -m your_project  # adjust entrypoint
```

---

## 4) CUDA + PyTorch installation (local)

**Default (pip via uv):**

- Detect the host CUDA you want to target (e.g., 13.0).
- Install PyTorch exactly as the official selector prints (pip or conda). Example:

```bash
# Example: CUDA 13.0 wheels (confirm on pytorch.org first)
uv add "torch==2.8.*+cu130" "torchvision==0.21.*+cu130" --index-url https://download.pytorch.org/whl/cu130
```

Always copy the exact command from the PyTorch site to match torch/torchvision/cuda triplets. ([PyTorch][3])

> If native/system deps become painful (e.g., ffmpeg/GEOS), prefer **micromamba** with `conda-forge` and pin CUDA there. It’s a tiny, base-less conda compatible manager ideal for CI/containers. ([mamba.readthedocs.io][8])

---

## 5) Containers (CPU and GPU)

### `Dockerfile.cpu`

```dockerfile
FROM python:3.12-slim

# system basics
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# uv installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-editable

COPY . .
CMD ["uv", "run", "python", "-m", "your_project"]
```

(`uv` + lockfile = deterministic builds.) ([Astral Docs][1])

### `Dockerfile.gpu`

```dockerfile
# Choose a CUDA runtime matching your torch build (e.g., 13.0)
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

SHELL ["/bin/bash","-lc"]
RUN apt-get update && apt-get install -y curl python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-editable
# Install torch for CUDA build selected (match to pytorch.org command)
RUN uv add "torch==2.8.*+cu130" "torchvision==0.21.*+cu130" --index-url https://download.pytorch.org/whl/cu130

COPY . .
CMD ["uv", "run", "python", "-m", "your_project"]
```

**Host requirement:** Install **NVIDIA Container Toolkit** and run with `--gpus all`.

```bash
docker run --rm --gpus all your-image:tag
```

Toolkit docs: install on your distro and Docker integrates GPU runtime. ([NVIDIA Docs][2], [GitHub][9])

---

## 6) Dev containers (VS Code / Cursor / JetBrains Gateway)

**Agent, create `.devcontainer/devcontainer.json`:**

```json
{
  "name": "your-project (GPU)",
  "build": { "dockerfile": "../Dockerfile.gpu", "context": ".." },
  "runArgs": ["--gpus", "all"],
  "features": {},
  "postCreateCommand": "uv sync && pre-commit install",
  "customizations": {
    "vscode": {
      "settings": { "python.defaultInterpreterPath": ".venv/bin/python" }
    }
  }
}
```

This gives every contributor (and coding agent) the same GPU-ready workspace.

---

## 7) CI (GitHub Actions) — fast install with uv

**Agent, add `.github/workflows/ci.yml`:**

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run pytest
```

For GPU CI, use self-hosted runners with NVIDIA toolkit. ([Astral Docs][10])

---

## 8) Micromamba fallback (only if needed)

If you hit native deps hell, switch the container to **micromamba**:

**`environment.yml`**

```yaml
name: proj
channels: [conda-forge]
dependencies:
  - python=3.12
  - pytorch=*=cuda* # or cpu if desired; pin to your CUDA
  - cudatoolkit=13.0
  - numpy
  - pandas
  - pip
  - pip:
      - ruff
      - mypy
      - pytest
```

**`Dockerfile.mamba` (sketch)**

```dockerfile
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04
SHELL ["/bin/bash","-lc"]
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -Ls https://micromamba.snakepit.net/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/micromamba

WORKDIR /app
COPY environment.yml .
RUN micromamba create -y -f environment.yml && micromamba clean -a -y
COPY . .
CMD ["bash","-lc","micromamba run -n proj python -m your_project"]
```

Micromamba is tiny, base-less, and excellent for CI/containers. ([mamba.readthedocs.io][8])

---

## 9) Notes for agents working inside IDEs

- Always run via `uv run …` so the venv is consistent.
- Never mix pip/conda in one env. If the project chooses micromamba, keep all deps in `environment.yml`.
- For PyTorch, always copy the official install line for the target CUDA (dev & container must match). ([PyTorch][3])
- Commit `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, CI workflow, and Dockerfiles.

---

## 10) Applying to `InfantLab/VideoAnnotator`

- Keep the defaults above (`uv` + CPU/GPU Dockerfiles).
- If the project adds heavy multimedia (ffmpeg/OpenCV) and wheels aren’t enough, use the **micromamba** fallback container.
- Export a short **RUNBOOK.md** with: _Local (uv)_, _Docker CPU_, _Docker GPU_, and _devcontainer_ quickstart.

---
