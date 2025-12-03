# Dev Container PyTorch CUDA Installation

## Optimized Setup

We have optimized the project configuration to handle PyTorch CUDA dependencies declaratively.

## How it Works

1.  **`pyproject.toml` Configuration**:
    We explicitly define the PyTorch CUDA index and map the relevant packages to it:
    ```toml
    [[tool.uv.index]]
    name = "pytorch-cu124"
    url = "https://download.pytorch.org/whl/cu124"
    explicit = true

    [tool.uv.sources]
    torch = { index = "pytorch-cu124" }
    torchvision = { index = "pytorch-cu124" }
    torchaudio = { index = "pytorch-cu124" }
    ```

2.  **`uv.lock`**:
    The lockfile now pins these packages to the versions found in the CUDA index (e.g., `+cu124` versions).

3.  **Dockerfile Optimization**:
    - `Dockerfile.gpu` simply runs `uv sync --frozen`.
    - It copies `pyproject.toml` and `uv.lock` *before* the source code to leverage Docker layer caching.
    - Dependencies are installed during the image build (because we set `SKIP_IMAGE_UV_SYNC=false` in `devcontainer.json`).

4.  **Dev Container**:
    - The container starts with all dependencies pre-installed.
    - `postCreateCommand` runs `uv sync` just to ensure everything is up-to-date, but it should be instant.

## Verification

To verify the installation inside the container:

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Troubleshooting

If you see "No solution found" errors during build:
1.  Ensure `uv.lock` is up-to-date with `pyproject.toml`. Run `uv lock` locally if needed.
2.  Check if the PyTorch index URL is accessible.
