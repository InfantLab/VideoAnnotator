# VideoAnnotator v1.2.0 Installation Guide

VideoAnnotator is a modern video analysis toolkit that uses AI models for comprehensive behavioral annotation. This guide covers installation using our modern **uv-based workflow** for fast, reliable dependency management.

## Prerequisites

- **Python 3.12+** (required)
- **Git** for cloning repositories 
- **uv** package manager (fast, modern Python dependency management)
- **CUDA Toolkit 12.4** (recommended for GPU acceleration)
- **NVIDIA GPU** with CUDA support (GTX 1060 6GB+ or better recommended)

## System Requirements

### Minimum Requirements
- **CPU**: Intel Core i5 (8th gen) or AMD Ryzen 5 (2000 series) or better
- **RAM**: 16GB
- **Storage**: 20GB free space
- **GPU**: NVIDIA GTX 1060 6GB (for CUDA acceleration)
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS 11+

### Recommended Requirements
- **CPU**: Intel Core i7/i9 (11th gen+) or AMD Ryzen 7/9 (5000 series+)
- **RAM**: 32GB or more
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA RTX 3060 12GB or better
- **OS**: Windows 11, Ubuntu 22.04+, or macOS 12+

## Quick Start (Recommended)

### 1. Install uv Package Manager

```bash
# Install uv (fast, modern Python package manager)
# Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/VideoAnnotator.git
cd VideoAnnotator

# Install all dependencies with uv (fast and reliable)
uv sync

# Install development dependencies
uv sync --extra dev
```

### 3. Install CUDA-enabled PyTorch (GPU acceleration)

```bash
# Install PyTorch with CUDA 12.4 support (for GPU acceleration)
uv add "torch==2.4.*+cu124" "torchvision==0.19.*+cu124" --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Native Dependencies (if needed)

Some dependencies like `dlib` require system-level installation:

```bash
# If using conda for native dependencies:
conda install -n videoannotator cmake dlib -c conda-forge

# Otherwise, install cmake system-wide from cmake.org
```

### 2. Install OpenFace 3.0 (Manual Installation Required)

OpenFace 3.0 requires manual compilation. Follow these steps:

#### Windows Installation

```bash
# Install additional dependencies
conda install -c conda-forge boost
conda install -c conda-forge openblas

# Clone OpenFace 3.0
git clone https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git
cd OpenFace-3.0

# Build OpenFace
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release

# Install Python bindings
cd ../python
python setup.py install
```

#### Linux/Mac Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libboost-all-dev

# Clone and build OpenFace 3.0
git clone https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git
cd OpenFace-3.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make -j4

# Install Python bindings
cd ../python
python setup.py install
```

## Verify Installation

```bash
# Test the installation
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')

# Test YOLO11 installation
from ultralytics import YOLO
print('YOLO11 available!')

# Test other key components
import cv2, transformers, scenedetect
print('All core dependencies installed!')
"

# Test the API server
uv run python api_server.py
# Should start server on http://localhost:8000
```

## Development Commands

Once installed, use these commands for development:

```bash
# Start API server
uv run python api_server.py

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run type checking  
uv run mypy src

# Run tests
uv run pytest

# Start API server
uv run python -m src.cli server
uv run python api_server.py
```

## Docker Installation (Alternative)

### CPU Container
```bash
# Build and run CPU version
docker build -f Dockerfile.cpu -t videoannotator:cpu .
docker run --rm -v $(pwd)/data:/app/data videoannotator:cpu
```

### GPU Container (Requires NVIDIA Container Toolkit)
```bash
# Build and run GPU version
docker build -f Dockerfile.gpu -t videoannotator:gpu .
docker run --gpus all --rm -v $(pwd)/data:/app/data videoannotator:gpu
```

### Dev Container (VS Code)
Open the project in VS Code and use "Reopen in Container" for a complete GPU-enabled development environment.

## Troubleshooting

### uv Installation Issues

1. **uv command not found**: 
   ```bash
   # Restart your terminal or run:
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Permission errors**: 
   ```bash
   # On Windows, run PowerShell as Administrator
   # On Linux/Mac, ensure you have write permissions to ~/.local/
   ```

3. **Slow dependency resolution**: uv is typically very fast, but large ML dependencies can take time on first install.

### CUDA/GPU Issues

1. **CUDA not found**: Run `nvidia-smi` to check GPU detection. Update drivers if needed.

2. **PyTorch doesn't detect CUDA**: 
   ```bash
   # Verify CUDA-enabled PyTorch installation
   uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **CUDA version mismatch**: Ensure CUDA Toolkit matches PyTorch CUDA version:
   - Check CUDA Toolkit: `nvcc --version`
   - Check PyTorch CUDA: `uv run python -c "import torch; print(torch.version.cuda)"`

### Native Dependencies

1. **dlib/cmake errors**: Install cmake system-wide or use conda for native dependencies
2. **OpenFace compilation**: Requires manual setup - see OpenFace section above
3. **Build tool errors**: Ensure you have Visual Studio Build Tools (Windows) or GCC (Linux/Mac)

## Modern Architecture

VideoAnnotator v1.2.0 uses:

- **uv** - Fast, reliable Python package management
- **Ruff** - Unified linting and formatting (replaces Black, isort, flake8)
- **FastAPI** - Modern API framework
- **Hatchling/setuptools** - Modern build backend
- **Docker** - CPU and GPU containerization
- **Python 3.12+** - Latest Python with performance improvements

## Dependencies Overview

| Category | Tools | Purpose | CUDA Support |
|----------|-------|---------|-------------|
| **Detection** | YOLO11 | Person/object detection | ✅ |
| **Tracking** | ByteTrack | Multi-object tracking | ✅ |
| **Scene** | PySceneDetect, OpenCLIP | Scene segmentation & classification | ✅ |
| **Face** | OpenFace 3.0, LAION Face | Face analysis & emotion | ✅ |
| **Audio** | Whisper, pyannote.audio | Speech & audio processing | ✅ |
| **API** | FastAPI, uvicorn | REST API server | N/A |

## Next Steps

After installation:
- See `docs/usage/GETTING_STARTED.md` for usage examples
- Check `docs/development/` for development workflows  
- Review `configs/` for configuration options
- Use `uv run python -m src.cli --help` to test the CLI
- Use `uv run python api_server.py` to start the server

## Performance Tips

- **GPU acceleration**: Install CUDA-enabled PyTorch for 10x speedup
- **Batch processing**: Process multiple videos for efficiency
- **Memory management**: Use appropriate model sizes for your GPU
- **Container deployment**: Use Docker for consistent environments
