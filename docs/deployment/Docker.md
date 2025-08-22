# Docker Deployment Guide

VideoAnnotator v1.2.0 includes modern Docker containers using uv for fast, reliable dependency management.

## Container Options

### CPU Container (Dockerfile.cpu)
For environments without GPU support:

```bash
# Build CPU container
docker build -f Dockerfile.cpu -t videoannotator:cpu .

# Run CPU container
docker run --rm -v $(pwd)/data:/app/data videoannotator:cpu
```

### GPU Container (Dockerfile.gpu)
For CUDA-enabled environments (requires NVIDIA Container Toolkit):

```bash
# Build GPU container
docker build -f Dockerfile.gpu -t videoannotator:gpu .

# Run GPU container
docker run --gpus all --rm -v $(pwd)/data:/app/data videoannotator:gpu
```

## Prerequisites

### NVIDIA Container Toolkit (for GPU)
Install NVIDIA Container Toolkit for GPU support:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Development Containers

### VS Code Dev Container
Use the `.devcontainer/devcontainer.json` for a complete development environment:

1. Open project in VS Code
2. Command: "Reopen in Container"
3. Complete GPU-enabled development environment with uv, Ruff, and all dependencies

### Docker Compose (if present)
```bash
# Start development environment
docker-compose up --build

# GPU development
docker-compose --profile gpu up --build
```

## Production Deployment

### API Server
```bash
# Run API server in container
docker run --rm -p 8000:8000 --gpus all videoannotator:gpu

# Access API at http://localhost:8000/docs
```

### Batch Processing
```bash
# Mount data volumes for batch processing
docker run --rm \
  -v /path/to/videos:/app/data \
  -v /path/to/results:/app/output \
  --gpus all \
  videoannotator:gpu \
  uv run python main.py --input /app/data --output /app/output --batch
```

## Container Features

- **uv package manager** for fast, reliable dependency management
- **Python 3.12** runtime
- **CUDA 12.4** support (GPU container)
- **All AI models** pre-configured
- **FastAPI server** ready for deployment
- **Automatic dependency resolution** via uv.lock
