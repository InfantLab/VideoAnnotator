# VideoAnnotator Docker Deployment Guide

VideoAnnotator v1.2.0 includes modern Docker containers using uv for fast, reliable dependency management.

## 🐳 Available Docker Images

### 1. **videoannotator:cpu** - Production CPU-only
- **Size**: ~15-20GB (optimized)  
- **Use case**: Production deployment without GPU
- **Features**: CPU-only PyTorch, models download on first use
- **Build time**: ~10-15 minutes

```bash
# Build CPU image
docker build -f Dockerfile.cpu -t videoannotator:cpu .

# Run CPU container
docker run -p 8000:8000 --rm -v "${PWD}/data:/app/data" -v "${PWD}/output:/app/output" videoannotator:cpu
```

### 2. **videoannotator:gpu** - Production GPU
- **Size**: ~25-30GB (optimized)
- **Use case**: Production deployment with GPU acceleration  
- **Features**: CUDA PyTorch, models download on first use
- **Build time**: ~15-20 minutes

```bash
# Build GPU image
docker build -f Dockerfile.gpu -t videoannotator:gpu .

# Run GPU container
docker run -p 8000:8000 --gpus all --rm -v "${PWD}/data:/app/data" -v "${PWD}/output:/app/output" videoannotator:gpu
```

### 3. **videoannotator:dev** - Development with Local Model Cache  
- **Size**: ~50GB (includes your local models/weights)
- **Use case**: Development and testing - instant model access
- **Features**: GPU support + your exact local model cache copied
- **Build time**: ~15-20 minutes (no model downloading, just copying)

```bash
# Build development image (includes models)
docker build -f dockerfile.dev -t videoannotator:dev .

# Run development container
docker run -p 8000:8000 --gpus all --rm \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/output:/app/output" \
  -v "${PWD}/database:/app/database" \
  -v "${PWD}/logs:/app/logs" \
  videoannotator:dev
```

## 📊 Image Comparison

| Image | Size | Build Time | Model Access | GPU Support | Use Case |
|-------|------|------------|-------------|-------------|----------|
| **cpu** | ~15GB | 10-15 min | Downloads at runtime | ❌ | Production CPU |
| **gpu** | ~25GB | 15-20 min | Downloads at runtime | ✅ | Production GPU |
| **dev** | ~50GB | 15-20 min | Local cache copied | ✅ | Development/Testing |

## 🚀 Docker Compose Usage

### Development with GPU (Recommended for Testing)
```bash
# Start development service with pre-cached models
docker compose --profile dev-gpu up videoannotator-dev-gpu

# Access API at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Production GPU
```bash
# Start production GPU service
docker compose --profile gpu up videoannotator-gpu
```

### CPU-only Production
```bash
# Start CPU service  
docker compose --profile prod up videoannotator-prod
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

## 🗂️ Volume Mounts

### Essential Mounts
- **`./data:/app/data`** - Input videos and test data
- **`./output:/app/output`** - Processing results and annotations
- **`./logs:/app/logs`** - Application logs (dev only)
- **`./database:/app/database`** - Persistent SQLite database (dev only)

### Example with All Mounts
```bash
docker run -p 8000:8000 --gpus all --rm \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/output:/app/output" \
  -v "${PWD}/logs:/app/logs" \
  -v "${PWD}/database:/app/database" \
  -e VIDEOANNOTATOR_DB_PATH=/app/database/videoannotator.db \
  videoannotator:dev
```

## 🔧 Development Workflow

### For Fast Testing (Use dev image)
1. **Build once**: `docker build -f dockerfile.dev -t videoannotator:dev .`
2. **Run anytime**: `docker run -p 8000:8000 --gpus all videoannotator:dev`
3. **No wait**: Your local models copied, immediate testing

### For Production (Use optimized images)
1. **Build**: `docker build -f Dockerfile.gpu -t videoannotator:gpu .`  
2. **Deploy**: Models download on first use, smaller image size
3. **Scale**: Suitable for container orchestration

## 🎯 Quick Commands

```bash
# Build all images
docker build -f Dockerfile.cpu -t videoannotator:cpu .
docker build -f Dockerfile.gpu -t videoannotator:gpu .  
docker build -f dockerfile.dev -t videoannotator:dev .

# Run development server (fastest for testing)
docker run -p 8000:8000 --gpus all --rm videoannotator:dev

# Run production server
docker run -p 8000:8000 --gpus all --rm videoannotator:gpu

# Check GPU access in container
docker run --gpus all --rm videoannotator:dev python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🐛 Troubleshooting

### Build Issues
- **"NVIDIA files in CPU build"**: Fixed in latest Dockerfile.cpu with CPU-only PyTorch
- **"Models folder too large"**: Use .dockerignore to exclude models/ for prod builds  
- **"Build takes too long"**: Use dev image only when you need pre-cached models
- **"Package libgl1-mesa-glx not available"**: Fixed with correct Debian package names
- **"No solution found when resolving dependencies"**: Fixed with --frozen flag and Python <3.13 constraint

### Runtime Issues
- **"No GPU detected"**: Ensure `--gpus all` flag and NVIDIA Docker runtime installed
- **"Models downloading slowly"**: Use dev image for testing, or cache models on host
- **"Permission denied"**: Check volume mount permissions and paths

### Windows PowerShell
```powershell
# Windows-specific command format
docker run -p 8000:8000 --gpus all --rm -v "${PWD}\data:/app/data" -v "${PWD}\output:/app/output" videoannotator:dev
```

## Container Features

- **uv package manager** for fast, reliable dependency management
- **Python 3.12** runtime
- **CUDA 13.0** support (GPU container)
- **Models download automatically** on first use (production) or pre-cached (dev)
- **FastAPI server** ready for deployment
- **Automatic dependency resolution** via uv.lock
