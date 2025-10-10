# Upgrade to Latest Versions (September 2025)

This document explains the upgrade from older dependency versions to the latest stable releases.

## What Was Updated

### Container Images

- **Ubuntu**: 22.04 LTS → **24.04.3 LTS** (Noble Numbat)
- **CUDA**: 12.4.1 → **13.0.1** (latest stable)
- **Python**: 3.12 → **3.13** (CPU containers)

### PyTorch

- **PyTorch**: 2.4.0 → **2.8.0** (current stable)
- **TorchVision**: 0.19.0 → **0.21.0**
- **TorchAudio**: 2.4.0 → **2.8.0**
- **CUDA Build**: cu124 → **cu130**

## Why Update?

**No documented justification** was found for staying on the older versions. The project documentation used these versions without explanation.

**Benefits of updating:**

- Latest performance improvements and bug fixes
- Better security with recent Ubuntu LTS
- Access to newest PyTorch features and optimizations
- Future compatibility with newer ML libraries
- Ubuntu 24.04 LTS has 10 years of support (until 2034)

## Compatibility

### What Should Work

- All existing VideoAnnotator functionality
- Existing model files and weights
- Pipeline configurations and outputs
- API endpoints and client code

### What Might Need Testing

- Third-party ML models (some may need updates for PyTorch 2.8)
- Custom pipeline extensions using PyTorch internals
- Performance characteristics (should be same or better)

## Migration Guide

### For Docker Users

Rebuild images with the updated Dockerfiles:

```bash
# GPU image
docker build -f Dockerfile.gpu -t videoannotator:gpu-latest .

# CPU image
docker build -f Dockerfile.cpu -t videoannotator:cpu-latest .

# Dev image
docker build -f Dockerfile.dev -t videoannotator:dev-latest .
```

### For Local Development

Update your PyTorch installation:

```bash
# CUDA 13.0
uv pip install "torch==2.8.*+cu130" "torchvision==0.21.*+cu130" --index-url https://download.pytorch.org/whl/cu130

# CPU only
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### For Production Deployments

- Test the updated images in staging first
- Monitor performance and error rates
- Keep rollback images available during initial deployment

## Version Matrix

| Component   | Previous | Current     | Notes                       |
| ----------- | -------- | ----------- | --------------------------- |
| Ubuntu      | 22.04    | 24.04.3 LTS | Latest LTS, 10 year support |
| CUDA        | 12.4.1   | 13.0.1      | Latest stable               |
| Python      | 3.12     | 3.13        | Latest stable               |
| PyTorch     | 2.4.0    | 2.8.0       | Current stable release      |
| TorchVision | 0.19.0   | 0.21.0      | Matches PyTorch 2.8         |

## Rollback Plan

If issues arise, you can revert by:

1. Using older image tags or building with previous Dockerfiles
2. Downgrading PyTorch: `uv pip install "torch==2.4.0+cu124" ...`
3. The previous versions remain available on PyTorch's download servers

## Validation

After upgrading:

1. **Basic functionality**: Run health checks and pipeline smoke tests
2. **Performance**: Compare processing times on representative videos
3. **Model compatibility**: Test all enabled pipelines
4. **API compatibility**: Verify existing client integrations work

## Files Changed

- `Dockerfile.dev` - CUDA 13.0.1 + Ubuntu 24.04 + PyTorch 2.8
- `Dockerfile.gpu` - CUDA 13.0.1 + Ubuntu 24.04 + PyTorch 2.8
- `Dockerfile.cpu` - Python 3.13 + PyTorch 2.8 CPU
- `.devcontainer/devcontainer.json` - Updated PyTorch versions
- `docs/installation/*.md` - Updated version references
- `docs/deployment/Docker.md` - Updated CUDA version notes

## Questions?

If you encounter compatibility issues:

1. Check the [PyTorch 2.8 release notes](https://pytorch.org/blog/) for breaking changes
2. Verify model compatibility with newer PyTorch
3. File an issue with specific error messages and reproduction steps
