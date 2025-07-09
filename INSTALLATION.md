# 🚀 VideoAnnotator Installation Guide

VideoAnnotator is a powerful video analysis tool that leverages GPU-accelerated machine learning models for high-performance video processing. This guide will help you set up a CUDA-enabled environment for optimal performance.

## Prerequisites

- **Python 3.13** (recommended)
- **Git** for cloning repositories
- **Visual Studio Build Tools** (Windows) or **GCC** (Linux/Mac)
- **CMake** (version 3.22 or higher)
- **CUDA Toolkit 12.8** (recommended for GPU acceleration)
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

## Installation Steps

### 1. Create Conda Environment

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate VideoAnnotator
```

### 2. Install PyTorch with CUDA Support

```bash
# Install CUDA-enabled PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Dependencies

```bash
# Install all other Python dependencies
pip install -r requirements.txt
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

### 3. Verify Installation

```python
# Test CUDA availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Test YOLO11 installation
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
print("YOLO11 installed successfully!")

# Test OpenFace (after manual installation)
try:
    import openface
    print("OpenFace 3.0 installed successfully!")
except ImportError:
    print("OpenFace 3.0 requires manual installation - see above")

# Test other key components
import cv2
import transformers
import scenedetect
import mediapipe
print("All core dependencies installed!")
```

### 4. Download Required Models

```bash
# Download YOLO11 models
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt')"

# Download CLIP models
python -c "import clip; clip.load('ViT-B/32')"

# Download face recognition models (happens automatically on first use)
python -c "import deepface; deepface.DeepFace.represent('test.jpg', model_name='VGG-Face')" || echo "Will download on first use"
```

## Alternative Installation Methods

### Using pip only (without conda)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CUDA-enabled PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other requirements
pip install -r requirements.txt
```

### Docker Installation (GPU-enabled)

```bash
# Build Docker image with GPU support
docker build -t videoannotator .

# Run with NVIDIA GPU support
docker run --gpus all -it videoannotator

# Verify CUDA is available inside the container
docker run --gpus all -it videoannotator python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## Troubleshooting

### CUDA/GPU Issues

1. **CUDA not found**: Run `nvidia-smi` to check if your GPU is detected. Update GPU drivers if needed.

2. **PyTorch doesn't detect CUDA**: Ensure you installed the CUDA-enabled version of PyTorch. Verify with:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

3. **CUDA version mismatch**: Ensure your CUDA Toolkit version matches the PyTorch CUDA version.
   - Check CUDA Toolkit: `nvcc --version`
   - Check PyTorch CUDA: `python -c "import torch; print(torch.version.cuda)"`

4. **Out of memory errors**: Try:
   - Reduce batch size
   - Use a smaller model
   - Free unused tensors with `del variable` and `torch.cuda.empty_cache()`
   - Use model checkpointing for large models

### Common Installation Issues

1. **OpenFace compilation errors**: Ensure you have all build tools installed
2. **Package conflicts**: Try installing dependencies one by one to identify conflicts
3. **Memory issues**: Some models require significant RAM/VRAM
4. **Import errors**: Ensure all dependencies are installed in the active environment

## CUDA and PyTorch Installation

### 1. Install CUDA Toolkit

Ensure you have the CUDA Toolkit installed on your system:

- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Current recommended version: CUDA 12.8
- Verify installation with `nvcc --version`

### 2. Install CUDA-enabled PyTorch

```bash
# Install CUDA-enabled PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Verify CUDA Support

```python
# Verify CUDA is available and working
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device()}'); print(f'Device name: {torch.cuda.get_device_name(0)}')"
```

### 4. Common CUDA/PyTorch Issues

- **CUDA not detected**: Ensure your GPU drivers are up-to-date
- **Version mismatch**: Ensure PyTorch CUDA version matches your CUDA Toolkit
- **Memory errors**: Reduce batch sizes or model sizes
- **Import errors**: Ensure you've installed the CUDA-enabled PyTorch packages
- **Multiple GPUs**: Set `CUDA_VISIBLE_DEVICES` environment variable to select specific GPUs

### Performance Optimization

- **GPU Support**: With CUDA-enabled PyTorch, inference will be significantly faster
- **Model Caching**: Models will be cached after first download
- **Batch Processing**: Process multiple videos in batches for efficiency
- **Mixed Precision**: Use `torch.cuda.amp` for faster inference with minimal accuracy loss

## Environment Management

### Conda + Pip Hybrid Approach

This project uses a hybrid approach for dependency management:

1. **Conda** for system-level packages and environment management
   - Creates an isolated Python 3.13 environment
   - Handles C/C++ dependencies that are hard to install with pip
   - Provides consistent environment across platforms

2. **Pip** for Python-specific packages
   - Installs PyTorch with CUDA support
   - Installs all other Python dependencies
   - Ensures compatibility with Python 3.13

### Managing Environment Updates

To update your environment:

```bash
# Update conda environment
conda env update -f environment.yml

# Update pip packages
pip install -r requirements.txt --upgrade
```

### Multiple GPU Management

If you have multiple GPUs, you can select which ones to use:

```bash
# Set visible devices before running your script
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES="0,1"  # Use GPUs 0 and 1

# Linux/Mac
export CUDA_VISIBLE_DEVICES="0,1"  # Use GPUs 0 and 1
```

Or within Python:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use GPUs 0 and 1
import torch
```

## OpenFace 3.0 Features

Once installed, OpenFace 3.0 provides:
- ✅ **Face Detection & Tracking**
- ✅ **Facial Landmark Detection** (68-point model)
- ✅ **Gaze Estimation** (head pose and eye gaze)
- ✅ **Facial Action Units** (muscle movements)
- ✅ **Emotion Recognition** (7 basic emotions)
- ✅ **Age & Gender Estimation**
- ✅ **Real-time Processing** capabilities

## Dependencies Overview

| Category | Tools | Purpose | CUDA Support |
|----------|-------|---------|-------------|
| **Detection** | YOLO11, MediaPipe | Person/object detection | ✅ (YOLO11) |
| **Tracking** | YOLO11 tracking, ByteTrack | Multi-object tracking | ✅ |
| **Scene** | PySceneDetect, CLIP | Scene segmentation & classification | ✅ (CLIP) |
| **Face** | OpenFace 3.0, MediaPipe | Face analysis & emotion | ✅ (OpenFace) |
| **Audio** | Whisper, pyannote.audio | Speech & audio processing | ✅ |
| **Annotation** | Label Studio, FiftyOne | Data annotation & visualization | N/A |

### Python 3.13 Compatibility Notes

Some packages are not yet compatible with Python 3.13:

- **DeepFace**: TensorFlow dependency not yet Python 3.13 compatible
- **CLIP by OpenAI**: Original implementation not compatible with Python 3.13
- **face-recognition**: Not yet updated for Python 3.13

Alternative compatible packages are provided in the codebase.

## Next Steps

After installation, see:
- `README.md` for usage examples
- `docs/` directory for detailed documentation
- `src/pipelines/` for pipeline implementations
- `configs/` for configuration examples

## CUDA Performance Optimization

### Model Acceleration

1. **Precision Reduction**
   ```python
   # Use mixed precision for faster inference
   from torch.cuda.amp import autocast
   with autocast():
       outputs = model(inputs)
   ```

2. **Batch Processing**
   ```python
   # Process multiple inputs at once
   batch_inputs = torch.stack([input1, input2, input3])
   batch_outputs = model(batch_inputs)
   ```

3. **Model Optimization**
   ```python
   # Convert model to TorchScript for faster inference
   scripted_model = torch.jit.script(model)
   outputs = scripted_model(inputs)
   
   # Or quantize model for reduced memory usage
   quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```
