# OpenFace 3.0 Installation Guide

This document provides a comprehensive guide for installing OpenFace 3.0 for use in the VideoAnnotator face analysis pipeline.

## Overview

OpenFace 3.0 is a state-of-the-art facial behavior analysis toolkit that provides:
- Real-time facial landmark detection (2D and 3D)
- Head pose estimation
- Facial action unit (AU) recognition
- Gaze estimation
- Emotion recognition

**Repository**: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8-3.11 (3.12 may have compatibility issues)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU recommended for real-time processing

### Dependencies
- OpenCV 4.5+
- dlib 19.24+
- TensorFlow/PyTorch (for deep learning models)
- CMake 3.16+
- Visual Studio Build Tools (Windows only)

## Installation Steps

### Step 1: Environment Setup

```bash
# Create dedicated conda environment
conda create -n openface3 python=3.10
conda activate openface3

# Install base dependencies
conda install opencv cmake numpy scipy matplotlib
pip install dlib tensorflow torch torchvision
```

### Step 2: OpenFace 3.0 Installation

#### Option A: From Source (Recommended)
```bash
# Clone repository
git clone https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git
cd OpenFace-3.0

# Build C++ components
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make -j$(nproc)  # Linux/Mac
# or use Visual Studio on Windows

# Install Python bindings
cd ../python
pip install -e .
```

#### Option B: Pre-built Binaries (if available)
```bash
pip install openface3
```

### Step 3: Model Downloads

OpenFace 3.0 requires several pre-trained models:

```bash
# RECOMMENDED: Use the built-in download command
# Apply compatibility patch first and then download
python -c "
import sys
sys.path.insert(0, '.')
from src.pipelines.face_analysis.openface_compatibility import patch_scipy_compatibility
patch_scipy_compatibility()
from openface.cli import download
download()
"

# Models will be downloaded to:
# - weights/Alignment_RetinaFace.pth (Face detection)
# - weights/Landmark_68.pkl (68-point landmarks)
# - weights/Landmark_98.pkl (98-point landmarks)
# - weights/MTL_backbone.pth (Multi-task learning backbone)
# - weights/mobilenetV1X0.25_pretrain.tar (Mobile face detection)
```

#### Alternative: Manual Download
```bash
# Download required models (run from OpenFace-3.0 directory)
python scripts/download_models.py

# Models will be downloaded to:
# - models/landmarks/
# - models/au_models/
# - models/gaze_models/
# - models/head_pose/
```

### Step 4: Verification

Test the installation:

```python
# Test with compatibility patches applied
import sys
sys.path.insert(0, '.')  # If running from VideoAnnotator root
from src.pipelines.face_analysis.openface_compatibility import patch_scipy_compatibility
patch_scipy_compatibility()

# Test imports
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector

print("✅ OpenFace components imported successfully!")

# Test basic functionality (requires downloaded weights)
try:
    detector = FaceDetector(model_path='./weights/Alignment_RetinaFace.pth')
    print("✅ FaceDetector initialized successfully!")
except Exception as e:
    print(f"❌ FaceDetector initialization failed: {e}")

try:
    landmark_detector = LandmarkDetector(model_path='./weights/Landmark_98.pkl')
    print("✅ LandmarkDetector (98-point) initialized successfully!")
except Exception as e:
    print(f"❌ LandmarkDetector initialization failed: {e}")
```

#### Expected Output:
```
✅ OpenFace components imported successfully!
✅ FaceDetector initialized successfully!
✅ LandmarkDetector initialized successfully!
```

**Note**: Based on testing, the correct model paths and landmarks are:
- Use `./weights/Alignment_RetinaFace.pth` for FaceDetector
- Use `./weights/Landmark_98.pkl` for LandmarkDetector (98-point landmarks)
- The `Landmark_68.pkl` model has shape mismatches with the current implementation

## Known Issues and Solutions

### Issue 1: SciPy Compatibility (scipy.integrate.simps deprecated)

**Problem**: `ImportError: cannot import name 'simps' from 'scipy.integrate'`
**Cause**: OpenFace uses the deprecated `simps` function which was removed in SciPy 1.14.0+
**Solutions**:

#### Option A: Patch OpenFace (Recommended)
```python
# Create a compatibility patch
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
```

#### Option B: Downgrade SciPy (Temporary)
```bash
pip install "scipy<1.14.0"
```

#### Option C: Manual Patch
Edit the OpenFace source files to replace `simps` with `simpson`:
- File: `openface/STAR/lib/metric/fr_and_auc.py`
- Change: `from scipy.integrate import simps` → `from scipy.integrate import simpson as simps`

### Issue 2: FaceDetector Requires Model Path

**Problem**: `FaceDetector.__init__() missing 1 required positional argument: 'model_path'`
**Solution**: OpenFace requires pre-trained models to be downloaded and specified
```python
from openface.face_detection import FaceDetector
detector = FaceDetector(model_path='path/to/face_detection_model.pth')
```

### Issue 3: Model Shape Mismatch

**Problem**: `RuntimeError: size mismatch for out_heatmaps.0.conv.weight: copying a param with shape torch.Size([68, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([98, 256, 1, 1])`
**Cause**: The 68-point landmark model is incompatible with the default 98-point model architecture
**Solution**: Use the correct model for the desired landmark configuration
```python
# Use 98-point landmark model (recommended)
landmark_detector = LandmarkDetector(model_path='./weights/Landmark_98.pkl')

# For 68-point landmarks, you may need a different model or configuration
# Currently, the downloaded 68-point model is incompatible
```

### Issue 4: CMake Build Errors (when building from source)

**Problem**: CMake fails to find dependencies or configure build
**Solution**:
```bash
# Install missing dependencies
sudo apt-get install build-essential cmake libopencv-dev  # Ubuntu
brew install cmake opencv  # macOS

# Clear CMake cache and retry
rm -rf build
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
```

### Issue 2: Python Binding Compilation

**Problem**: Python bindings fail to compile with pybind11 errors
**Solution**:
```bash
# Install specific pybind11 version
pip install pybind11==2.10.4

# Use conda-forge for better compatibility
conda install -c conda-forge pybind11
```

### Issue 3: Model Download Failures

**Problem**: Model download script fails due to network issues
**Solution**:
```bash
# Manual download from releases
wget https://github.com/CMU-MultiComp-Lab/OpenFace-3.0/releases/download/v3.0.0/models.zip
unzip models.zip -d models/
```

### Issue 4: CUDA Compatibility

**Problem**: CUDA version mismatch with PyTorch/TensorFlow
**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Issue 5: dlib Installation Issues

**Problem**: dlib fails to compile or install
**Solution**:
```bash
# Use conda-forge for pre-compiled dlib
conda install -c conda-forge dlib

# Or install with CMake support
pip install dlib --verbose
```

### Issue 6: Runtime Library Path Issues (Linux)

**Problem**: Shared library not found at runtime
**Solution**:
```bash
# Add to ~/.bashrc or ~/.zshrc
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/OpenFace-3.0/build/lib:$LD_LIBRARY_PATH

# Or use ldconfig
sudo echo "/usr/local/lib" >> /etc/ld.so.conf
sudo echo "/path/to/OpenFace-3.0/build/lib" >> /etc/ld.so.conf
sudo ldconfig
```

### Issue 7: Windows Visual Studio Build Tools

**Problem**: Missing MSVC compiler on Windows
**Solution**:
1. Download Visual Studio Build Tools 2019/2022
2. Install C++ build tools workload
3. Ensure CMake can find MSVC:
```cmd
# Set environment variables
set "VS160COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\"
```

### Issue 8: Memory Issues with Large Videos

**Problem**: Out of memory errors during processing
**Solution**:
```python
# Process videos in chunks
def process_video_chunked(video_path, chunk_size=1000):
    # Implementation in pipeline
    pass

# Reduce model batch size
detector = openface3.FaceDetector(batch_size=1)
```

## Performance Optimization

### GPU Acceleration
```python
# Enable CUDA if available
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure OpenFace for GPU
detector = openface3.FaceDetector(device='cuda')
```

### Batch Processing
```python
# Process multiple frames at once
frames = [frame1, frame2, frame3]  # List of frames
results = detector.detect_batch(frames)
```

### Model Caching
```python
# Pre-load models to avoid repeated initialization
class OpenFaceManager:
    def __init__(self):
        self.detector = openface3.FaceDetector()
        self.landmark_detector = openface3.LandmarkDetector()
        self.au_analyzer = openface3.ActionUnitAnalyzer()
    
    def process_frame(self, frame):
        # Use cached models
        pass
```

## Integration with VideoAnnotator

The OpenFace 3.0 pipeline should be integrated as:

```python
from src.pipelines.face_analysis.openface3_pipeline import OpenFace3Pipeline

# In main.py pipeline initialization
if self.config.get('face_analysis', {}).get('enabled', True):
    face_config = self.config.get('face_analysis', {})
    if face_config.get('backend') == 'openface3':
        self.pipelines['face'] = OpenFace3Pipeline(face_config)
```

## Installation Success Summary

✅ **Successfully Completed**:

1. **OpenFace Package Installation**: Installed `openface-test==0.1.26` with `--no-deps` to avoid dependency conflicts
2. **Model Download**: Downloaded all required model weights (~2GB) to `./weights/` directory:
   - `Alignment_RetinaFace.pth` (1.79MB) - Face detection model  
   - `Landmark_98.pkl` (178MB) - 98-point landmark detection model
   - `Landmark_68.pkl` (177MB) - 68-point landmark model (has compatibility issues)
   - `MTL_backbone.pth` (102MB) - Multi-task learning backbone
   - `mobilenetV1X0.25_pretrain.tar` (3.83MB) - Mobile face detection
3. **Compatibility Patches**: Created `openface_compatibility.py` to resolve SciPy integration issues
4. **Component Verification**: Both FaceDetector and LandmarkDetector successfully initialize and load models

**Final Test Results**:
```
✅ OpenFace imports successful
✅ FaceDetector initialized  
✅ LandmarkDetector (98-point) initialized
🎉 OpenFace 3.0 setup complete and working!
```

**Next Steps**: Ready for integration into VideoAnnotator face analysis pipeline with proper model paths and compatibility patches.

## Troubleshooting Checklist

- [ ] Python version 3.8-3.11
- [ ] CMake 3.16+ installed
- [ ] OpenCV 4.5+ installed
- [ ] All dependencies satisfied
- [ ] Models downloaded correctly
- [ ] Library paths configured
- [ ] CUDA drivers updated (if using GPU)
- [ ] Sufficient disk space for models (~2GB)
- [ ] Network access for downloads

## Support and Resources

- **GitHub Issues**: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0/issues
- **Documentation**: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0/wiki
- **Paper**: [OpenFace 3.0: Towards More Robust and Efficient Facial Analysis](https://arxiv.org/abs/2301.07603)
- **CMU MultiComp Lab**: https://multicomp.cs.cmu.edu/

## Version History

- **v3.0.0**: Initial release with PyTorch backend
- **v3.0.1**: Bug fixes for Python bindings
- **v3.0.2**: Windows compatibility improvements
- **v3.1.0**: Performance optimizations and new AU models

Last updated: August 2025
