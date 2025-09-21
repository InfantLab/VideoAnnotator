# VideoAnnotator Versioning and Metadata System

## Overview

VideoAnnotator now includes a robust versioning and metadata system that ensures all output JSON files contain comprehensive information about the version of VideoAnnotator used, the models that generated the data, and the processing environment. This provides a clear audit trail for all annotations and enables reproducibility.

## Key Features

### 1. Version Information
- **Version String**: Semantic versioning (e.g., "1.0.0")
- **Release Date**: Date when the version was released
- **Build Date**: When the current build was created
- **Git Information**: Commit hash, branch, and repository status
- **Author and License**: Project metadata

### 2. System Information
- **Platform**: Operating system and version
- **Python Version**: Complete Python version information
- **Architecture**: System architecture (32-bit/64-bit)
- **Hostname**: Machine where processing occurred

### 3. Dependency Tracking
- **Core Dependencies**: Versions of key libraries (OpenCV, PyTorch, Pydantic, etc.)
- **Model Frameworks**: Ultralytics, OpenAI, etc.
- **Status Tracking**: Whether dependencies are installed, missing, or have errors

### 4. Model Information
- **Model Name**: Specific model used (e.g., "yolo11n-pose.pt")
- **Model Type**: Category (YOLO, CLIP, Whisper, etc.)
- **Framework**: Associated framework (Ultralytics, OpenAI, etc.)
- **Load Time**: When the model was loaded
- **File Information**: Size and modification date if available

### 5. Pipeline Metadata
- **Pipeline Name**: Which pipeline processed the data
- **Processing Timestamp**: When processing occurred
- **Configuration**: All parameters used for processing
- **Video Metadata**: Information about the processed video

## Implementation

### Core Module: `src/version.py`

The versioning system is centralized in `src/version.py` which provides:

```python
# Version constants
__version__ = "1.0.0"
__version_info__ = (1, 0, 0)
__release_date__ = "2025-07-10"

# Core functions
get_version_info()           # Complete version information
get_git_info()              # Git repository status
get_dependency_versions()    # Dependency version checking
get_model_info()            # Model-specific information
create_annotation_metadata() # Comprehensive metadata for outputs
print_version_info()        # Human-readable version display
```

### Pipeline Integration

All pipelines inherit from `BasePipeline` which provides:

```python
def set_model_info(model_name, model_path=None)
def create_output_metadata(video_metadata=None)
def save_annotations(annotations, output_path)
```

Each pipeline automatically includes metadata in their outputs:

```python
# Example usage in a pipeline
pipeline.set_model_info("yolo11n-pose.pt", "/path/to/model")
metadata = pipeline.create_output_metadata(video_info)
# Save with metadata included
pipeline.save_annotations(annotations, output_path)
```

### Output Format

All JSON outputs now include this structure:

```json
{
  "metadata": {
    "videoannotator": {
      "version": "1.0.0",
      "release_date": "2025-07-10",
      "git": {
        "commit_hash": "359d693e...",
        "branch": "master",
        "is_clean": false
      }
    },
    "system": {
      "platform": "Windows-11-10.0.26200-SP0",
      "python_version": "3.12.11",
      "architecture": "64bit"
    },
    "dependencies": {
      "opencv-python": "4.12.0",
      "ultralytics": "8.3.163",
      "pydantic": "2.11.7",
      "torch": "2.7.1+cpu"
    },
    "pipeline": {
      "name": "SceneDetectionPipeline",
      "processing_timestamp": "2025-07-10T16:01:15.443587",
      "processing_params": {
        "threshold": 30.0,
        "min_scene_length": 2.0
      }
    },
    "model": {
      "model_name": "PySceneDetect + CLIP",
      "model_type": "CLIP",
      "framework": "OpenAI",
      "loaded_at": "2025-07-10T16:01:15.230883"
    }
  },
  "pipeline": "scene_detection",
  "timestamp": "2025-07-10T16:01:15.443587",
  "annotations": [
    // Actual annotation data
  ]
}
```

## Testing

Comprehensive unit tests ensure the versioning system works correctly:

### Test Categories

1. **Unit Tests** (`TestVersioning`)
   - Version info structure and format
   - Git information handling
   - Dependency version checking
   - Model information for different types
   - Metadata creation functions

2. **Pipeline Tests** (`TestPipelineMetadata`)
   - Pipeline metadata integration
   - Save operations with metadata
   - Base pipeline methods

3. **Integration Tests** (`TestVersioningIntegration`)
   - Version consistency across modules
   - Pipeline metadata consistency
   - End-to-end metadata flow

### Running Tests

```bash
# Run all versioning tests
python -m pytest tests/test_pipelines.py::TestVersioning -v

# Run pipeline metadata tests
python -m pytest tests/test_pipelines.py::TestPipelineMetadata -v

# Run integration tests
python -m pytest tests/test_pipelines.py::TestVersioningIntegration -v

# Run all versioning-related tests
python -m pytest tests/test_pipelines.py::TestVersioning tests/test_pipelines.py::TestPipelineMetadata tests/test_pipelines.py::TestVersioningIntegration -v
```

## Usage Examples

### Display Version Information

```python
from src import print_version_info
print_version_info()
```

### Get Version Data Programmatically

```python
from src.version import get_version_info
version_info = get_version_info()
print(f"VideoAnnotator v{version_info['videoannotator']['version']}")
```

### Check Dependencies

```python
from src.version import get_dependency_versions
deps = get_dependency_versions()
for dep, version in deps.items():
    print(f"{dep}: {version}")
```

### Pipeline Usage with Metadata

```python
from src.pipelines import SceneDetectionPipeline

pipeline = SceneDetectionPipeline()
pipeline.initialize()

# Process video with automatic metadata inclusion
annotations = pipeline.process(
    video_path="video.mp4",
    output_dir="output/"
)

# Metadata is automatically included in saved JSON files
```

## Benefits

1. **Reproducibility**: Full environment and configuration tracking
2. **Debugging**: Clear information about what version and models were used
3. **Audit Trail**: Complete history of how annotations were generated
4. **Version Management**: Easy tracking of VideoAnnotator versions across projects
5. **Compatibility**: Clear dependency requirements for reproducing results
6. **Quality Assurance**: Comprehensive testing ensures metadata accuracy

## Future Enhancements

- **CI/CD Integration**: Automatic version bumping and git information
- **Model Checksums**: Verify model file integrity
- **Performance Metrics**: Include processing time and resource usage
- **Configuration Validation**: Ensure all pipeline parameters are captured
- **Export Formats**: Support for different metadata export formats
