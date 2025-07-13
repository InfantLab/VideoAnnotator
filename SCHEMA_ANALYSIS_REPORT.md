# Schema Analysis Report: Aggressive Standards Migration

**Goal**: Replace ALL custom schemas with industry standards (no backward compatibility needed)

## 🗑️ **ALL Custom Schemas Should Be REMOVED** ❌

Since we're migrating to use industry standards exclusively and don't need backward compatibility, we can remove ALL custom schema files and replace them with native standard format usage.

### **Remove ALL Schema Files** ❌
1. **`base_schema.py`** - Replace with native format structures
2. **`face_schema.py`** - Replace with COCO format annotations 
3. **`person_schema.py`** - Replace with COCO person/keypoint annotations
4. **`audio_schema.py`** - Replace with WebVTT/RTTM/TextGrid formats
5. **`scene_schema.py`** - Replace with appropriate standard format
6. **`scene_schema_clean.py`** - Remove duplicate
7. **`simple_schemas.py`** - Remove unused schema

## 🎯 **Standards-Based Replacement Strategy**

### Face Detection → **COCO Format**
- **Current**: Custom `FaceDetection`, `FaceEmotion`, `FaceGaze` classes
- **Replace with**: Native COCO annotations using pycocotools
- **Benefits**: Direct COCO compatibility, no custom wrappers

### Person Tracking → **COCO Person/Keypoints**  
- **Current**: Custom `PersonDetection`, `PoseKeypoints`, `PersonTrajectory` classes
- **Replace with**: Native COCO person annotations with keypoint extensions
- **Benefits**: Standard COCO-17 keypoint format, tracking via annotation metadata

### Audio Processing → **Native Audio Standards**
- **Current**: Custom `SpeechRecognition`, `SpeakerDiarization` classes  
- **Replace with**: Direct WebVTT/RTTM/TextGrid objects from respective libraries
- **Benefits**: Native format objects, no conversion needed

### Base Structures → **Standard Format Primitives**
- **Current**: Custom `BoundingBox`, `KeyPoint`, `VideoMetadata` classes
- **Replace with**: Native format structures (COCO bbox arrays, standard metadata)
- **Benefits**: Direct compatibility with standard tools

## 🚀 **Implementation Plan**

### Phase 1: Update Pipeline Imports
Replace all schema imports with direct standard format usage:

```python
# OLD: Custom schemas
from ...schemas.face_schema import FaceDetection, FaceEmotion
from ...schemas.person_schema import PersonDetection, PoseKeypoints  
from ...schemas.base_schema import BoundingBox, KeyPoint
from ...schemas.audio_schema import SpeechRecognition, SpeakerDiarization

# NEW: Direct standard format usage
# Use native dictionaries and standard format objects directly
# Face detection: Native COCO annotation dicts
# Person tracking: Native COCO person/keypoint dicts
# Audio: Direct webvtt.Caption, pyannote.Segment objects
# Base structures: Standard Python types, COCO format arrays
```

### Phase 2: Remove All Schema Files
Delete entire `src/schemas/` directory:
- `base_schema.py` ❌
- `face_schema.py` ❌  
- `person_schema.py` ❌
- `audio_schema.py` ❌
- `scene_schema.py` ❌
- `scene_schema_clean.py` ❌
- `simple_schemas.py` ❌
- `__init__.py` ❌

### Phase 3: Pipeline Refactoring
Update all pipelines to work directly with standard formats:
- **Face pipeline**: Return COCO annotation dicts directly
- **Person pipeline**: Return COCO person/keypoint dicts directly  
- **Audio pipeline**: Return webvtt.Caption/pyannote.Segment objects directly
- **Scene pipeline**: Use appropriate standard format (JSON-LD, etc.)

## 💡 **Benefits of Complete Schema Removal**

1. **True Standards Compliance**: No custom wrapper layers
2. **Reduced Complexity**: Eliminate abstraction overhead
3. **Better Interoperability**: Direct standard format compatibility
4. **Simplified Maintenance**: Use established library APIs instead of custom code
5. **Performance**: No conversion between internal and export formats

## ✅ **Immediate Action Items**

1. **Remove all schema imports** from pipeline files
2. **Update pipelines** to work with native format objects
3. **Delete entire schemas directory** 
4. **Update tests** to expect standard format outputs
5. **Update documentation** to reflect standards-only approach

This aggressive migration fully aligns with the goal of using industry standards instead of custom implementations.
