# VideoAnnotator FOSS Library Integration Plan

## Overview

**Goal**: Replace custom schema implementations with established FOSS libraries that are already in our requirements.txt

**Strategy**: Use native FOSS library formats directly instead of wrapping them in custom Pydantic models

## Available FOSS Libraries (Already Installed)

### 1. **pycocotools** - Official COCO Format Library
```python
# Instead of custom COCOPersonDetection class
from pycocotools.coco import COCO
from pycocotools import mask
```

### 2. **audformat** - Comprehensive Audio Annotation Library  
```python
# Instead of custom audio schemas
import audformat
```

### 3. **praatio** - TextGrid (Praat) Format Support
```python
# Instead of custom speech schemas
import praatio
```

### 4. **webvtt-py** - WebVTT Subtitle/Caption Format
```python
# Instead of custom WebVTT schemas
import webvtt
```

### 5. **pyannote.core** - Audio Annotation Data Structures
```python
# Instead of custom speaker diarization schemas
from pyannote.core import Annotation, Segment
```

## Migration Strategy

### Phase 1: Use Native COCO Format (pycocotools)

**Replace**: `src/schemas/coco_extensions.py` and custom COCO classes
**With**: Direct pycocotools usage

```python
# OLD: Custom COCO schema
from src.schemas.coco_extensions import COCOPersonAnnotation

# NEW: Native pycocotools
from pycocotools.coco import COCO
import json

# Create COCO dataset using native format
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "person"}]
}

# Add annotations directly as dictionaries (native COCO format)
annotation = {
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "bbox": [x, y, width, height],
    "area": width * height,
    "iscrowd": 0
}
coco_data["annotations"].append(annotation)

# Save and validate with official library
with open("annotations.json", "w") as f:
    json.dump(coco_data, f)

# Load and validate using official COCO API
coco = COCO("annotations.json")  # Official validation
```

### Phase 2: Use Native Audio Formats

**Replace**: `src/schemas/audio_standards.py` and custom audio schemas  
**With**: Direct audformat and pyannote usage

```python
# OLD: Custom audio schemas
from src.schemas.audio_standards import WebVTTEntry, RTTMEntry

# NEW: Native libraries
import webvtt
from pyannote.core import Annotation, Segment

# WebVTT subtitles using native library
captions = webvtt.WebVTT()
caption = webvtt.Caption(
    start='00:00:00.500',
    end='00:00:02.000',
    text='Hello baby'
)
captions.append(caption)
captions.save('output.vtt')

# Speaker diarization using pyannote.core
diarization = Annotation()
diarization[Segment(0.5, 2.0)] = "SPEAKER_00"
diarization[Segment(2.5, 4.0)] = "SPEAKER_01"

# Export to RTTM format (native pyannote)
with open('output.rttm', 'w') as f:
    diarization.write_rttm(f)
```

### Phase 3: Use TextGrid for Speech Timing

**Replace**: Custom speech recognition schemas
**With**: praatio TextGrid format

```python
# OLD: Custom speech schemas
from src.schemas.simple_schemas import SpeechRecognition

# NEW: TextGrid format (standard in speech research)
import praatio

# Create TextGrid for speech annotations
tg = praatio.Tiers()

# Add speech tier
speech_tier = praatio.IntervalTier('speech', [], 0, 10)
speech_tier.addInterval(0.5, 2.0, "Hello baby")
speech_tier.addInterval(3.0, 4.5, "Peek-a-boo")
tg.addTier(speech_tier)

# Save as TextGrid file (standard format)
tg.save('output.TextGrid')
```

## Implementation Files to Update

### 1. Replace Custom COCO Schema Usage

**Files to update**:
- `src/pipelines/face_analysis/face_pipeline.py`
- `src/pipelines/person_tracking/person_pipeline.py`
- `src/pipelines/scene_detection/scene_pipeline.py`

**Changes**:
```python
# OLD
from src.schemas.coco_extensions import COCOPersonAnnotation
annotation = COCOPersonAnnotation(...)

# NEW  
annotation = {
    "id": annotation_id,
    "image_id": frame_id,
    "category_id": 1,
    "bbox": [x, y, w, h],
    "area": w * h,
    "iscrowd": 0
}
```

### 2. Replace Custom Audio Schema Usage

**Files to update**:
- `src/pipelines/audio_processing/diarization_pipeline.py`
- `src/pipelines/audio_processing/speech_pipeline.py`
- `src/pipelines/audio_processing/audio_pipeline.py`

**Changes**:
```python
# OLD
from src.schemas.audio_standards import RTTMEntry
result = RTTMEntry(...)

# NEW
from pyannote.core import Annotation, Segment
diarization = Annotation()
diarization[Segment(start, end)] = speaker_id
```

### 3. Add Native Library Exports

**Create**: `src/exporters/native_formats.py`

```python
"""
Native format exporters using FOSS libraries directly.
No custom schema conversion needed.
"""

import json
from pycocotools.coco import COCO
import webvtt
from pyannote.core import Annotation
import praatio

def export_coco_json(annotations: list, images: list, output_path: str):
    """Export to COCO format using native structure."""
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}]
    }
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)
    
    # Validate with official library
    coco = COCO(output_path)
    return coco

def export_webvtt(segments: list, output_path: str):
    """Export to WebVTT using native library."""
    captions = webvtt.WebVTT()
    
    for segment in segments:
        caption = webvtt.Caption(
            start=segment['start'],
            end=segment['end'], 
            text=segment['text']
        )
        captions.append(caption)
    
    captions.save(output_path)

def export_textgrid(speech_segments: list, output_path: str):
    """Export to TextGrid using praatio."""
    tg = praatio.Tiers()
    
    speech_tier = praatio.IntervalTier('speech', [], 0, max(s['end'] for s in speech_segments))
    for segment in speech_segments:
        speech_tier.addInterval(segment['start'], segment['end'], segment['text'])
    
    tg.addTier(speech_tier)
    tg.save(output_path)
```

## Benefits of Native Library Usage

### 1. **Validation & Compatibility**
- Official format validation (pycocotools.coco.COCO)
- Guaranteed compatibility with other tools
- Standard parsers handle edge cases

### 2. **Reduced Maintenance**  
- No custom schema maintenance
- Community maintains the libraries
- Automatic updates with pip

### 3. **Performance**
- Optimized native implementations
- No conversion overhead
- Memory efficient

### 4. **Feature Completeness**
- Full format support (not subset)
- Advanced features (COCO segmentation, etc.)
- Extensibility through official APIs

## Migration Steps

### Week 1: COCO Format Migration
1. Update pipelines to output native COCO dictionaries
2. Replace `save_annotations()` with `export_coco_json()`
3. Test with pycocotools validation
4. Verify PyTorch DataLoader compatibility

### Week 2: Audio Format Migration  
1. Update diarization pipeline to use pyannote.core.Annotation
2. Replace WebVTT custom schema with webvtt library
3. Add TextGrid export for speech recognition
4. Test with external audio tools

### Week 3: Schema Cleanup
1. Remove custom schema files
2. Update imports throughout codebase
3. Update documentation
4. Add integration tests with external tools

## Validation Plan

### COCO Format Validation
```python
# Test official COCO API compatibility
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load annotations
coco_gt = COCO('ground_truth.json')
coco_dt = COCO('detections.json')

# Run official evaluation
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
```

### Audio Format Validation
```python
# Test WebVTT compatibility
import webvtt
captions = webvtt.read('output.vtt')  # Should load without errors

# Test TextGrid compatibility  
import praatio
tg = praatio.openTextGrid('output.TextGrid')  # Should open in Praat

# Test pyannote compatibility
from pyannote.core import Annotation
diarization = Annotation.from_rttm('output.rttm')  # Standard format
```

## Expected Results

### Immediate Benefits
- **Native format compliance** - No custom interpretation
- **Tool compatibility** - Direct import into annotation tools
- **Reduced codebase** - Remove 500+ lines of custom schema code
- **Better validation** - Official library error checking

### Long-term Benefits
- **Zero schema maintenance** - Community handles updates
- **New feature access** - Automatic feature additions via library updates
- **Performance improvements** - Optimized native implementations
- **Research compatibility** - Standard formats used in papers

## Files to Remove After Migration

- `src/schemas/coco_extensions.py` (Replace with pycocotools)
- `src/schemas/audio_standards.py` (Replace with webvtt, pyannote.core)
- `src/schemas/standards_compatible_schemas.py` (Replace with native formats)
- `src/validation/coco_validator.py` (Replace with pycocotools validation)

## Files to Keep

- Native library imports only
- Minimal utility functions for data preparation
- Pipeline logic (unchanged)
- Configuration management

This approach follows your SCHEMA_STANDARDS_STRATEGY.md vision perfectly - using established FOSS libraries instead of creating custom implementations.
