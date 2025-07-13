# PHASE 2: CLEANUP - Remove All Custom Schema Files

## Current Status ✅
- ✅ **Phase 1 Complete**: Standards-only pipelines created and integrated
- ✅ **Native Formats Module**: Complete FOSS library integration
- ✅ **Main Pipeline**: Updated to use standards-only pipelines
- ✅ **Validation**: All outputs comply with official standards

## Phase 2: Schema Elimination Checklist

### Files to Delete (Custom Schemas)
```bash
# Custom schema files - DELETE ALL
src/schemas/base_schema.py
src/schemas/face_schema.py  
src/schemas/person_schema.py
src/schemas/audio_schema.py
src/schemas/export_schema.py
src/schemas/__init__.py

# Old pipeline implementations - DELETE ALL
src/pipelines/face_analysis/face_pipeline.py
src/pipelines/person_tracking/person_pipeline.py
src/pipelines/audio_processing/audio_pipeline.py

# Schema directory itself
src/schemas/                        # DELETE ENTIRE DIRECTORY
```

### Files to Keep (Standards-Only)
```bash
# Native format integration
src/exporters/native_formats.py    # ✅ KEEP - FOSS library integration

# Standards-only pipelines  
src/pipelines/face_analysis/face_pipeline_standards.py    # ✅ KEEP
src/pipelines/person_tracking/person_pipeline_standards.py # ✅ KEEP  
src/pipelines/audio_processing/audio_pipeline_standards.py # ✅ KEEP

# Updated main runner
main.py                             # ✅ KEEP - Updated to use standards
```

### Dependencies to Update

#### requirements.txt - Add FOSS Libraries
```txt
# Add these for standards compliance
pycocotools>=2.0.7          # Official COCO format
webvtt-py>=0.4.6            # WebVTT standard
pyannote.core>=5.0.0        # RTTM diarization  
praatio>=6.0.0              # TextGrid speech analysis
```

#### Update Import Statements in Test Files
Search and replace in all test files:
```python
# OLD imports to find and remove
from src.schemas.face_schema import FaceDetection, FaceEmotion
from src.schemas.person_schema import PersonDetection, Pose
from src.schemas.audio_schema import AudioSegment, SpeakerTurn

# NEW imports to use instead
from src.exporters.native_formats import (
    create_coco_annotation,
    create_webvtt_caption, 
    create_rttm_turn
)
```

### Cleanup Commands

#### Step 1: Backup Current State (Optional)
```bash
# Create backup before deletion
git add -A
git commit -m "Backup before schema elimination - Phase 2"
git tag "pre-schema-deletion"
```

#### Step 2: Delete Custom Schema Files
```bash
# Remove custom schema directory
rm -rf src/schemas/

# Remove old pipeline implementations
rm src/pipelines/face_analysis/face_pipeline.py
rm src/pipelines/person_tracking/person_pipeline.py  
rm src/pipelines/audio_processing/audio_pipeline.py
```

#### Step 3: Update Dependencies
```bash
# Install new FOSS libraries
pip install pycocotools>=2.0.7 webvtt-py>=0.4.6 pyannote.core>=5.0.0 praatio>=6.0.0
```

#### Step 4: Verify No Remaining References
```bash
# Search for any remaining custom schema imports
grep -r "from src.schemas" .
grep -r "import.*schema" . --include="*.py"

# Should return NO RESULTS after cleanup
```

## Testing After Cleanup

### Unit Tests to Update
1. **Face Analysis Tests**: Update to expect `List[Dict[str, Any]]` (COCO format)
2. **Person Tracking Tests**: Update to expect COCO keypoint annotations  
3. **Audio Processing Tests**: Update to expect WebVTT/RTTM objects

### Integration Test
```python
# Test complete pipeline with standards-only output
python main.py --video_path test_video.mp4 --output_dir test_output/

# Verify outputs are valid standard formats
from pycocotools.coco import COCO
coco_api = COCO("test_output/face_results.json")  # Should load without errors

import webvtt
captions = webvtt.read("test_output/speech.vtt")  # Should parse without errors
```

## Success Criteria ✅

After Phase 2 completion:
- ❌ **Zero custom schema files** remain in codebase
- ✅ **100% standards-based** output formats
- ✅ **All tests pass** with native format expectations
- ✅ **No import errors** from removed schema modules
- ✅ **FOSS library validation** passes on all outputs

## Final Verification Commands

```bash
# 1. Ensure no schema imports remain
find . -name "*.py" -exec grep -l "from src.schemas" {} \; | wc -l
# Expected result: 0

# 2. Verify standards-only pipelines work
python -c "
from src.pipelines.face_analysis.face_pipeline_standards import FaceAnalysisPipelineStandards
from src.pipelines.person_tracking.person_pipeline_standards import PersonTrackingPipelineStandards
from src.pipelines.audio_processing.audio_pipeline_standards import AudioProcessingPipelineStandards
print('✅ All standards-only pipelines import successfully')
"

# 3. Verify native formats integration
python -c "
from src.exporters.native_formats import (
    create_coco_annotation, 
    create_webvtt_caption, 
    create_rttm_turn,
    validate_coco_json,
    validate_webvtt,
    validate_rttm
)
print('✅ Native formats module working')
"
```

## Ready for Execution

The codebase is prepared for **complete schema elimination**. All standards-only replacements are in place and tested. Phase 2 can proceed with confidence.

**User approval received**: "Yes please" - Complete schema elimination approved.

**Next Action**: Execute Phase 2 cleanup to achieve 100% standards-only architecture.
