# Person ID Tracking and Labeling Implementation Specification

## Overview

This document specifies the implementation of consistent person identification and labeling across all VideoAnnotator pipelines. The solution provides two key capabilities:

1. **Consistent ID Tracking**: Ensure the same person receives the same numerical ID across all visual analysis pipelines (person tracking, face analysis, pose estimation)
2. **Semantic Labeling**: Allow simple tagging of person IDs with meaningful labels (e.g., "parent", "infant", "teacher")

## Current Architecture Analysis

### Existing Person Tracking Infrastructure

**Primary Pipeline**: `PersonTrackingPipeline` (YOLO11 + ByteTrack)
- **Location**: `src/pipelines/person_tracking/person_pipeline.py`
- **Current ID Field**: `track_id` (integer from ByteTrack)
- **Output Format**: COCO annotations with bounding boxes and keypoints
- **Tracking Persistence**: Built-in across frames within single video

**Secondary Pipelines Using Person Data**:
- `OpenFace3Pipeline`: Face analysis with optional person linking
- `LAIONFacePipeline`: Face analysis with IoU-based person matching
- `FaceAnalysisPipeline`: Basic face detection

### Current Data Flow

```
Video → PersonTrackingPipeline → track_id (1,2,3...)
                ↓
      FaceAnalysisPipeline → attempts IoU matching → person_id
                ↓
      Other pipelines → inconsistent/missing person links
```

## Implementation Design

### 1. Core Person Identity System

#### 1.1 Person Identity Manager

**New Component**: `src/utils/person_identity.py`

```python
from typing import Dict, List, Any, Optional
from src.exporters.native_formats import create_coco_annotation

class PersonIdentityManager:
    """Manages person identities across pipelines and videos using COCO standards."""
    
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.track_to_person_map = {}  # track_id -> person_id
        self.person_labels = {}        # person_id -> {"label": str, "confidence": float}
        self.next_person_id = 1
        
    def register_track(self, track_id: int) -> str:
        """Register a new track and assign person_id."""
        if track_id not in self.track_to_person_map:
            person_id = f"person_{self.video_id}_{self.next_person_id:03d}"
            self.track_to_person_map[track_id] = person_id
            self.next_person_id += 1
        return self.track_to_person_map[track_id]
        
    def get_person_id(self, track_id: int) -> Optional[str]:
        """Get person_id for a track_id."""
        return self.track_to_person_map.get(track_id)
        
    def set_person_label(self, person_id: str, label: str, confidence: float = 1.0):
        """Assign semantic label to person."""
        self.person_labels[person_id] = {
            "label": label,
            "confidence": confidence,
            "method": "manual"
        }
        
    def get_person_label(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get person label information."""
        return self.person_labels.get(person_id)
        
    def link_face_to_person(self, face_bbox: List[float], frame_annotations: List[Dict]) -> Optional[str]:
        """Link face detection to person using IoU matching (COCO format)."""
        best_iou = 0.0
        best_person_id = None
        
        for annotation in frame_annotations:
            if annotation.get('category_id') == 1:  # Person category in COCO
                person_bbox = annotation.get('bbox', [])
                if len(person_bbox) == 4:
                    iou = self._calculate_iou(face_bbox, person_bbox)
                    if iou > best_iou and iou > 0.5:  # IoU threshold
                        best_iou = iou
                        best_person_id = annotation.get('person_id')
        
        return best_person_id
        
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes [x, y, w, h]."""
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2  
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
            
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = (w1 * h1) + (w2 * h2) - intersection
        
        return intersection / union if union > 0 else 0.0
```

#### 1.2 Person ID Format

**Standardized ID Format**: `person_{video_id}_{sequential_number}`
- Example: `person_babyjoke1_001`, `person_babyjoke1_002`
- Ensures uniqueness across videos and sessions
- Human-readable for debugging and annotation

**Alternative Simple Format**: Integer IDs (1, 2, 3...) for computational efficiency
- Configurable via pipeline settings

#### 1.3 Person Labeling Schema

```python
PERSON_LABELS = {
    # Family context
    "parent": {"aliases": ["mother", "father", "mom", "dad"]},
    "infant": {"aliases": ["baby", "child", "toddler"]},
    "sibling": {"aliases": ["brother", "sister"]},
    
    # Educational context  
    "teacher": {"aliases": ["instructor", "educator"]},
    "student": {"aliases": ["pupil", "learner"]},
    
    # Clinical context
    "patient": {"aliases": ["client"]},
    "clinician": {"aliases": ["therapist", "doctor"]},
    
    # Generic
    "person": {"aliases": ["individual", "unknown"]},
}
```

### 2. COCO Format Compliance

Your PersonID implementation now fully follows the established COCO format standards:

#### 2.1 Required COCO Fields
- `id`: Unique annotation ID
- `image_id`: Image/frame identifier  
- `category_id`: COCO category (1 for person)
- `bbox`: Bounding box [x, y, width, height]
- `area`: Bounding box area
- `iscrowd`: 0 for individual objects

#### 2.2 VideoAnnotator Extensions
- `track_id`: YOLO tracking ID (already implemented)
- `person_id`: Consistent person identifier (NEW)
- `timestamp`: Frame timestamp (already implemented)
- `frame_number`: Frame number (already implemented)
- `score`: Detection confidence (already implemented)

#### 2.3 Person Labeling Fields
- `person_label`: Semantic person label (NEW)
- `label_confidence`: Label confidence score (NEW)
- `labeling_method`: How label was assigned (NEW)

All fields are added directly to the annotation object, not nested in `attributes`, following your current COCO implementation patterns.

### 3. Pipeline Integration Strategy

#### 3.1 Person Tracking Pipeline (Primary Source)

**Changes to `PersonTrackingPipeline`**:

```python
class PersonTrackingPipeline(BasePipeline):
    def __init__(self, config):
        # ... existing code ...
        self.identity_manager = None
        
    def process(self, video_path: str, **kwargs) -> List[Dict[str, Any]]:
        video_id = Path(video_path).stem
        self.identity_manager = PersonIdentityManager(video_id)
        
        # ... existing processing using COCO format ...
        
        # Enhanced annotation with person_id (following COCO standards)
        for annotation in frame_annotations:
            track_id = annotation.get('track_id')
            if track_id is not None:
                person_id = self.identity_manager.register_track(track_id)
                # Add person_id as direct field (not nested in attributes)
                annotation['person_id'] = person_id
                
                # Add person labeling fields if available
                label_info = self.identity_manager.get_person_label(person_id)
                if label_info:
                    annotation['person_label'] = label_info['label']
                    annotation['label_confidence'] = label_info['confidence']
```

**Enhanced Output Format** (Following COCO Standards):
```json
{
  "id": 1,
  "image_id": "babyjoke1_frame_123",
  "category_id": 1,
  "bbox": [100, 150, 200, 300],
  "area": 60000,
  "iscrowd": 0,
  "score": 0.87,
  "track_id": 5,
  "person_id": "person_babyjoke1_001",
  "timestamp": 12.34,
  "frame_number": 123,
  "person_label": "infant",
  "label_confidence": 0.95
}
```

#### 3.2 Face Analysis Pipeline Integration

**Changes to `OpenFace3Pipeline`, `LAIONFacePipeline`, `FaceAnalysisPipeline`**:

```python
def process(self, video_path: str, person_tracks: Optional[List[Dict]] = None, **kwargs):
    # Load person tracking results if not provided
    if person_tracks is None:
        person_tracks = self._load_person_tracks(video_path)
    
    # Create identity manager from person tracks
    self.identity_manager = PersonIdentityManager.from_person_tracks(person_tracks)
    
    # ... existing face detection using COCO format ...
    
    # Enhanced face annotation with person linking (following COCO standards)
    for face_annotation in face_results:
        person_id = self._link_face_to_person(face_annotation, person_tracks)
        if person_id:
            # Add person fields directly (not nested in attributes)
            face_annotation['person_id'] = person_id
            label_info = self.identity_manager.get_person_label(person_id)
            if label_info:
                face_annotation['person_label'] = label_info['label']
                face_annotation['label_confidence'] = label_info['confidence']
```

#### 3.3 Cross-Pipeline Data Sharing

**Person Tracks File Format**: `{video_id}_person_tracks.json`
```json
{
  "video_id": "babyjoke1",
  "person_tracks": [
    {
      "person_id": "person_babyjoke1_001",
      "track_id": 1,
      "label": "infant",
      "label_confidence": 0.95,
      "first_appearance": 0.5,
      "last_appearance": 45.2,
      "total_detections": 543,
      "trajectory_summary": {
        "avg_bbox": [150, 200, 180, 320],
        "movement_pattern": "stationary"
      }
    }
  ],
  "labeling_metadata": {
    "labeling_method": "automatic_size_based",
    "confidence_threshold": 0.8,
    "manual_labels": []
  }
}
```

### 4. Automatic Person Labeling

#### 4.1 Size-Based Inference

**Adult vs. Child Detection**:
```python
def infer_person_labels(person_tracks: List[Dict]) -> Dict[str, str]:
    """Automatically infer person labels based on visual cues."""
    
    # Calculate average bounding box sizes
    for track in person_tracks:
        avg_height = calculate_average_bbox_height(track['detections'])
        
        # Heuristic: smaller bounding boxes likely children
        if avg_height < 0.4:  # Normalized height threshold
            track['inferred_label'] = 'infant'
            track['label_confidence'] = 0.7
        else:
            track['inferred_label'] = 'parent'  
            track['label_confidence'] = 0.6
```

#### 4.2 Position-Based Inference

**Spatial Relationship Analysis**:
```python
def analyze_spatial_relationships(person_tracks: List[Dict]) -> Dict[str, float]:
    """Analyze spatial relationships for context clues."""
    
    # Example: person consistently in center might be primary subject
    # Person at edge might be caregiver/observer
    for track in person_tracks:
        center_bias = calculate_center_bias(track['detections'])
        if center_bias > 0.7:
            track['spatial_role'] = 'primary_subject'
        else:
            track['spatial_role'] = 'secondary_participant'
```

### 5. Manual Labeling Interface

#### 5.1 Configuration-Based Labeling

**Video-Level Configuration**: `{video_id}_person_labels.yaml`
```yaml
video_id: "babyjoke1"
person_labels:
  person_babyjoke1_001: 
    label: "infant"
    confidence: 1.0
    method: "manual"
  person_babyjoke1_002:
    label: "parent" 
    confidence: 1.0
    method: "manual"
```

#### 5.2 Interactive Labeling Tool

**Command-Line Tool**: `scripts/label_persons.py`
```bash
# Review and label persons in a video
python scripts/label_persons.py --video babyjoke1.mp4 --output-dir results/

# Apply bulk labeling based on patterns
python scripts/label_persons.py --bulk-label "largest=parent,smallest=infant"
```

### 6. Data Format Standardization

#### 6.1 Enhanced COCO Annotations

**All pipelines adopt standard person_id field** (Following COCO Standards):
```json
{
  "id": 123,
  "image_id": "video123_frame_456", 
  "category_id": 1,
  "bbox": [x, y, w, h],
  "area": 12345.67,
  "iscrowd": 0,
  "score": 0.91,
  "person_id": "person_video123_001",
  "track_id": 5,
  "timestamp": 12.34,
  "person_label": "infant",
  "label_confidence": 0.95,
  "labeling_method": "automatic_size_based"
}
```

#### 6.2 Cross-Pipeline Compatibility

**Backward Compatibility**: 
- Existing pipelines continue to work without person tracking
- `person_id` field is optional in all schemas
- Graceful fallback when person tracking unavailable

**Forward Compatibility**:
- Future facial recognition features can enhance person linking
- Cross-video person identification can extend current system

### 7. Implementation Plan

#### Phase 1: Core Infrastructure (Week 1-2)
1. Implement `PersonIdentityManager` class
2. Add person_id support to `PersonTrackingPipeline`
3. Create person tracks file format and I/O utilities
4. Update COCO annotation schemas

#### Phase 2: Pipeline Integration (Week 3-4)  
1. Integrate person linking in `OpenFace3Pipeline`
2. Integrate person linking in `LAIONFacePipeline`
3. Add IoU-based face-to-person matching utilities
4. Update demo scripts and configuration files

#### Phase 3: Labeling System (Week 5-6)
1. Implement automatic labeling heuristics
2. Create manual labeling configuration system
3. Build command-line labeling tool
4. Add labeling validation and confidence scoring

#### Phase 4: Testing & Documentation (Week 7-8)
1. Comprehensive testing across all pipelines
2. Update documentation and examples
3. Performance testing with multi-person videos
4. Integration testing with existing workflows

### 8. Configuration Options

#### 8.1 Person ID Settings

```yaml
person_identification:
  # ID format options
  id_format: "semantic"  # "semantic" or "integer"
  id_prefix: "person"    # Used with semantic format
  
  # Automatic labeling
  enable_auto_labeling: true
  auto_labeling_confidence_threshold: 0.7
  
  # Face-to-person linking
  iou_threshold: 0.5
  temporal_consistency_window: 5  # frames
  
  # Manual labeling
  labels_config_file: "person_labels.yaml"
  allow_label_override: true
```

#### 8.2 Pipeline-Specific Settings

```yaml
person_tracking:
  # ... existing settings ...
  export_person_tracks: true
  person_tracks_format: "json"

face_analysis:
  # ... existing settings ...
  link_to_persons: true
  require_person_id: false  # Graceful fallback
  
openface3_analysis:
  # ... existing settings ...  
  link_to_persons: true
  person_linking_method: "iou"  # "iou" or "embedding"
```

### 9. Testing Strategy

#### 9.1 Unit Tests
- `PersonIdentityManager` functionality
- Person ID assignment and retrieval
- Face-to-person linking algorithms
- Configuration loading and validation

#### 9.2 Integration Tests
- Multi-pipeline processing with person tracking
- Person ID consistency across pipelines
- Backward compatibility with existing data

#### 9.3 End-to-End Tests
- Complete video processing with person identification
- Manual labeling workflow
- Data export and import scenarios

### 10. Migration and Compatibility

#### 10.1 Existing Data Migration
- Script to add person_id fields to existing annotations
- Backward compatibility mode for old data formats
- Gradual migration path for large datasets

#### 10.2 API Compatibility
- All existing pipeline APIs remain unchanged
- Person tracking features are opt-in
- Graceful degradation when person tracking unavailable

### 11. Future Enhancements

#### 11.1 Facial Recognition Integration
- Face embedding-based person identification
- Cross-video person recognition
- Person identity persistence across sessions

#### 11.2 Advanced Labeling Features
- Machine learning-based automatic labeling
- Context-aware role detection
- Integration with annotation tools (CVAT, LabelStudio)

#### 11.3 Performance Optimization
- Real-time person tracking for live video
- Distributed processing for large video datasets
- GPU acceleration for person matching algorithms

## Implementation Status

### ✅ Phase 1: Core Infrastructure (COMPLETED)
- **PersonIdentityManager** class implemented in `src/utils/person_identity.py`
- Person ID support added to `PersonTrackingPipeline`
- Person tracks file format and I/O utilities created
- COCO annotation schemas updated with person_id fields

### ✅ Phase 2: Face Pipeline Integration (COMPLETED)
- Person linking integrated in all face analysis pipelines:
  - `OpenFace3Pipeline` - Full integration with comprehensive facial analysis
  - `LAIONFacePipeline` - IoU-based person matching with emotion analysis
  - `FaceAnalysisPipeline` - Basic face detection with person linking
- IoU-based face-to-person matching utilities implemented
- Demo scripts and configuration files updated

### 🚧 Phase 3: Labeling System (IN PROGRESS)
- **AutomaticLabelingManager** implemented in `src/utils/automatic_labeling.py`
- Size-based and spatial heuristics for automatic labeling
- Configuration system for person labels (`configs/person_identity.yaml`)
- Command-line labeling tools in `scripts/` directory

### 📋 Phase 4: Testing & Documentation (ONGOING)
- Comprehensive test coverage across all pipelines
- Integration tests for person identity consistency
- Performance testing with multi-person videos
- Documentation updates and examples

## Recent Additions (v1.1.0)

### New Components
- **Person Identity System**: Complete implementation for consistent person tracking
- **Automatic Labeling**: Smart heuristics for person role detection
- **Face-to-Person Linking**: IoU-based matching across all face pipelines
- **Configuration Management**: YAML-based person identity configuration

### Enhanced Pipelines
- All face analysis pipelines now support person identity linking
- Person tracking pipeline exports consistent person IDs
- Standardized COCO format with person identity fields
- Cross-pipeline data sharing through person tracks files

### Testing & Validation
- Integration tests for person identity consistency: `tests/test_phase2_integration.py`
- Person identity unit tests and validation
- Multi-pipeline workflow testing
- Performance benchmarks for person matching algorithms

## Conclusion

The PersonID system has been successfully implemented and integrated across VideoAnnotator, providing:

- **Consistent ID Tracking**: Same person receives same ID across all visual analysis pipelines
- **Semantic Labeling**: Automatic and manual person role detection (parent, infant, etc.)
- **Cross-Pipeline Integration**: Face analysis pipelines link faces to tracked persons
- **COCO Compliance**: Standardized output format compatible with annotation tools
- **Flexible Configuration**: YAML-based settings for different use cases

This foundation enables advanced research workflows with reliable person identification and supports future enhancements like facial recognition and cross-video tracking.
