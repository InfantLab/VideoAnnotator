# Size-Based Person Analysis

## Overview

Size-based person analysis is now integrated by default into the VideoAnnotator person tracking pipeline. This feature automatically classifies detected persons as adults ("parent") or children ("infant") based on their relative bounding box heights.

## How It Works

### Core Logic
1. **Detection Grouping**: Groups person detections by `person_id` across video frames
2. **Height Calculation**: Calculates average bounding box height for each person
3. **Normalization**: Normalizes heights relative to the tallest person (0.0 to 1.0 scale)
4. **Classification**: Applies threshold-based classification:
   - `< 0.4` (default) → "infant" 
   - `>= 0.4` → "parent"
5. **Integration**: Updates person annotations with automatic labels

### Default Configuration

The size-based analysis is **enabled by default** in all configuration files:

```yaml
person_tracking:
  person_identity:
    enabled: true
    automatic_labeling:
      enabled: true
      size_based:
        enabled: true
        use_simple_analyzer: true  # Uses simplified analyzer by default
        height_threshold: 0.4
        confidence: 0.7
        adult_label: "parent"
        child_label: "infant"
        min_detections_for_analysis: 2
```

## Pipeline Integration

### Automatic Execution
When you run person tracking, size-based analysis automatically runs after detection:

```python
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline

# Create pipeline (size-based analysis enabled by default)
pipeline = PersonTrackingPipeline()

# Process video - automatic labeling happens automatically
annotations = pipeline.process("video.mp4", output_dir="results/")

# Annotations now include person labels
for ann in annotations:
    if 'person_label' in ann:
        print(f"Person {ann['person_id']}: {ann['person_label']} "
              f"(confidence: {ann['label_confidence']:.2f})")
```

### Output Format
Annotations are enhanced with labeling information:

```json
{
  "id": 123,
  "person_id": "person_video_001",
  "bbox": [100, 150, 200, 300],
  "person_label": "infant",
  "label_confidence": 0.70,
  "labeling_method": "size_based_inference"
}
```

## Configuration Options

### Basic Settings
```yaml
person_tracking:
  person_identity:
    automatic_labeling:
      size_based:
        enabled: true                        # Enable/disable size-based analysis
        height_threshold: 0.4                # Classification threshold (0.0-1.0)
        confidence: 0.7                      # Confidence score for labels
        min_detections_for_analysis: 2       # Minimum detections needed
```

### Advanced Settings
```yaml
person_tracking:
  person_identity:
    automatic_labeling:
      confidence_threshold: 0.7             # Global confidence threshold
      size_based:
        adult_label: "parent"               # Label for adults
        child_label: "infant"               # Label for children
        use_simple_analyzer: true           # Use simplified analyzer (recommended)
```

### Alternative Labels
You can customize the labels for different contexts:

```yaml
# Clinical context
size_based:
  adult_label: "clinician"
  child_label: "patient"

# Educational context  
size_based:
  adult_label: "teacher"
  child_label: "student"
```

## Performance Characteristics

### Speed
- **Minimal Overhead**: Adds ~1-2% processing time to person tracking
- **Efficient**: Operates on already-computed bounding boxes
- **Scalable**: Handles multiple persons efficiently

### Accuracy
- **Best For**: Clear adult-child size differences (parent-infant interactions)
- **Limitations**: May struggle with teenagers, seated persons, or similar-sized adults
- **Confidence**: Provides confidence scores to assess reliability

## Filtering and Quality Control

### Minimum Detections
Only persons with sufficient detections are analyzed:

```yaml
min_detections_for_analysis: 2  # Require at least 2 detections
```

### Confidence Thresholding
Labels below confidence threshold are filtered out:

```yaml
confidence_threshold: 0.7  # Only apply labels with >= 70% confidence
```

### Temporal Consistency
Analysis uses multiple frames for robust classification:
- Averages bounding box heights across all detections
- Reduces impact of single-frame anomalies
- More stable than single-frame classification

## Usage Examples

### Standard Video Processing
```bash
# Process video with default settings (size analysis enabled)
python main.py --input video.mp4 --output results/

# Results will include automatic person labels
```

### Custom Configuration
```bash
# Use custom configuration
python main.py --input video.mp4 --config configs/default.yaml --output results/
```

### Programmatic Usage
```python
from src.utils.size_based_person_analysis import run_size_based_analysis

# Direct analysis of annotations
annotations = [
    {"person_id": "person_001", "bbox": [100, 100, 80, 200]},  # Adult
    {"person_id": "person_002", "bbox": [200, 150, 50, 80]},   # Child
]

results = run_size_based_analysis(annotations, height_threshold=0.4)
print(results)
```

## Troubleshooting

### No Labels Generated
- Check that `person_identity.enabled: true`
- Verify `automatic_labeling.enabled: true`
- Ensure persons have sufficient detections (`min_detections_for_analysis`)
- Check confidence threshold settings

### Incorrect Classifications
- Adjust `height_threshold` (lower = more children, higher = more adults)
- Increase `min_detections_for_analysis` for more stable results
- Check that persons have clear size differences

### Performance Issues
- Size-based analysis should add minimal overhead
- If slow, verify `use_simple_analyzer: true`
- Consider reducing `min_detections_for_analysis`

## Integration with Other Features

### Face Analysis Pipelines
Person labels from size-based analysis are automatically propagated to face analysis:

```python
# Face analysis will include person labels
face_pipeline = OpenFace3Pipeline()
face_results = face_pipeline.process("video.mp4", person_tracks=person_results)

# Face annotations now include person context
for face_ann in face_results:
    if 'person_label' in face_ann:
        print(f"Face belongs to: {face_ann['person_label']}")
```

### Person Tracks Export
Size-based labels are included in person tracks files:

```json
{
  "video_id": "family_video",
  "person_tracks": [
    {
      "person_id": "person_family_video_001",
      "track_id": 1,
      "label": "parent",
      "confidence": 0.75,
      "method": "size_based_inference"
    }
  ]
}
```

## Migration from Manual Labeling

If you were previously using manual labeling, size-based analysis provides automatic baseline labels that can be refined:

1. **Automatic Baseline**: Size-based analysis provides initial labels
2. **Manual Override**: Use manual labeling configs to override specific persons
3. **Confidence Scoring**: Low-confidence automatic labels can be manually verified

## Future Enhancements

The current size-based analysis is a foundation for more advanced features:

- **Multi-modal Analysis**: Combine with position, temporal, and activity cues
- **Learning-based Classification**: Train models on size patterns
- **Cross-video Identity**: Maintain person identities across multiple videos
- **Contextual Adaptation**: Adjust thresholds based on video context

## API Reference

### Main Functions

#### `run_size_based_analysis(annotations, height_threshold=0.4, confidence=0.7)`
Convenience function for size-based person analysis.

**Parameters:**
- `annotations`: List of person detection annotations
- `height_threshold`: Height threshold for adult/child classification
- `confidence`: Confidence score for labels

**Returns:** Dict mapping person_id to label information

#### `SizeBasedPersonAnalyzer(height_threshold=0.4, confidence=0.7)`
Main analyzer class for size-based person classification.

**Methods:**
- `analyze_persons(annotations)`: Analyze and classify persons
- Configuration through constructor parameters

### Configuration Schema

See `configs/person_identity.yaml` for complete configuration options and documentation.