# VideoAnnotator Schema Design & Standards

## Executive Summary

VideoAnnotator uses **simplified, specification-compliant schemas** that prioritize research flexibility and annotation tool compatibility. Our schema design follows the "simple JSON for interoperability" philosophy.

## Design Principles

### ✅ Current Approach

1. **Simple JSON Structure**: `List[dict]` outputs for maximum compatibility
2. **Flexible ID Support**: String or integer IDs as research needs require
3. **Extension Friendly**: Models can add custom fields via `extra="allow"`
4. **Tool Integration**: Direct export to CVAT, LabelStudio, ELAN
5. **Minimal Validation**: Focus on data exchange over complex rules

## Schema Implementation

### Core Schema Pattern

All VideoAnnotator schemas follow this streamlined pattern:

```python
class FaceEmotion(BaseAnnotation):
    face_id: Union[int, str] = Field(..., description="Face identifier")
    emotion: str = Field(..., description="Dominant emotion")
    person_id: Union[int, str] = Field(..., description="Associated person")

    model_config = ConfigDict(extra="allow")  # ✅ Allows model extensions
```

### Current Schema Types

#### BaseAnnotation (Foundation)

```python
class BaseAnnotation(BaseModel):
    video_id: str = Field(..., description="Video identifier")
    t: float = Field(..., description="Timestamp in seconds")
    confidence: float = Field(default=1.0, description="Confidence score")

    model_config = ConfigDict(extra="allow")
```

#### PersonDetection

```python
class PersonDetection(BaseAnnotation):
    person_id: Union[int, str] = Field(..., description="Person identifier")
    bbox: List[float] = Field(..., description="Bounding box [x,y,w,h]")
```

#### FaceEmotion

```python
class FaceEmotion(BaseAnnotation):
    face_id: Union[int, str] = Field(..., description="Face identifier")
    emotion: str = Field(..., description="Dominant emotion")
    person_id: Union[int, str] = Field(..., description="Associated person")
```

#### SpeechTranscript

```python
class SpeechTranscript(BaseAnnotation):
    text: str = Field(..., description="Transcribed speech")
    start: float = Field(..., description="Start time")
    end: float = Field(..., description="End time")
    speaker_id: Union[int, str] = Field(..., description="Speaker identifier")
```

## Usage Examples

### Creating Annotations

```python
# Person detection
person = PersonDetection(
    video_id="baby_video_001",
    t=12.34,
    person_id="person_1",  # String ID
    bbox=[0.2, 0.3, 0.4, 0.5],
    confidence=0.87,
    custom_field="allowed"  # Extra fields supported
)

# Face emotion
emotion = FaceEmotion(
    video_id="baby_video_001",
    t=12.34,
    face_id=1,  # Integer ID
    emotion="happy",
    person_id="person_1",
    confidence=0.91
)
```

### Serialization

```python
# Export to JSON (annotation tool compatible)
annotations = [person, emotion]
json_output = [ann.model_dump(by_alias=True) for ann in annotations]

# Produces clean JSON:
[
  {
    "type": "person_bbox",
    "video_id": "baby_video_001",
    "t": 12.34,
    "person_id": "person_1",
    "bbox": [0.2, 0.3, 0.4, 0.5],
    "confidence": 0.87,
    "custom_field": "allowed"
  }
]
```

## Standards Compliance

### Research Reproducibility ✅

- Simple data formats for easy analysis
- Flexible ID systems matching domain needs
- Extension support for model evolution

### Tool Integration ✅

- CVAT export ready
- LabelStudio compatible
- ELAN format support

### Performance ✅

- Minimal validation overhead
- Fast serialization/deserialization
- Memory efficient for large datasets

## Benefits of Current Approach

### Development Velocity

- **Faster iteration** with minimal schema constraints
- **Easy debugging** with readable JSON outputs
- **Quick integration** of new models and features

### Research Flexibility

- **String IDs** for human-readable annotations (`"speaker_1"`, `"face_001"`)
- **Integer IDs** for computational efficiency when needed
- **Custom fields** for model-specific metadata

### Production Ready

- **Tool compatibility** ensures research → production pipeline
- **Simple formats** reduce integration complexity
- **Extension support** allows model improvements without breaking changes

---

## Next Steps

The current schema approach successfully balances:

- ✅ **Simplicity** for ease of use
- ✅ **Flexibility** for research needs
- ✅ **Standards compliance** for tool integration
- ✅ **Performance** for production deployment

**Recommendation**: Continue with current simplified schema approach - it perfectly matches research requirements while enabling production scalability.
