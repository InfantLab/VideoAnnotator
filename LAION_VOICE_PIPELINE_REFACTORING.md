# Stage 4: LAIONVoicePipeline Refactoring Implementation

This document outlines the specific changes needed to refactor the `LAIONVoicePipeline` class to inherit from the new `WhisperBasePipeline`.

## Overview

The `LAIONVoicePipeline` refactoring will involve:
1. Changing the inheritance from `BasePipeline` to `WhisperBasePipeline`
2. Updating the configuration handling to align with the base pipeline
3. Removing duplicated Whisper model loading and audio extraction code
4. Maintaining the MLP classifier logic for emotion prediction
5. Adapting the embedding processing to use the base pipeline
6. Ensuring backward compatibility with existing integrations

## Code Changes

### 1. Update Imports and Inheritance

```python
# OLD
from ..base_pipeline import BasePipeline

class LAIONVoicePipeline(BasePipeline):
    # ...

# NEW
from .whisper_base_pipeline import WhisperBasePipeline

class LAIONVoicePipeline(WhisperBasePipeline):
    # ...
```

### 2. Update Configuration Handling

```python
# OLD
def __init__(self, config: Optional[Dict[str, Any]] = None):
    default_config = {
        # Model configuration
        "model_size": "small",  # "small" or "large"
        "whisper_model": "mkrausio/EmoWhisper-AnS-Small-v0.1",
        "model_cache_dir": "./models/laion_voice",
        
        # Audio processing
        "sample_rate": 16000,
        "normalize_audio": True,
        "min_segment_duration": 1.0,
        "max_segment_duration": 30.0,
        
        # Segmentation strategy
        "segmentation_mode": "fixed_interval",
        "segment_overlap": 0.0,
        
        # Integration options
        "enable_diarization": False,
        "enable_scene_alignment": False,
        
        # Output configuration
        "include_raw_scores": False,
        "include_transcription": False,
        "top_k_emotions": 5,
        
        # Performance configuration
        "device": "auto",
    }
    
    merged_config = default_config.copy()
    if config:
        merged_config.update(config)
    super().__init__(merged_config)
    
    self.logger = logging.getLogger(__name__)
    self.whisper_model = None
    self.whisper_processor = None
    self.device = None
    self.classifiers = {}

# NEW
def __init__(self, config: Optional[Dict[str, Any]] = None):
    # Default config for LAION voice pipeline
    laion_config = {
        # Model configuration
        "model_size": "small",  # "small" or "large"
        "whisper_model": "mkrausio/EmoWhisper-AnS-Small-v0.1",
        "model_cache_dir": "./models/laion_voice",
        
        # Audio processing (shared with base pipeline)
        "sample_rate": 16000,
        "normalize_audio": True,
        "min_segment_duration": 1.0,
        "max_segment_duration": 30.0,
        
        # Segmentation strategy
        "segmentation_mode": "fixed_interval",
        "segment_overlap": 0.0,
        
        # Integration options
        "enable_diarization": False,
        "enable_scene_alignment": False,
        
        # Output configuration
        "include_raw_scores": False,
        "include_transcription": False,
        "top_k_emotions": 5,
    }
    
    if config:
        laion_config.update(config)
    
    # Initialize the base pipeline
    super().__init__(laion_config)
    
    self.logger = logging.getLogger(__name__)
    self.classifiers = {}
```

### 3. Update initialize() Method

```python
# OLD
def initialize(self) -> None:
    self.logger.info(f"Initializing LAIONVoicePipeline with model_size: {self.config['model_size']}")
    
    # Determine device
    if self.config["device"] == "auto":
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        self.device = torch.device(self.config["device"])
        
    # Load Whisper model
    self._load_whisper_model()
    
    # Load emotion classifiers
    self._load_classifiers()
    
    # Set model info metadata
    self.set_model_info(self.config["whisper_model"], self.config.get("model_cache_dir"))
    
    self.is_initialized = True
    self.logger.info(f"LAIONVoicePipeline initialized with device: {self.device}")

# NEW
def initialize(self) -> None:
    if self.is_initialized:
        return
        
    # Call base pipeline initialization (handles Whisper model)
    super().initialize()
    
    self.logger.info(f"Initializing LAIONVoicePipeline with model_size: {self.config['model_size']}")
    
    # Load emotion classifiers
    self._load_classifiers()
    
    # Set model info metadata
    self.set_model_info(self.config["whisper_model"], self.config.get("model_cache_dir"))
    
    self.is_initialized = True
    self.logger.info(f"LAIONVoicePipeline initialized with device: {self.device}")
```

### 4. Remove Duplicated Methods

The following methods can be removed as they're handled by `WhisperBasePipeline`:
- `_load_whisper_model()`
- `_extract_audio_from_video()`
- Most of the current audio processing and embedding generation logic

### 5. Update process() Method

```python
# OLD
def process(self, video_path: str, start_time: float = 0.0, end_time: Optional[float] = None,
            pps: float = 0.2, output_dir: Optional[str] = None,
            diarization_results: Optional[Dict[str, Any]] = None,
            scene_detection_results: Optional[List[Dict[str, Any]]] = None,
            include_transcription: bool = False) -> List[Dict[str, Any]]:
    # [Existing implementation with audio extraction and processing]

# NEW
def process(self, video_path: str, start_time: float = 0.0, end_time: Optional[float] = None,
            pps: float = 0.2, output_dir: Optional[str] = None,
            diarization_results: Optional[Dict[str, Any]] = None,
            scene_detection_results: Optional[List[Dict[str, Any]]] = None,
            include_transcription: bool = False) -> List[Dict[str, Any]]:
    if not self.is_initialized:
        self.initialize()
        
    try:
        # Extract audio using base pipeline
        audio, sample_rate = self.extract_audio_from_video(video_path)
        
        # Segment audio (use base pipeline method)
        segments = self.segment_audio(
            audio=audio, 
            sample_rate=sample_rate,
            pps=pps,
            start_time=start_time,
            end_time=end_time,
            min_segment_duration=self.config["min_segment_duration"],
            max_segment_duration=self.config["max_segment_duration"],
            segment_overlap=self.config["segment_overlap"]
        )
        
        # Process segments
        emotion_results = []
        for segment in segments:
            # Get embedding using base pipeline
            embedding = self.get_whisper_embedding(segment["audio"])
            
            # Predict emotions using LAION classifiers
            emotions = self._predict_emotions(embedding)
            
            # Create result
            result = {
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "emotions": emotions,
                "speaker_id": segment.get("speaker_id"),
            }
            
            emotion_results.append(result)
            
        return emotion_results
        
    except Exception as e:
        self.logger.error(f"Error processing voice emotions: {e}")
        return []
```

### 6. Update Emotion Prediction Methods

The emotion prediction methods should remain largely unchanged as they represent the specialized functionality of this pipeline.

### 7. Update cleanup() Method

This can be simplified to focus on just the classifier resources, since Whisper model cleanup is handled by the base pipeline.

## Testing Strategy

1. Create unit tests that verify the emotion predictions match the original implementation
2. Test with both small and large model variants
3. Verify that memory usage is optimized compared to the original implementation
4. Test backward compatibility with existing interfaces

## Timeline

- Day 1: Update imports, inheritance, and configuration handling
- Day 2: Implement and test emotion prediction with base pipeline embeddings
- Day 3: Testing and validation
- Day 4: Documentation and cleanup

## Notes

Fix the bug in the `get_pipeline_info()` method where it's trying to access `self.config["model_variant"]` which doesn't exist (it should be using `model_size` instead).
