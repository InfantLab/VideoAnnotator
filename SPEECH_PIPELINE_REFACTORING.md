# Stage 3: SpeechPipeline Refactoring Implementation

This document outlines the specific changes needed to refactor the `SpeechPipeline` class to inherit from the new `WhisperBasePipeline`.

## Overview

The `SpeechPipeline` refactoring will involve:
1. Changing the inheritance from `BasePipeline` to `WhisperBasePipeline`
2. Updating the configuration handling to align with the base pipeline
3. Removing duplicated audio extraction and model loading code
4. Adapting the transcription logic to work with the base pipeline
5. Ensuring backward compatibility with existing integrations

## Code Changes

### 1. Update Imports and Inheritance

```python
# OLD
from ..base_pipeline import BasePipeline

class SpeechPipeline(BasePipeline):
    # ...

# NEW
from .whisper_base_pipeline import WhisperBasePipeline

class SpeechPipeline(WhisperBasePipeline):
    # ...
```

### 2. Update Configuration Handling

```python
# OLD
def __init__(self, config: Optional[Dict[str, Any]] = None):
    default_config = {
        "model": "base",
        "language": None,
        "task": "transcribe",
        "beam_size": 5,
        "word_timestamps": True,
        "min_segment_duration": 1.0
    }
    if config:
        default_config.update(config)
    
    super().__init__(default_config)
    self.logger = logging.getLogger(__name__)

    # Initialize model
    self._whisper_model = None

# NEW
def __init__(self, config: Optional[Dict[str, Any]] = None):
    speech_config = {
        "whisper_model": "base",  # Using whisper_model key for base pipeline
        "language": None,
        "task": "transcribe",
        "beam_size": 5,
        "word_timestamps": True,
        "min_segment_duration": 1.0
    }
    
    if config:
        speech_config.update(config)
    
    # Handle legacy "model" key for backward compatibility
    if config and "model" in config and "whisper_model" not in config:
        speech_config["whisper_model"] = config["model"]
    
    super().__init__(speech_config)
    self.logger = logging.getLogger(__name__)
```

### 3. Remove Duplicated initialize() Method

The `initialize()` method can be completely removed as it's handled by `WhisperBasePipeline`.

### 4. Update process() Method

```python
# OLD
def process(self, video_path: str, start_time: float = 0.0, end_time: Optional[float] = None, 
           pps: float = 0.0, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if not self.is_initialized:
        self.initialize()

    try:
        # First extract audio from video if needed
        video_path = Path(video_path)

        # Use FFmpeg to extract audio
        from .ffmpeg_utils import extract_audio_from_video

        if output_dir:
            output_path = Path(output_dir) / f"{video_path.stem}_audio.wav"
        else:
            output_path = video_path.parent / f"{video_path.stem}_audio.wav"

        # Extract audio if it doesn't exist
        if not output_path.exists():
            extracted_audio = extract_audio_from_video(
                video_path=video_path,
                output_path=output_path,
                sample_rate=16000,  # Whisper's native sample rate
                channels=1,  # Mono audio for speech recognition
            )

            if not extracted_audio:
                raise RuntimeError(f"Failed to extract audio from {video_path}")

            audio_path = Path(extracted_audio)
        else:
            audio_path = output_path

        # Perform speech recognition
        result = self.transcribe_audio(audio_path)
        return [result] if result else []

    except Exception as e:
        self.logger.error(f"Error processing speech recognition: {e}")
        return []

# NEW
def process(self, video_path: str, start_time: float = 0.0, end_time: Optional[float] = None, 
           pps: float = 0.0, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if not self.is_initialized:
        self.initialize()

    try:
        # Convert path to Path object
        video_path = Path(video_path)
        
        # Use base pipeline to extract audio
        audio, sample_rate = self.extract_audio_from_video(video_path)
        
        # Transcribe the audio
        result = self.transcribe_audio(audio)
        return [result] if result else []

    except Exception as e:
        self.logger.error(f"Error processing speech recognition: {e}")
        return []
```

### 5. Update transcribe_audio() Method

This method will need significant changes to use the Whisper model and embeddings from the base pipeline. The implementation details will depend on the final `WhisperBasePipeline` implementation.

### 6. Update cleanup() Method

The `cleanup()` method can be removed as it's handled by `WhisperBasePipeline`.

## Testing Strategy

1. Create unit tests that verify the transcription quality matches the original implementation
2. Test with different model sizes to ensure proper configuration handling
3. Verify that memory usage is optimized compared to the original implementation
4. Test backward compatibility with existing interfaces

## Timeline

- Day 1: Update imports, inheritance, and configuration handling
- Day 2: Implement and test transcribe_audio() method
- Day 3: Testing and validation
- Day 4: Documentation and cleanup
