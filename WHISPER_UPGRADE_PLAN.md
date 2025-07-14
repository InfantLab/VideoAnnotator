# Whisper Pipeline Upgrade Plan

## Overview

After analyzing the current implementation of speech and audio processing pipelines in VideoAnnotator, this plan proposes a significant architectural enhancement to improve code reusability, standardize Whisper model handling, and optimize performance across different use cases. The focus is on creating a shared `WhisperBasePipeline` to support both speech transcription and LAION voice emotion analysis.

## Current State

VideoAnnotator currently has two separate pipelines that utilize Whisper models:

1. **SpeechPipeline** (`speech_pipeline.py`)
   - Uses standard OpenAI Whisper for speech recognition
   - Focuses on transcription with word-level timestamps
   - Produces WebVTT format output

2. **LAIONVoicePipeline** (`laion_voice_pipeline.py`) 
   - Uses specialized Whisper model for audio embedding extraction
   - Processes embeddings through MLP classifiers for emotion prediction
   - Implements memory-efficient processing of audio segments

## Proposed Architecture

### 1. Create a New `WhisperBasePipeline` Class

The cornerstone of this upgrade is the creation of a `WhisperBasePipeline` class in `src/pipelines/audio_processing/whisper_base_pipeline.py` that will:

- Provide a shared foundation for all Whisper-based audio processing
- Handle model loading, audio extraction, and embedding generation
- Support both standard Whisper and Hugging Face Whisper models
- Implement optimized memory management and GPU acceleration

### 2. Refactor Existing Pipelines

Both the `SpeechPipeline` and `LAIONVoicePipeline` will be refactored to extend this base class, focusing only on their specialized functionality:

- **SpeechPipeline**: Focus on transcription and word-level timestamp generation
- **LAIONVoicePipeline**: Focus on emotion prediction and MLP classifier management

### 3. Implement Shared Utilities

Common audio processing utilities will be moved to shared modules:

- Audio extraction and preprocessing
- Device management and GPU acceleration
- Embedding caching and memory optimization

## Implementation Plan

### Phase 1: WhisperBasePipeline Development

1. **Create Base Pipeline Class**
   - Implement `WhisperBasePipeline` with core Whisper functionality
   - Support both standard Whisper and HF Whisper models
   - Add GPU acceleration and memory management
   - Implement audio extraction and preprocessing utilities

2. **Add Embedding Generation**
   - Implement methods for generating and processing Whisper embeddings
   - Add embedding padding/truncation to standardized sequence length
   - Support optional embedding caching for reuse

3. **Add Configuration and Device Management**
   - Implement automatic device detection and management
   - Support half-precision (FP16) for improved performance
   - Add torch.compile support for PyTorch 2.0+

### Phase 2: Speech Pipeline Refactoring

1. **Refactor SpeechPipeline**
   - Update to extend `WhisperBasePipeline`
   - Remove redundant audio extraction and model loading code
   - Focus solely on transcription-specific functionality
   - Ensure backward compatibility with existing interfaces

2. **Testing and Validation**
   - Verify transcription quality matches current implementation
   - Test performance with different Whisper model sizes
   - Validate word-level timestamp accuracy

### Phase 3: LAION Voice Pipeline Integration

1. **Refactor LAIONVoicePipeline**
   - Update to extend `WhisperBasePipeline`
   - Focus on MLP classifier management and emotion prediction
   - Leverage base pipeline for audio segmentation and embedding generation
   - Implement optimized embedding processing based on LAION reference

2. **Enhance Memory Management**
   - Implement staged MLP loading for memory efficiency
   - Add optional CPU offloading for large models
   - Optimize GPU memory usage across pipelines

3. **Validate Emotion Prediction**
   - Ensure emotion scores match reference implementation
   - Test with both small and large model variants
   - Benchmark performance across different hardware configurations

### Phase 4: Advanced Features

1. **Implement Embedding Caching**
   - Add export/import functionality for embeddings
   - Support reuse of embeddings across pipelines
   - Implement efficient temporary storage for long videos

2. **Add Multi-Modal Integration**
   - Support optional integration with diarization pipeline
   - Add scene-based segmentation support
   - Implement voice activity detection for improved segmentation

## Technical Details

### WhisperBasePipeline Architecture

```python
class WhisperBasePipeline(BasePipeline):
    """
    Base pipeline for Whisper-based audio processing.
    
    This provides common functionality for loading Whisper models,
    processing audio, and extracting embeddings.
    """
    
    def __init__(self, config):
        default_config = {
            "whisper_model": "base",  # Standard Whisper model or HF model ID
            "sample_rate": 16000,     # Whisper's preferred sample rate
            "device": "auto",         # "cpu", "cuda", or "auto"
            "use_fp16": True,         # Use half precision when possible
            "cache_dir": "./models/whisper",  # Local cache for models
        }
        # Additional configuration initialization
        
    def initialize(self):
        # Device setup and model loading
        
    def _load_whisper_model(self):
        # Support both standard and HF Whisper models
        
    def get_whisper_embedding(self, audio, pad_or_trim=True):
        # Extract embeddings with proper sequence handling
        
    def extract_audio_from_video(self, video_path):
        # Extract and preprocess audio from video files
        
    def cleanup(self):
        # Resource cleanup and memory management
```

### SpeechPipeline Refactoring

```python
class SpeechPipeline(WhisperBasePipeline):
    """
    Speech recognition pipeline using Whisper.
    """
    
    def __init__(self, config=None):
        default_config = {
            "model": "base",
            "language": None,
            "task": "transcribe",
            "beam_size": 5,
            "word_timestamps": True,
            "min_segment_duration": 1.0
        }
        # Merge with base config and initialize
        
    def process(self, video_path, start_time=0.0, end_time=None, pps=0.0, output_dir=None):
        # Process audio using base pipeline and generate transcription
        
    def transcribe_audio(self, audio_path):
        # Focus only on transcription-specific functionality
```

### LAIONVoicePipeline Refactoring

```python
class LAIONVoicePipeline(WhisperBasePipeline):
    """
    LAION Empathic Insight Voice Pipeline for audio emotion analysis.
    """
    
    def __init__(self, config=None):
        default_config = {
            "model_size": "small",
            "whisper_model": "mkrausio/EmoWhisper-AnS-Small-v0.1",
            # Additional LAION-specific config
        }
        # Merge with base config and initialize
        
    def _load_classifiers(self):
        # Load MLP classifiers for emotion prediction
        
    def _predict_emotions(self, embedding):
        # Focus only on emotion prediction from embeddings
        
    def process(self, video_path, start_time=0.0, end_time=None, pps=0.2, output_dir=None):
        # Process audio using base pipeline for embeddings, then predict emotions
```

## Benefits and Impact

This architecture upgrade offers several significant benefits:

1. **Code Reusability**: Eliminates duplication of Whisper model handling code
2. **Standardization**: Provides consistent audio processing across pipelines
3. **Performance**: Optimizes memory usage and GPU acceleration
4. **Maintainability**: Centralizes bug fixes and improvements
5. **Flexibility**: Supports both standard Whisper and HF models

## Integration with LAION Implementation Plan

The proposed Whisper pipeline upgrade aligns perfectly with Phase 3 of the existing LAION Implementation Plan. This approach maintains all the planned functionality for the `LAIONVoicePipeline` while improving its internal architecture through the shared `WhisperBasePipeline`.

### Updates to LAION Implementation Plan:

1. **Architecture Enhancement**: Add `WhisperBasePipeline` as the foundation for `LAIONVoicePipeline`
2. **Technical Additions**:
   - Add embedding caching for improved performance
   - Implement standardized audio segmentation strategies
   - Support both standard Whisper and HF Whisper models

3. **Implementation Timeline**: The Whisper pipeline upgrade should be completed before or alongside Phase 3 (Voice Pipeline Development) of the LAION Implementation Plan.

## Conclusion

This Whisper Pipeline Upgrade Plan provides a clear path to improving the architecture of VideoAnnotator's audio processing pipelines. By creating a shared foundation for Whisper-based functionality, we can enhance code quality, improve performance, and facilitate future extensions while maintaining all planned functionality for both speech recognition and LAION voice emotion analysis.

The upgrade maintains backward compatibility with existing interfaces while providing a more robust and efficient implementation under the hood. This architectural improvement will benefit both current pipelines and provide a solid foundation for future audio processing extensions.

## Bug Fixes

In the current implementation of the `LAIONVoicePipeline`, there's a bug in the `get_pipeline_info()` method where it's trying to access `self.config["model_variant"]` which doesn't exist (it should be using `model_size` instead). This bug will be fixed as part of the refactoring process.
