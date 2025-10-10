# Speaker Diarization Pipeline

This document explains how to use the speaker diarization functionality in VideoAnnotator.

## Overview

The speaker diarization pipeline identifies different speakers in audio/video files and segments the audio by speaker. It uses the PyAnnote.audio library, which provides state-of-the-art speaker diarization models.

## Setup

### 1. Install Dependencies

Install the required audio processing dependencies:

```bash
pip install -r requirements_audio.txt
```

### 2. HuggingFace Token

You need a HuggingFace token to use PyAnnote models:

1. Go to https://huggingface.co/settings/tokens
2. Create a new token or use an existing one
3. Accept the terms for the PyAnnote models at:

   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

4. Set the token as an environment variable:

```bash
export HUGGINGFACE_TOKEN=your_token_here
```

Or on Windows:

```cmd
set HUGGINGFACE_TOKEN=your_token_here
```

## Usage

### Basic Usage

```python
from src.pipelines.audio_processing import DiarizationPipeline, DiarizationPipelineConfig

# Create configuration
config = DiarizationPipelineConfig(
    diarization_model="pyannote/speaker-diarization-3.1",
    use_gpu=True  # Use GPU if available
)

# Create and initialize pipeline
pipeline = DiarizationPipeline(config)
pipeline.initialize()

# Process a video file
results = pipeline.process("path/to/your/video.mp4")

if results:
    diarization = results[0]
    print(f"Found {len(diarization.speakers)} speakers")
    print(f"Total speech time: {diarization.total_speech_time:.2f} seconds")

    # Print speaker segments
    for segment in diarization.segments:
        speaker = segment['speaker_id']
        start = segment['start_time']
        end = segment['end_time']
        print(f"{speaker}: {start:.2f}s - {end:.2f}s")
```

### Configuration Options

The `DiarizationPipelineConfig` supports the following options:

- `huggingface_token`: Your HuggingFace token (or set HUGGINGFACE_TOKEN env var)
- `diarization_model`: Model to use (default: "pyannote/speaker-diarization-3.1")
- `min_speakers`: Minimum number of speakers (default: 1)
- `max_speakers`: Maximum number of speakers (default: 10)
- `use_gpu`: Whether to use GPU if available (default: True)

### Working with Audio Files

You can also directly process audio files:

```python
# Process audio file directly
audio_result = pipeline.diarize_audio("path/to/audio.wav")
```

The pipeline works best with WAV files, but can also handle other audio formats through automatic conversion.

## Output Format

The diarization pipeline returns a `SpeakerDiarization` object with the following structure:

```python
{
    "speakers": ["speaker_0", "speaker_1", ...],  # List of speaker IDs
    "segments": [
        {
            "speaker_id": "speaker_0",
            "start_time": 0.5,
            "end_time": 3.2,
            "confidence": 1.0
        },
        # ... more segments
    ],
    "total_speech_time": 45.3,  # Total time with speech
    "speaker_change_points": [3.2, 7.8, ...]  # When speakers change
}
```

## Examples

### Run the Test Script

```bash
cd VideoAnnotator
python scripts/test_diarization.py
```

### Run the Example

```bash
cd VideoAnnotator
python examples/diarization_example.py
```

## Troubleshooting

### Common Issues

1. **Missing HuggingFace Token**

   - Make sure HUGGINGFACE_TOKEN is set
   - Verify you've accepted the terms for PyAnnote models

2. **GPU Memory Issues**

   - Set `use_gpu=False` in config if you run out of GPU memory
   - Or use a smaller model if available

3. **Audio Format Issues**

   - The pipeline works best with WAV files
   - Other formats are automatically converted using moviepy

4. **Model Download Issues**
   - Ensure you have internet connection for first-time model download
   - Models are cached locally after first use

### Performance Tips

- Use GPU for faster processing (`use_gpu=True`)
- WAV format gives best performance
- Longer audio files may take significant time to process

## Integration with Legacy Code

The pipeline is designed to work alongside the existing audio processing code in `src/processors/audio_processor.py`. You can use both systems together:

```python
# Legacy approach
from src.processors.audio_processor import diarize_audio

# New pipeline approach
from src.pipelines.audio_processing import DiarizationPipeline
```

## Future Enhancements

- Support for custom speaker models
- Real-time diarization for streaming audio
- Integration with speech recognition for complete transcription with speaker labels
- Voice activity detection preprocessing
