# VideoAnnotator Examples

This directory contains example scripts that demonstrate how to use the modernized VideoAnnotator pipeline system.

## Available Examples

### 1. Basic Video Processing (`basic_video_processing.py`)

Demonstrates the complete video annotation pipeline processing a single video file through all available pipelines.

**Usage:**
```bash
python examples/basic_video_processing.py --video_path /path/to/video.mp4 --output_dir /path/to/output
```

**Features:**
- Processes video through all pipelines (scene, person, face, audio)
- Saves individual pipeline results as JSON files
- Comprehensive logging and error handling
- Configurable via YAML configuration files

### 2. Batch Processing (`batch_processing.py`)

Processes multiple videos in parallel using the complete pipeline system.

**Usage:**
```bash
python examples/batch_processing.py --input_dir /path/to/videos --output_dir /path/to/outputs --max_workers 4
```

**Features:**
- Parallel processing of multiple videos
- Configurable number of worker threads
- Batch processing reports and statistics
- CSV summary export
- Individual video result tracking

### 3. Individual Pipeline Testing (`test_individual_pipelines.py`)

Tests individual pipelines in isolation, useful for debugging and development.

**Usage:**
```bash
# Test scene detection pipeline
python examples/test_individual_pipelines.py --pipeline scene --video_path /path/to/video.mp4

# Test audio processing pipeline
python examples/test_individual_pipelines.py --pipeline audio --audio_path /path/to/audio.wav
```

**Features:**
- Test individual pipelines independently
- Detailed pipeline information and capabilities
- Results summary and statistics
- Optional JSON output saving

### 4. Custom Pipeline Configuration (`custom_pipeline_config.py`)

Demonstrates how to create and use custom pipeline configurations for different use cases.

**Usage:**
```bash
python examples/custom_pipeline_config.py --video_path /path/to/video.mp4 --config_type research
```

**Configuration Types:**
- `high_performance`: Maximum accuracy, resource-intensive
- `lightweight`: Fast processing, minimal resources
- `research`: All features enabled, experimental settings

## Configuration Files

All examples use YAML configuration files located in the `configs/` directory:

- `configs/default.yaml` - Balanced settings for general use
- `configs/high_performance.yaml` - High-accuracy settings
- `configs/lightweight.yaml` - Fast, resource-efficient settings

## Common Command Line Arguments

Most examples support these common arguments:

- `--config`: Path to YAML configuration file (default: `configs/default.yaml`)
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--output_dir`: Directory to save results
- `--video_path`: Path to input video file
- `--audio_path`: Path to input audio file (for audio-specific examples)

## Output Formats

All examples generate structured JSON output with consistent schemas:

### Scene Detection Results
```json
{
  "scenes": [
    {
      "scene_id": "scene_001",
      "start_time": 0.0,
      "end_time": 5.2,
      "scene_type": "indoor",
      "confidence": 0.85
    }
  ],
  "total_duration": 120.5,
  "total_scenes": 15
}
```

### Person Tracking Results
```json
{
  "tracks": [
    {
      "track_id": "person_001",
      "detections": [...],
      "poses": [...],
      "duration": 45.2
    }
  ],
  "total_tracks": 3,
  "total_detections": 1250
}
```

### Face Analysis Results
```json
{
  "faces": [
    {
      "face_id": "face_001",
      "emotions": [...],
      "landmarks": [...],
      "gaze": [...]
    }
  ],
  "face_tracks": [...],
  "total_faces": 8
}
```

### Audio Processing Results
```json
{
  "duration": 120.5,
  "speech_transcription": {
    "text": "Hello world...",
    "language": "en",
    "word_timestamps": [...]
  },
  "speaker_diarization": {
    "num_speakers": 2,
    "segments": [...]
  }
}
```

## Environment Setup

Before running the examples, ensure you have:

1. **Installed dependencies:**
   ```bash
   pip install -r requirements.txt
   # or
   conda env create -f environment.yml
   ```

2. **Set up OpenFace 3.0** (if using face analysis):
   Follow the instructions in `INSTALLATION.md`

3. **Downloaded required models:**
   Some pipelines will automatically download models on first use

## Error Handling

All examples include comprehensive error handling:

- **Missing files**: Clear error messages for missing input files
- **Pipeline failures**: Individual pipeline errors don't stop the entire process
- **Resource constraints**: Graceful handling of memory/compute limitations
- **Model loading**: Fallback options when models are unavailable

## Performance Considerations

### Memory Usage
- Large videos may require chunked processing
- Batch processing limits concurrent videos
- Pipeline-specific memory optimizations

### Processing Speed
- Use `--max_workers` for parallel processing
- Choose appropriate model sizes in configurations
- Consider frame skipping for faster processing

### Storage Requirements
- Results are saved as JSON (human-readable but larger)
- Consider compression for large batch processing
- Individual pipeline results can be disabled if not needed

## Extending the Examples

To create your own custom example:

1. **Import the required pipelines:**
   ```python
   from src.pipelines.scene_detection import ScenePipeline, ScenePipelineConfig
   ```

2. **Create configuration:**
   ```python
   config = ScenePipelineConfig(
       threshold=0.3,
       min_scene_length=2.0
   )
   ```

3. **Initialize and run pipeline:**
   ```python
   pipeline = ScenePipeline(config)
   results = pipeline.process_video(video_path)
   ```

4. **Handle results:**
   ```python
   with open('results.json', 'w') as f:
       json.dump(results, f, indent=2, default=str)
   ```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Model download failures**: Check internet connection and disk space
3. **CUDA/GPU issues**: Verify GPU drivers and CUDA installation
4. **Memory errors**: Reduce batch sizes or use lightweight configurations

### Debug Mode

Enable debug logging for detailed information:
```bash
python examples/basic_video_processing.py --log_level DEBUG --video_path video.mp4
```

### Performance Profiling

For performance analysis, consider adding timing information:
```python
import time
start_time = time.time()
results = pipeline.process_video(video_path)
processing_time = time.time() - start_time
```

## Contributing

To contribute new examples:

1. Follow the existing code structure and naming conventions
2. Include comprehensive error handling and logging
3. Add configuration options for flexibility
4. Document usage and output formats
5. Test with various input types and edge cases
