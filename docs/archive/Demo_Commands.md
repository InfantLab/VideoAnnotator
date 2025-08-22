# Improved Demo Commands

The demo script has been enhanced to match main.py's interface while providing sensible defaults and batch processing capabilities.

## 🎯 Quick Start Commands

### Auto-Detect and Run with Size-Based Analysis (Default)
```bash
# Automatically finds demo video and runs with size-based person analysis enabled
python demo.py
```

### Single Video with Person Tracking
```bash
# Process specific video with person tracking (includes size-based analysis)
python demo.py --video_path "demovideos/babyjokes/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4" --pipeline person
```

### Batch Process All Baby Joke Videos
```bash
# Process all videos in babyjokes folder with person tracking
python demo.py --batch_dir demovideos/babyjokes/ --pipeline person
```

### Recursive Batch Processing
```bash
# Process all videos in demovideos/ recursively with custom output
python demo.py --batch_dir demovideos/ --output_dir my_results/ --recursive --pipeline person
```

## 🚀 Key Improvements

### 1. Same Parameters as main.py
- `--video_path` (instead of `--video`)
- `--batch_dir` for batch processing
- `--output_dir` (instead of `--output`)
- `--pipeline` with same choices: `scene`, `person`, `face`, `audio`
- `--recursive` for recursive directory search
- `--max_workers` for parallel processing

### 2. Sensible Defaults
- **Auto-detect demo videos** if no input specified
- **Auto-generate output directories** with timestamps if not specified
- **Size-based analysis enabled by default** through default.yaml configuration
- **Smart path handling** for both files and directories

### 3. Batch Processing Support
```bash
# Process with parallel workers
python demo.py --batch_dir demovideos/babyjokes/ --max_workers 4 --pipeline person

# Recursive search in subdirectories
python demo.py --batch_dir demovideos/ --recursive --pipeline person
```

### 4. Quality Presets
```bash
# Fast processing (lightweight.yaml)
python demo.py --fast --pipeline person

# High quality processing (high_performance.yaml) 
python demo.py --high_quality --pipeline person
```

## 📋 Complete Command Reference

### Basic Usage
```bash
# Default: auto-detect video, auto-generate output, all pipelines
python demo.py

# Specific video
python demo.py --video_path path/to/video.mp4

# Specific pipelines only
python demo.py --pipeline person face

# Custom output directory
python demo.py --output_dir results/my_demo/
```

### Batch Processing
```bash
# Batch process directory
python demo.py --batch_dir demovideos/babyjokes/

# Recursive batch processing
python demo.py --batch_dir demovideos/ --recursive

# Parallel batch processing
python demo.py --batch_dir demovideos/babyjokes/ --max_workers 4

# Batch with custom output and specific pipelines
python demo.py --batch_dir demovideos/babyjokes/ --output_dir batch_results/ --pipeline person
```

### Configuration Options
```bash
# Use custom config file
python demo.py --config configs/person_identity.yaml

# Fast processing (lightweight.yaml)
python demo.py --fast

# High quality processing (high_performance.yaml)
python demo.py --high_quality

# Custom log level
python demo.py --log_level DEBUG
```

### Information Commands
```bash
# Show version
python demo.py --version

# Show pipeline information
python demo.py --info

# Help message
python demo.py --help
```

## 📊 Expected Output with Size-Based Analysis

When running person tracking, you'll see:

### Log Messages
```
INFO - Using simplified size-based person analysis...
INFO - Size-based analysis completed: 2 persons labeled
INFO - Auto-labeled person_babyvideo_001: parent (confidence=0.70, method=size_based_inference)
INFO - Auto-labeled person_babyvideo_002: infant (confidence=0.70, method=size_based_inference)
```

### Demo Summary
```
[SUCCESS] Person
    Detections: 45
    Unique Persons: 2
    Person Labels: parent, infant
    Time: 12.34s
```

### Output Files
- `{video_name}_person_tracking.json` - COCO format with person labels
- `{video_name}_person_tracks.json` - Person identity summary
- `complete_results.json` - Full processing results

## 🎯 Recommended Commands for Testing Size-Based Analysis

### Single Video Test
```bash
python demo.py --video_path "demovideos/babyjokes/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4" --pipeline person
```

### Batch Test (All Baby Jokes)
```bash
python demo.py --batch_dir demovideos/babyjokes/ --pipeline person --max_workers 2
```

### Fast Test
```bash
python demo.py --fast --pipeline person
```

These commands will automatically run size-based person analysis and show you the results with person labels (parent/infant) based on relative heights!
