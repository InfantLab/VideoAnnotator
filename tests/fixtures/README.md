# Test Fixtures

This directory contains real media files for integration testing.

## Directory Structure

```
tests/fixtures/
├── audio/          # Audio files for speech/diarization tests
│   ├── speech_single_speaker.wav    # 5-10 seconds, one person speaking
│   ├── speech_multiple_speakers.wav # 10-15 seconds, 2-3 people speaking
│   └── README.md
├── video/          # Video files for tracking/analysis tests
│   ├── person_single.mp4            # Short clip with one person
│   ├── person_multiple.mp4          # Short clip with multiple people
│   └── README.md
└── images/         # Images for face/object detection tests
    └── README.md
```

## Requirements

### Audio Files
- **Format**: WAV (16kHz, mono preferred)
- **Duration**: 5-15 seconds max
- **Content**: Real speech (not synthetic/silence)
- **License**: Must be licensed for testing/distribution or created by us
- **Size**: Keep files small (<1MB each)

**Needed for:**
- Speaker diarization tests (requires multiple speakers)
- Speech recognition tests (requires clear speech)
- Audio pipeline integration tests

### Video Files
- **Format**: MP4 (H.264)
- **Duration**: 3-10 seconds max
- **Resolution**: 640x480 or 720p max
- **Content**: Real people/scenes for detection
- **License**: Must be licensed for testing/distribution
- **Size**: Keep files small (<2MB each)

**Needed for:**
- Person tracking tests
- Face detection tests
- Scene detection tests

### Images
- **Format**: JPG/PNG
- **Resolution**: 640x480 or similar
- **Content**: Faces, people, scenes
- **License**: Must be licensed for testing/distribution
- **Size**: <500KB each

## Creating Test Files

### Option 1: Record Your Own
```bash
# Record 10 seconds of speech
ffmpeg -f pulse -i default -t 10 -ar 16000 -ac 1 tests/fixtures/audio/speech_single_speaker.wav

# Extract from existing video
ffmpeg -i input.mp4 -t 5 -vf scale=640:480 -c:v libx264 -preset fast -crf 23 tests/fixtures/video/person_single.mp4
```

### Option 2: Use Public Domain Content
- Find CC0/public domain audio/video
- Ensure licensing allows redistribution
- Document sources in this README

### Option 3: Generate Synthetic (Last Resort)
- Use text-to-speech for audio (but note it may not work well with diarization)
- Use synthetic video generators

## Git LFS (Large File Storage)

If files exceed 1MB, consider using Git LFS:

```bash
# Install Git LFS
git lfs install

# Track media files
git lfs track "tests/fixtures/**/*.wav"
git lfs track "tests/fixtures/**/*.mp4"
git lfs track "tests/fixtures/**/*.jpg"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking for test fixtures"
```

## Adding New Test Files

1. Create the file in the appropriate directory
2. Document the source/license
3. Keep size minimal (<1MB preferred)
4. Update fixture references in conftest.py
5. Test that integration tests pass

## Current Files

### Audio
- **speech_single_speaker.wav** - [To be added] - Single person speaking clearly
- **speech_multiple_speakers.wav** - [To be added] - Conversation with 2-3 people

### Video
- **person_single.mp4** - [To be added] - Short clip with one person visible
- **person_multiple.mp4** - [To be added] - Short clip with multiple people

### Images
- [To be added]

## License Notes

All test fixtures must either be:
1. Created by the VideoAnnotator team (MIT License)
2. Public domain (CC0)
3. Licensed for testing/redistribution

Document sources here:
- [List sources and licenses for each file]
