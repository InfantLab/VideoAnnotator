# Audio Test Fixtures

This directory should contain real audio files with speech for integration testing.

## Required Files

### 1. speech_single_speaker.wav
**Purpose:** Test speech recognition and single-speaker scenarios

**Requirements:**
- Format: WAV, 16kHz, mono
- Duration: 5-10 seconds
- Content: One person speaking clearly in English
- Quality: Good recording (not compressed, minimal background noise)
- Size target: ~500KB

**Example content:**
"Hello, this is a test recording. The quick brown fox jumps over the lazy dog. Testing one two three."

**How to create:**
```bash
# Record from microphone (10 seconds)
ffmpeg -f pulse -i default -t 10 -ar 16000 -ac 1 -sample_fmt s16 speech_single_speaker.wav

# Or convert existing file
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 speech_single_speaker.wav
```

### 2. speech_multiple_speakers.wav
**Purpose:** Test speaker diarization (identifying who speaks when)

**Requirements:**
- Format: WAV, 16kHz, mono
- Duration: 10-15 seconds
- Content: 2-3 different people speaking in turns
- Quality: Good recording, clear speaker changes
- Size target: ~1MB

**Example content:**
```
Person A: "Let's discuss the project timeline."
Person B: "I think we can complete it by next week."
Person A: "That sounds reasonable to me."
Person C: "I agree, let's proceed with that plan."
```

**How to create:**
```bash
# Record conversation
ffmpeg -f pulse -i default -t 15 -ar 16000 -ac 1 -sample_fmt s16 speech_multiple_speakers.wav

# Or mix separate recordings
# 1. Record each person separately
# 2. Use audio editing software (Audacity) to combine
# 3. Export as WAV, 16kHz, mono
```

## Licensing

All audio files must be:
1. Created by VideoAnnotator contributors (MIT License), OR
2. Public domain (CC0), OR
3. Licensed for testing/redistribution

**Document sources:**
- `speech_single_speaker.wav`: [Source/License]
- `speech_multiple_speakers.wav`: [Source/License]

## Testing Without Real Audio

If these files are not present, tests will either:
- Use synthetic sine wave audio (unit tests only)
- Skip with message asking for real audio (integration tests)

## Quick Recording Guide

### Using built-in microphone (Linux/macOS):
```bash
# Check audio devices
arecord -l  # Linux
# or
ffmpeg -f avfoundation -list_devices true -i ""  # macOS

# Record 10 seconds
arecord -f S16_LE -r 16000 -c 1 -d 10 output.wav  # Linux
# or
ffmpeg -f avfoundation -i ":0" -t 10 -ar 16000 -ac 1 output.wav  # macOS
```

### Using existing audio/video:
```bash
# Extract and convert from video
ffmpeg -i video.mp4 -ar 16000 -ac 1 -sample_fmt s16 -t 10 output.wav

# Convert from MP3/M4A
ffmpeg -i audio.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

### Using Text-to-Speech (backup option):
```bash
# Install espeak
sudo apt-get install espeak

# Generate speech
espeak "Hello, this is a test recording." -w temp.wav
ffmpeg -i temp.wav -ar 16000 -ac 1 -sample_fmt s16 speech_single_speaker.wav
```

Note: TTS may not work well for diarization tests as all speakers sound similar.

## Verification

Test that files work:
```bash
# Play audio
ffplay speech_single_speaker.wav

# Check format
ffprobe speech_single_speaker.wav

# Run tests
export HF_AUTH_TOKEN="your_token"
export TEST_INTEGRATION=1
uv run pytest tests/pipelines/test_audio_individual_components.py -k integration
```
