"""
Test script for OpenAI Whisper speech recognition pipeline.

This script demonstrates:
1. Audio extraction            if speech_result.words:
                print("   First 10 words                 print(f"   Text: '{result.transcript[:100]}{'...' if len(result.transcript) > 100 else ''}')")               print(f"   Text: '{result.transcript[:100]}{'...' if len(result.transcript) > 100 else ''}')")ith timestamps:")
                for i, word in enumerate(speech_result.words[:10]):om video using FFmpeg
2. Speech recognition using Whisper with word and segment timestamps
3. Integration with .env file for configuration
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import with absolute imports
from src.pipelines.audio_processing.speech_pipeline import (
    SpeechPipeline, 
    SpeechPipelineConfig
)
from src.pipelines.audio_processing.ffmpeg_utils import check_ffmpeg_available


def test_speech_pipeline():
    """Test the speech recognition pipeline with a sample video."""
    print("=== Testing Whisper Speech Recognition Pipeline ===")
    
    # Check prerequisites
    print("\n1. Checking Prerequisites:")
    print(f"   FFmpeg available: {check_ffmpeg_available()}")
    
    # Check if Whisper is available
    try:
        import whisper
        print("   Whisper available: ✓ Found")
    except ImportError:
        print("   Whisper available: ✗ Not found")
        print("   ERROR: Please install Whisper with: pip install openai-whisper")
        return False
    
    # Check for sample video
    video_dirs = [
        Path("demovideos/babyjokes"),
        Path("demovideos/babyjokes videos"),
        Path("babyjokes videos"),
        Path("babyjokes"),
        Path("demovideos"),
        Path("data/demovideos")
    ]
    
    video_dir = None
    for dir_path in video_dirs:
        if dir_path.exists():
            video_dir = dir_path
            break
    
    if not video_dir:
        print(f"\n   ERROR: No video directory found. Checked: {[str(d) for d in video_dirs]}")
        return False
    
    # Find first video file
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"\n   ERROR: No MP4 files found in {video_dir}")
        return False
    
    sample_video = video_files[0]
    print(f"   Sample video: {sample_video}")
    
    # Initialize pipeline
    print("\n2. Initializing Speech Recognition Pipeline:")
    try:
        config = SpeechPipelineConfig(
            model_name="tiny",  # Use smallest model for testing
            language=None,  # Auto-detect language
            word_timestamps=True,
            use_gpu=False  # Use CPU for consistent testing
        )
        
        pipeline = SpeechPipeline(config)
        pipeline.initialize()
        print("   ✓ Pipeline initialized successfully")
        
    except Exception as e:
        print(f"   ✗ Pipeline initialization failed: {e}")
        return False
    
    # Test processing
    print("\n3. Testing Speech Recognition:")
    try:
        print(f"   Processing video: {sample_video}")
        results = pipeline.process(str(sample_video))
        
        if results:
            # Results is a list, get the first result
            speech_result = results[0] if isinstance(results, list) else results
            
            print("   ✓ Speech recognition completed!")
            print(f"   Language detected: {speech_result.language}")
            print(f"   Text length: {len(speech_result.transcript)} characters")
            print(f"   Number of words: {len(speech_result.words) if speech_result.words else 0}")
            print(f"   Number of segments: {speech_result.metadata.get('num_segments', 0)}")
            
            # Show transcribed text
            print(f"\n   Transcribed text:")
            print(f"   '{speech_result.transcript}'")
            
            # Show first few word timestamps
            if speech_result.words:
                print(f"\n   First few words with timestamps:")
                for i, word in enumerate(speech_result.words[:10]):
                    print(f"     '{word['word']}' ({word['start']:.2f}s - {word['end']:.2f}s, conf: {word['confidence']:.2f})")
                    if i >= 4:  # Limit to first 5 words
                        break
            
            # Show segment information
            segments = speech_result.metadata.get('segments', [])
            if segments:
                print(f"\n   Segment breakdown:")
                for i, segment in enumerate(segments[:3]):
                    print(f"     Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
                    print(f"       Text: '{segment['text']}'")
                    print(f"       No speech prob: {segment['no_speech_prob']:.3f}")
                    if i >= 2:  # Limit to first 3 segments
                        break
            
            return True
        else:
            print("   ✗ No speech recognition results returned")
            return False
            
    except Exception as e:
        print(f"   ✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        try:
            pipeline.cleanup()
            print("\n4. Pipeline cleanup completed")
        except Exception as e:
            print(f"\n4. Cleanup warning: {e}")


def test_different_models():
    """Test different Whisper model sizes."""
    print("\n=== Testing Different Whisper Models ===")
    
    # Find a short video for model comparison
    video_dirs = [Path("demovideos/babyjokes"), Path("demovideos/babyjokes videos"), Path("babyjokes videos"), Path("babyjokes")]
    video_files = []
    
    for dir_path in video_dirs:
        if dir_path.exists():
            video_files = list(dir_path.glob("*.mp4"))
            if video_files:
                break
    
    if not video_files:
        print("No video files found for model testing")
        return
    
    sample_video = video_files[0]
    models_to_test = ["tiny", "base"]  # Start with smaller models
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} model ---")
        try:
            config = SpeechPipelineConfig(
                model_name=model_name,
                use_gpu=False,
                word_timestamps=True
            )
            
            pipeline = SpeechPipeline(config)
            pipeline.initialize()
            
            results = pipeline.process(str(sample_video))
            
            if results:
                result = results[0]
                print(f"   Model: {model_name}")
                print(f"   Text: '{result.text[:100]}{'...' if len(result.text) > 100 else ''}'")
                print(f"   Words: {len(result.words) if result.words else 0}")
                print(f"   Language: {result.language}")
            
            pipeline.cleanup()
            
        except Exception as e:
            print(f"   Error with {model_name} model: {e}")


if __name__ == "__main__":
    print("Testing Whisper Speech Recognition Pipeline...")
    success = test_speech_pipeline()
    
    if success:
        print("\n🎉 Basic test completed successfully!")
        
        # Optionally test different models
        test_models = input("\nTest different Whisper models? (y/N): ").lower().strip()
        if test_models.startswith('y'):
            test_different_models()
        
        print("\nNext steps:")
        print("- Try with different videos in your collection")
        print("- Test different model sizes (base, small, medium)")
        print("- Integrate with diarization pipeline for speaker-aware transcription")
        print("- Run full test suite: python -m pytest tests/test_pipelines.py::TestSpeechPipeline -v")
    else:
        print("\n❌ Test failed.")
        print("\nTroubleshooting:")
        print("- Install Whisper: pip install openai-whisper")
        print("- Ensure FFmpeg is installed and available in PATH")
        print("- Check that video files exist in the expected directories")
