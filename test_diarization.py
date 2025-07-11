"""
Test script for PyAnnote speaker diarization pipeline with FFmpeg audio extraction.

This script demonstrates:
1. Audio extraction from video using FFmpeg
2. Speaker diarization using PyAnnote 
3. Integration with .env file for secure token storage
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
from src.pipelines.audio_processing.diarization_pipeline import (
    DiarizationPipeline, 
    DiarizationPipelineConfig
)
from src.pipelines.audio_processing.ffmpeg_utils import check_ffmpeg_available


def test_diarization_pipeline():
    """Test the diarization pipeline with a sample video."""
    print("=== Testing PyAnnote Diarization Pipeline ===")
    
    # Check prerequisites
    print("\n1. Checking Prerequisites:")
    print(f"   FFmpeg available: {check_ffmpeg_available()}")
    
    # Look for HuggingFace token
    hf_token = os.getenv("HF_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    print(f"   HuggingFace token: {'✓ Found' if hf_token else '✗ Not found'}")
    
    if not hf_token:
        print("\n   ERROR: Please set HF_AUTH_TOKEN in your .env file")
        print("   Steps to fix:")
        print("   1. Edit the .env file in the project root")
        print("   2. Add: HF_AUTH_TOKEN=your_token_here")
        print("   3. Get a token from: https://huggingface.co/settings/tokens")
        print("   4. Accept terms for: https://huggingface.co/pyannote/speaker-diarization-3.1")
        return False
    
    # Check for sample video
    video_dirs = [
        Path("demovideos/babyjokes"),
        Path("demovideos/VEATIC"),
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
    print("\n2. Initializing Diarization Pipeline:")
    try:
        config = DiarizationPipelineConfig(
            huggingface_token=hf_token,
            diarization_model="pyannote/speaker-diarization-3.1",
            use_gpu=True
        )
        
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()
        print("   ✓ Pipeline initialized successfully")
        
    except Exception as e:
        print(f"   ✗ Pipeline initialization failed: {e}")
        return False
    
    # Test processing
    print("\n3. Testing Audio Processing:")
    try:
        print(f"   Processing video: {sample_video}")
        results = pipeline.process(str(sample_video))
        
        if results:
            # Results is a list, get the first result
            diarization_result = results[0] if isinstance(results, list) else results
            
            print("   ✓ Diarization completed!")
            print(f"   Number of speakers: {len(diarization_result.speakers)}")
            print(f"   Number of segments: {len(diarization_result.segments)}")
            print(f"   Total speech time: {diarization_result.total_speech_time:.2f} seconds")
            
            # Print first few segments
            print("\n   First few segments:")
            for i, segment in enumerate(diarization_result.segments[:5]):
                print(f"     Segment {i+1}: Speaker {segment['speaker_id']} "
                      f"from {segment['start_time']:.2f}s to {segment['end_time']:.2f}s")
            
            return True
        else:
            print("   ✗ No diarization results returned")
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


if __name__ == "__main__":
    print("Testing Diarization Pipeline with .env Configuration...")
    success = test_diarization_pipeline()
    if success:
        print("\n🎉 Test completed successfully!")
        print("\nNext steps:")
        print("- Try with different videos in your babyjokes collection")
        print("- Integrate with speech recognition pipeline")
        print("- Run full test suite: python -m pytest tests/test_pipelines.py::TestDiarizationPipeline -v")
    else:
        print("\n❌ Test failed.")
        print("\nTroubleshooting:")
        print("- Check that HF_AUTH_TOKEN is set in .env file")
        print("- Verify your HuggingFace token has access to pyannote models")
        print("- Ensure FFmpeg is installed and available in PATH")
