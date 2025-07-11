"""
Simple diarization test script that can be run from the project root.

This script demonstrates how to use the diarization pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to test diarization."""
    
    # Check if HuggingFace token is available
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning("HUGGINGFACE_TOKEN not found in environment variables.")
        logger.info("Please set your HuggingFace token to use PyAnnote models:")
        logger.info("export HUGGINGFACE_TOKEN=your_token_here")
        logger.info("You can get a token from: https://huggingface.co/settings/tokens")
        return False
    
    try:
        # Import the diarization classes
        from src.pipelines.audio_processing import DiarizationPipeline, DiarizationPipelineConfig
        
        # Create pipeline config
        config = DiarizationPipelineConfig(
            huggingface_token=hf_token,
            diarization_model="pyannote/speaker-diarization-3.1",
            use_gpu=True
        )
        
        # Initialize pipeline
        pipeline = DiarizationPipeline(config)
        logger.info("Pipeline created successfully")
        
        # Get pipeline info
        info = pipeline.get_pipeline_info()
        logger.info(f"PyAnnote available: {info['requirements']['pyannote_available']}")
        logger.info(f"CUDA available: {info['requirements']['cuda_available']}")
        logger.info(f"Has auth token: {info['requirements']['has_auth_token']}")
        
        # Check if PyAnnote is available
        if not info['requirements']['pyannote_available']:
            logger.error("PyAnnote not available. Please install with: pip install pyannote.audio")
            return False
        
        # Initialize the pipeline
        logger.info("Initializing pipeline...")
        pipeline.initialize()
        logger.info("Pipeline initialized successfully")
        
        # Find a test video file
        test_videos = []
        
        # Look for videos in common locations
        video_dirs = [
            Path("babyjokes videos"),
            Path("data/demovideos"),
            Path("data")
        ]
        
        for video_dir in video_dirs:
            if video_dir.exists():
                test_videos.extend(list(video_dir.glob("*.mp4")))
                test_videos.extend(list(video_dir.glob("*.avi")))
                test_videos.extend(list(video_dir.glob("*.mov")))
        
        if test_videos:
            test_video = test_videos[0]
            logger.info(f"Testing with video: {test_video}")
            
            # Process the video
            logger.info("Starting diarization...")
            results = pipeline.process(str(test_video))
            
            if results:
                diarization = results[0]
                logger.info("Diarization completed!")
                logger.info(f"Number of speakers: {len(diarization.speakers)}")
                logger.info(f"Number of segments: {len(diarization.segments)}")
                logger.info(f"Total speech time: {diarization.total_speech_time:.2f} seconds")
                
                # Print first few segments
                logger.info("First few speaker segments:")
                for i, segment in enumerate(diarization.segments[:5]):
                    logger.info(f"  Segment {i+1}: {segment['speaker_id']} from {segment['start_time']:.2f}s to {segment['end_time']:.2f}s")
                
                return True
            else:
                logger.error("No diarization results returned")
                return False
        else:
            logger.error("No test video files found in common directories")
            logger.info("Please check if videos exist in: babyjokes videos/, data/demovideos/, or data/")
            return False
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure you're running from the project root directory")
        return False
    except Exception as e:
        logger.error(f"Error testing diarization pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing Speaker Diarization Pipeline...")
    print("="*50)
    success = main()
    print("="*50)
    if success:
        print("✅ Diarization test completed successfully!")
    else:
        print("❌ Diarization test failed.")
