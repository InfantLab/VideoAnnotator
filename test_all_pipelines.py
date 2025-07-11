#!/usr/bin/env python3
"""
Quick test script to run all pipelines on demo videos from the demovideos folder.
"""

import sys
import os
import time
import logging
from pathlib import Path
import json
from main import VideoAnnotatorRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_all_pipelines_on_demo_videos():
    """
    Run all available pipelines on videos from the demovideos folder.
    
    This function will:
    1. Find all .mp4 files in the 'demovideos' folder
    2. Run scene detection, person tracking, face analysis, and audio processing
    3. Save results to an output directory
    4. Print a summary of results
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Running All Pipelines on Demo Videos")
    logger.info("=" * 60)
    
    # Set up paths
    demo_dir = Path("demovideos/VEATIC")
    output_dir = Path("demo_results")
    
    # Check if demo directory exists
    if not demo_dir.exists():
        logger.error(f"Demo videos directory not found: {demo_dir}")
        logger.info("Please ensure the 'demovideos/VEATIC' folder exists in the current directory")
        return False
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Find video files
    video_files = list(demo_dir.glob("*.mp4"))
    if not video_files:
        logger.error(f"No MP4 files found in {demo_dir}")
        return False
    
    logger.info(f"Found {len(video_files)} video files:")
    for video in video_files[:5]:  # Show first 5
        logger.info(f"  - {video.name}")
    if len(video_files) > 5:
        logger.info(f"  ... and {len(video_files) - 5} more videos")
    
    # Configuration for all pipelines
    config = {
        "scene_detection": {
            "enabled": True,
            "threshold": 30.0,
            "min_scene_length": 1.0,
            "scene_prompts": [
                "baby", "child", "toy", "nursery", "living room", 
                "kitchen", "bedroom", "outdoor", "playground"
            ]
        },
        "person_tracking": {
            "enabled": True,
            "model": "yolo11n-pose.pt",
            "conf_threshold": 0.4,
            "iou_threshold": 0.7,
            "track_mode": True,
            "tracker": "bytetrack",
            "pose_format": "coco_17",
            "min_keypoint_confidence": 0.3
        },
        "face_analysis": {
            "enabled": True,
            "backend": "mediapipe",
            "detection_confidence": 0.5,
            "enable_emotion": True,
            "enable_gaze": False,  # Disable gaze for faster processing
            "enable_landmarks": True
        },
        "audio_processing": {
            "enabled": True,
            "whisper_model": "tiny",  # Use smallest model for speed
            "sample_rate": 16000,
            "enable_diarization": False,  # Will be enabled if HF token available
            "enable_speech_recognition": True,
            "word_timestamps": True
        },
        "diarization": {
            "enabled": False,  # Will be enabled if HF token available
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "min_speakers": 1,
            "max_speakers": 10,
            "use_gpu": True
        }
    }
    
    try:
        # Check for HuggingFace token for diarization
        hf_token = os.getenv("HF_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            logger.info("✓ HuggingFace token found - enabling diarization pipeline")
            config["diarization"]["enabled"] = True
            config["audio_processing"]["enable_diarization"] = True
        else:
            logger.info("⚠ No HuggingFace token found - diarization will be skipped")
            logger.info("  To enable diarization, set HF_AUTH_TOKEN in your .env file")
        
        # Initialize the runner with configuration
        runner = VideoAnnotatorRunner()
        runner.config = config
        runner._initialize_pipelines()
        
        # If diarization is enabled, also initialize the separate diarization pipeline
        diarization_pipeline = None
        if config["diarization"]["enabled"]:
            try:
                from src.pipelines.audio_processing import DiarizationPipeline, DiarizationPipelineConfig
                diarization_config = DiarizationPipelineConfig(
                    huggingface_token=hf_token,
                    diarization_model=config["diarization"]["diarization_model"],
                    min_speakers=config["diarization"]["min_speakers"],
                    max_speakers=config["diarization"]["max_speakers"],
                    use_gpu=config["diarization"]["use_gpu"]
                )
                diarization_pipeline = DiarizationPipeline(diarization_config)
                diarization_pipeline.initialize()
                logger.info("✓ Standalone diarization pipeline initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize diarization pipeline: {e}")
                diarization_pipeline = None
        
        logger.info("All available pipelines initialized successfully!")
        
        # Process just the first video for testing (change to process all if desired)
        test_video = video_files[0]
        logger.info(f"\nProcessing test video: {test_video.name}")
        
        # Create video-specific output directory
        video_output_dir = output_dir / test_video.stem
        video_output_dir.mkdir(exist_ok=True)
        
        # Process the video through all pipelines
        results = runner.process_video(
            video_path=test_video,
            output_dir=video_output_dir,
            selected_pipelines=None  # Run all pipelines
        )
        
        # Print results summary
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING RESULTS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Video processed: {test_video.name}")
        logger.info(f"Total processing time: {results.get('total_duration', 0):.2f} seconds")
        logger.info(f"Status: {results.get('status', 'unknown')}")
        
        # Pipeline-specific results
        pipeline_results = results.get('pipeline_results', {})
        
        for pipeline_name, pipeline_data in pipeline_results.items():
            logger.info(f"\n{pipeline_name.upper()} PIPELINE:")
            logger.info(f"  Status: {pipeline_data.get('status', 'unknown')}")
            logger.info(f"  Processing time: {pipeline_data.get('processing_time', 0):.2f}s")
            
            # Show specific results based on pipeline type
            pipeline_results_data = pipeline_data.get('results', [])
            
            if pipeline_name == 'scene':
                logger.info(f"  Scenes detected: {len(pipeline_results_data)}")
                if pipeline_results_data:
                    for i, scene in enumerate(pipeline_results_data[:3]):  # Show first 3
                        start_time = getattr(scene, 'start_time', 'N/A')
                        end_time = getattr(scene, 'end_time', 'N/A')
                        scene_type = getattr(scene, 'scene_type', 'N/A')
                        logger.info(f"    Scene {i+1}: {start_time:.1f}s-{end_time:.1f}s ({scene_type})")
                    
            elif pipeline_name == 'person':
                logger.info(f"  Person detections: {len(pipeline_results_data)}")
                if pipeline_results_data:
                    logger.info(f"    First detection at: {getattr(pipeline_results_data[0], 'timestamp', 'N/A'):.1f}s")
                    
            elif pipeline_name == 'face':
                logger.info(f"  Face detections: {len(pipeline_results_data)}")
                if pipeline_results_data:
                    logger.info(f"    First detection at: {getattr(pipeline_results_data[0], 'timestamp', 'N/A'):.1f}s")
                    
            elif pipeline_name == 'audio':
                if isinstance(pipeline_results_data, dict):
                    transcript = pipeline_results_data.get('transcript', '')
                    logger.info(f"  Speech transcript: '{transcript[:100]}...' ({len(transcript)} chars)")
                elif isinstance(pipeline_results_data, list) and pipeline_results_data:
                    # Handle list of SpeechRecognition objects
                    first_result = pipeline_results_data[0]
                    transcript = getattr(first_result, 'transcript', '')
                    logger.info(f"  Speech transcript: '{transcript[:100]}...' ({len(transcript)} chars)")
        
        # Test standalone diarization pipeline if available
        if diarization_pipeline:
            logger.info(f"\nDIARIZATION PIPELINE (STANDALONE):")
            try:
                diarization_start = time.time()
                diarization_results = diarization_pipeline.process(str(test_video))
                diarization_time = time.time() - diarization_start
                
                logger.info(f"  Status: completed")
                logger.info(f"  Processing time: {diarization_time:.2f}s")
                
                if diarization_results:
                    diarization_result = diarization_results[0]
                    logger.info(f"  Speakers detected: {len(diarization_result.speakers)}")
                    logger.info(f"  Speaker segments: {len(diarization_result.segments)}")
                    logger.info(f"  Total speech time: {diarization_result.total_speech_time:.2f}s")
                    
                    # Show first few speaker segments
                    logger.info(f"  First few speaker segments:")
                    for i, segment in enumerate(diarization_result.segments[:5]):
                        speaker = segment['speaker_id']
                        start = segment['start_time']
                        end = segment['end_time']
                        logger.info(f"    {speaker}: {start:.2f}s - {end:.2f}s")
                        if i >= 2:  # Limit to first 3 segments
                            break
                else:
                    logger.info(f"  No diarization results returned")
                    
            except Exception as e:
                logger.error(f"  Diarization failed: {e}")
        
        # Clean up diarization pipeline
        if diarization_pipeline:
            try:
                diarization_pipeline.cleanup()
            except Exception as e:
                logger.warning(f"Diarization cleanup warning: {e}")
        
        # Print output file locations
        logger.info(f"\nOutput files saved to: {video_output_dir}")
        output_files = list(video_output_dir.glob("*.json"))
        for output_file in output_files:
            logger.info(f"  - {output_file.name}")
        
        logger.info(f"\nTesting completed successfully! ✓")
        logger.info(f"To process all {len(video_files)} videos, modify this script to loop through all video_files")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_batch_processing_all_videos():
    """
    Alternative function to process ALL videos in the demovideos folder using batch processing.
    WARNING: This will take a long time and process all videos!
    """
    logger.info("Running BATCH processing on ALL demo videos")
    logger.info("WARNING: This will process all videos and may take a long time!")
    
    demo_dir = Path("demovideos")
    output_dir = Path("demo_results_batch")
    
    if not demo_dir.exists():
        logger.error(f"Demo videos directory not found: {demo_dir}")
        return False
    
    # Initialize runner with config
    runner = VideoAnnotatorRunner()
    
    # Process all videos in batch
    try:
        batch_results = runner.process_videos_batch(
            input_dir=demo_dir,
            output_dir=output_dir,
            selected_pipelines=None,  # Run all pipelines
            max_workers=2  # Use 2 parallel workers
        )
        
        logger.info("Batch processing completed!")
        logger.info(f"Total videos: {batch_results['total_videos']}")
        logger.info(f"Completed: {batch_results['summary']['completed']}")
        logger.info(f"Failed: {batch_results['summary']['failed']}")
        
        return batch_results['summary']['failed'] == 0
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return False


if __name__ == "__main__":
    """
    Run this script to test all pipelines on demo videos.
    
    Usage:
        python test_all_pipelines.py              # Process one test video
        python test_all_pipelines.py --batch      # Process ALL videos (takes long time)
    """
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        success = run_batch_processing_all_videos()
    else:
        success = run_all_pipelines_on_demo_videos()
    
    sys.exit(0 if success else 1)
