#!/usr/bin/env python3
"""
Comprehensive demo script to test all VideoAnnotator pipelines with real videos.
Tests all available pipelines and saves results to the data/ directory.
"""

import sys
import os
import logging
import json
from datetime import datetime
import pytest
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import all available pipelines
from src.pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline
from src.pipelines.face_analysis.face_pipeline import FaceAnalysisPipeline
from src.pipelines.audio_processing import AudioPipeline

# Note: After standards migration, pipelines return native format dictionaries
# rather than custom schema objects. Individual speech/diarization pipelines
# are now combined in the main AudioPipeline.

from src import print_version_info

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_demo.log')
    ]
)

logger = logging.getLogger(__name__)


def get_demo_video_path():
    """Get the first available demo video from multiple possible locations."""
    # Try multiple possible demo video locations
    possible_locations = [
        Path("data/demovideos"),
        Path("demovideos/babyjokes"),
        Path("babyjokes videos"),
        Path(".")
    ]
    
    for location in possible_locations:
        if location.exists():
            video_files = list(location.glob("*.mp4"))
            if video_files:
                logger.info(f"Found demo videos in: {location}")
                logger.info(f"Available videos: {[v.name for v in video_files[:3]]}...")  # Show first 3
                return video_files[0], location
    
    logger.error("No demo videos found in any expected location")
    logger.info("Expected locations:")
    for loc in possible_locations:
        logger.info(f"  - {loc.absolute()}")
    return None, None


def ensure_output_dir():
    """Ensure the data output directory exists."""
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def test_scene_detection_pipeline():
    """Test scene detection pipeline with real video."""
    video_path, _ = get_demo_video_path()
    if not video_path:
        pytest.skip("No demo video available")
    
    output_dir = ensure_output_dir()
    
    logger.info("Testing Scene Detection Pipeline")
    pipeline = SceneDetectionPipeline()
    results = pipeline.process(str(video_path))
    
    # Validate result structure
    assert isinstance(results, list)
    if results:
        # Check first result is a COCO-format annotation
        first_result = results[0]
        assert isinstance(first_result, dict)
        # COCO annotations should have these fields
        expected_fields = ['id', 'image_id', 'category_id']
        for field in expected_fields:
            assert field in first_result, f"Missing required COCO field: {field}"
    
    # Save result to data directory
    output_file = output_dir / f"scene_detection_{video_path.stem}.json"
    with open(output_file, 'w') as f:
        # Results are already dictionaries from native formats
        json.dump(results, f, indent=2, cls=DateTimeEncoder)
    
    logger.info(f"Scene detection result saved to: {output_file}")
    logger.info(f"Detected {len(results)} scenes")
    return results


def test_person_tracking_pipeline():
    """Test person tracking pipeline with real video."""
    video_path, _ = get_demo_video_path()
    if not video_path:
        pytest.skip("No demo video available")
    
    output_dir = ensure_output_dir()
    
    logger.info("Testing Person Tracking Pipeline")
    pipeline = PersonTrackingPipeline()
    results = pipeline.process(str(video_path))
    
    # Validate result structure
    assert isinstance(results, list)
    if results:
        # Check first result is a COCO-format annotation
        first_result = results[0]
        assert isinstance(first_result, dict)
        # COCO annotations should have these fields
        expected_fields = ['id', 'image_id', 'category_id']
        for field in expected_fields:
            assert field in first_result, f"Missing required COCO field: {field}"
    
    # Save result to data directory
    output_file = output_dir / f"person_tracking_{video_path.stem}.json"
    with open(output_file, 'w') as f:
        # Results are already dictionaries from native formats
        json.dump(results, f, indent=2, cls=DateTimeEncoder)
    
    logger.info(f"Person tracking result saved to: {output_file}")
    logger.info(f"Detected {len(results)} person detections")
    return results


# Note: Diarization is now part of the combined AudioPipeline
# def test_diarization_pipeline():
#     """Test diarization pipeline with real video."""
#     video_path, _ = get_demo_video_path()
#     if not video_path:
#         pytest.skip("No demo video available")
#     
#     output_dir = ensure_output_dir()
#     
#     # Check for HuggingFace token
#     hf_token = os.getenv("HF_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
#     if not hf_token:
#         logger.warning("No HuggingFace token found - skipping diarization test")
#         logger.info("To enable diarization: set HF_AUTH_TOKEN environment variable")
#         pytest.skip("No HuggingFace token available")
#     
#     logger.info("Testing Diarization Pipeline")
#     pipeline = DiarizationPipeline()
#     results = pipeline.process(str(video_path))
#     
#     # Validate result structure
#     assert isinstance(results, dict)  # Diarization pipeline returns dict with RTTM data
#     assert 'turns' in results or 'speakers' in results
#     
#     # Save result to data directory
#     output_file = output_dir / f"diarization_{video_path.stem}.json"
#     with open(output_file, 'w') as f:
#         # Results are already dictionaries from native formats
#         json.dump(results, f, indent=2, cls=DateTimeEncoder)
#     
#     logger.info(f"Diarization result saved to: {output_file}")
#     if isinstance(results, dict):
#         turns_count = len(results.get('turns', []))
#         logger.info(f"Detected diarization with {turns_count} speaker turns")
#     else:
#         logger.info(f"Detected diarization results: {type(results)}")
    return results


def test_audio_pipeline():
    """Test modular audio pipeline with real video."""
    video_path, _ = get_demo_video_path()
    if not video_path:
        pytest.skip("No demo video available")
    
    output_dir = ensure_output_dir()
    
    logger.info("Testing Modular Audio Pipeline")
    pipeline = AudioPipeline()
    results = pipeline.process(str(video_path))
    
    # Validate result structure - modular pipeline returns list of pipeline results
    assert isinstance(results, list)
    if results:
        # Each result should be a pipeline result with separate data streams
        first_result = results[0]
        assert isinstance(first_result, dict)
        # Should have pipeline metadata
        assert 'pipeline' in first_result
        assert 'format' in first_result  
        assert 'data' in first_result
        assert 'metadata' in first_result
    
    # Save result to data directory
    output_file = output_dir / f"audio_{video_path.stem}.json"
    with open(output_file, 'w') as f:
        # Results are already dictionaries from native formats
        json.dump(results, f, indent=2, cls=DateTimeEncoder)
    
    logger.info(f"Audio result saved to: {output_file}")
    logger.info(f"Modular audio processing completed with {len(results)} pipeline results")
    return results


# Note: Speech recognition is now part of the combined AudioPipeline
# def test_speech_recognition_pipeline():
#     """Test speech recognition pipeline with real video."""
#     video_path, _ = get_demo_video_path()
#     if not video_path:
#         pytest.skip("No demo video available")
#     
#     output_dir = ensure_output_dir()
#     
#     logger.info("Testing Speech Recognition Pipeline")
#     pipeline = SpeechPipeline()
#     results = pipeline.process(str(video_path))
#     
#     # Validate result structure
#     assert isinstance(results, list)
#     if results:
#         # Check first result format - should be WebVTT or similar format
#         first_result = results[0]
#         assert isinstance(first_result, dict)
#         # WebVTT captions should have text content
#         assert 'text' in first_result or 'transcript' in first_result
#         
#         # Save result to data directory
#         output_file = output_dir / f"speech_recognition_{video_path.stem}.json"
#         with open(output_file, 'w') as f:
#             # Results are already dictionaries from native formats
#             json.dump(results, f, indent=2, cls=DateTimeEncoder)
#         
#         logger.info(f"Speech recognition result saved to: {output_file}")
#         logger.info(f"Found {len(results)} speech recognition results")
#         
#         # Show sample transcripts
#         for i, result in enumerate(results[:3]):  # Show first 3
#             transcript = result.get('text', result.get('transcript', 'N/A'))[:50]
#             logger.info(f"  Result {i+1}: '{transcript}...'")
#     else:
#         logger.warning("No speech recognition results found")
#     
#     return results


def test_face_analysis_pipeline():
    """Test face analysis pipeline with real video."""
    video_path, _ = get_demo_video_path()
    if not video_path:
        pytest.skip("No demo video available")
    
    output_dir = ensure_output_dir()
    
    logger.info("Testing Face Analysis Pipeline")
    pipeline = FaceAnalysisPipeline()
    results = pipeline.process(str(video_path))
    
    # Validate result structure
    assert isinstance(results, list)
    if results:
        # Check first result is a COCO-format annotation
        first_result = results[0]
        assert isinstance(first_result, dict)
        # COCO annotations should have these fields
        expected_fields = ['id', 'image_id', 'category_id']
        for field in expected_fields:
            assert field in first_result, f"Missing required COCO field: {field}"
    
    # Save result to data directory
    output_file = output_dir / f"face_analysis_{video_path.stem}.json"
    with open(output_file, 'w') as f:
        # Results are already dictionaries from native formats
        json.dump(results, f, indent=2, cls=DateTimeEncoder)
    
    logger.info(f"Face analysis result saved to: {output_file}")
    logger.info(f"Detected {len(results)} face detections")
    
    # Show emotion summary - Note: This will need to be adapted for COCO format
    # For now, just log the result count
    logger.info(f"COCO annotations contain {len(results)} face detections")
    
    return results


def run_comprehensive_demo():
    """Run all pipeline demos and save results to data directory."""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE VIDEO ANNOTATOR PIPELINE DEMO")
    logger.info("=" * 60)
    
    # Show version information
    print("\n=== VideoAnnotator Version Information ===")
    print_version_info()
    print()
    
    video_path, video_dir = get_demo_video_path()
    if not video_path:
        logger.error("Cannot run demo - no video files found")
        return
    
    output_dir = ensure_output_dir()
    logger.info(f"Using demo video: {video_path}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("")
    
    results = {}
    
    # Test all pipelines  
    pipeline_tests = [
        ("Scene Detection", test_scene_detection_pipeline),
        ("Person Tracking", test_person_tracking_pipeline),
        ("Audio Processing", test_audio_pipeline),  # Combines speech + diarization
        ("Face Analysis", test_face_analysis_pipeline),
    ]
    
    for name, test_func in pipeline_tests:
        try:
            logger.info(f"Running {name} Pipeline...")
            result = test_func()
            results[name] = "SUCCESS"
            logger.info(f"[SUCCESS] {name} Pipeline completed successfully")
        except pytest.skip.Exception as e:
            logger.warning(f"[SKIPPED] {name} Pipeline skipped: {str(e)}")
            results[name] = f"SKIPPED: {str(e)}"
        except Exception as e:
            logger.error(f"[FAILED] {name} Pipeline failed: {str(e)}")
            results[name] = f"FAILED: {str(e)}"
        logger.info("-" * 40)
    
    # Summary
    logger.info("DEMO SUMMARY:")
    for name, status in results.items():
        if status == "SUCCESS":
            status_icon = "[SUCCESS]"
        elif status.startswith("SKIPPED"):
            status_icon = "[SKIPPED]"
        else:
            status_icon = "[FAILED]"
        logger.info(f"{status_icon} {name}: {status}")
    
    logger.info(f"\nAll results saved to: {output_dir.absolute()}")
    logger.info("Demo completed!")


def main():
    """Main function for command line execution."""
    run_comprehensive_demo()


if __name__ == "__main__":
    main()
