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
from src.pipelines.scene_detection.scene_pipeline_legacy import SceneDetectionPipeline
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline
from src.pipelines.face_analysis.face_pipeline import FaceAnalysisPipeline
from src.pipelines.audio_processing.audio_pipeline import AudioPipeline
from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline
from src.pipelines.audio_processing.diarization_pipeline import DiarizationPipeline

# Import annotation schemas (not result classes)
from src.schemas.scene_schema import SceneAnnotation
from src.schemas.person_schema import PersonDetection
from src.schemas.face_schema import FaceDetection
from src.schemas.audio_schema import SpeechRecognition, SpeakerDiarization

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
        # Check first result is a scene annotation
        first_result = results[0]
        assert hasattr(first_result, 'scene_id') or hasattr(first_result, 'start_time')
    
    # Save result to data directory
    output_file = output_dir / f"scene_detection_{video_path.stem}.json"
    with open(output_file, 'w') as f:
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, 'model_dump'):
                serializable_results.append(result.model_dump())
            elif hasattr(result, 'to_dict'):
                serializable_results.append(result.to_dict())
            else:
                # Fallback for basic objects
                serializable_results.append(str(result))
        json.dump(serializable_results, f, indent=2, cls=DateTimeEncoder)
    
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
        # Check first result is a person detection
        first_result = results[0]
        assert hasattr(first_result, 'person_id') or hasattr(first_result, 'bbox') or hasattr(first_result, 'timestamp')
    
    # Save result to data directory
    output_file = output_dir / f"person_tracking_{video_path.stem}.json"
    with open(output_file, 'w') as f:
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, 'model_dump'):
                serializable_results.append(result.model_dump())
            elif hasattr(result, 'to_dict'):
                serializable_results.append(result.to_dict())
            else:
                # Fallback for basic objects
                serializable_results.append(str(result))
        json.dump(serializable_results, f, indent=2, cls=DateTimeEncoder)
    
    logger.info(f"Person tracking result saved to: {output_file}")
    logger.info(f"Detected {len(results)} person detections")
    return results


def test_diarization_pipeline():
    """Test diarization pipeline with real video."""
    video_path, _ = get_demo_video_path()
    if not video_path:
        pytest.skip("No demo video available")
    
    output_dir = ensure_output_dir()
    
    # Check for HuggingFace token
    hf_token = os.getenv("HF_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning("No HuggingFace token found - skipping diarization test")
        logger.info("To enable diarization: set HF_AUTH_TOKEN environment variable")
        pytest.skip("No HuggingFace token available")
    
    logger.info("Testing Diarization Pipeline")
    pipeline = DiarizationPipeline()
    results = pipeline.process(str(video_path))
    
    # Validate result structure
    assert isinstance(results, list)
    if results:
        # Check first result has speaker diarization info
        first_result = results[0]
        assert hasattr(first_result, 'speaker_id') or hasattr(first_result, 'speakers')
    
    # Save result to data directory
    output_file = output_dir / f"diarization_{video_path.stem}.json"
    with open(output_file, 'w') as f:
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, 'model_dump'):
                serializable_results.append(result.model_dump())
            elif hasattr(result, 'to_dict'):
                serializable_results.append(result.to_dict())
            else:
                # Fallback for basic objects
                serializable_results.append(str(result))
        json.dump(serializable_results, f, indent=2, cls=DateTimeEncoder)
    
    logger.info(f"Diarization result saved to: {output_file}")
    logger.info(f"Detected diarization with {len(results)} results")
    return results


def test_audio_pipeline():
    """Test audio pipeline with real video."""
    video_path, _ = get_demo_video_path()
    if not video_path:
        pytest.skip("No demo video available")
    
    output_dir = ensure_output_dir()
    
    logger.info("Testing Audio Pipeline")
    pipeline = AudioPipeline()
    results = pipeline.process(str(video_path))
    
    # Validate result structure
    assert isinstance(results, list)
    if results:
        # Check first result has audio info
        first_result = results[0]
        assert hasattr(first_result, 'audio_file') or hasattr(first_result, 'timestamp')
    
    # Save result to data directory
    output_file = output_dir / f"audio_{video_path.stem}.json"
    with open(output_file, 'w') as f:
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, 'model_dump'):
                serializable_results.append(result.model_dump())
            elif hasattr(result, 'to_dict'):
                serializable_results.append(result.to_dict())
            else:
                # Fallback for basic objects
                serializable_results.append(str(result))
        json.dump(serializable_results, f, indent=2, cls=DateTimeEncoder)
    
    logger.info(f"Audio result saved to: {output_file}")
    logger.info(f"Audio processing completed with {len(results)} results")
    return results


def test_speech_recognition_pipeline():
    """Test speech recognition pipeline with real video."""
    video_path, _ = get_demo_video_path()
    if not video_path:
        pytest.skip("No demo video available")
    
    output_dir = ensure_output_dir()
    
    logger.info("Testing Speech Recognition Pipeline")
    pipeline = SpeechPipeline()
    results = pipeline.process(str(video_path))
    
    # Validate result structure
    assert isinstance(results, list)
    if results:
        # Check first result
        first_result = results[0]
        assert hasattr(first_result, 'transcript')
        
        # Save result to data directory
        output_file = output_dir / f"speech_recognition_{video_path.stem}.json"
        with open(output_file, 'w') as f:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                if hasattr(result, 'model_dump'):
                    serializable_results.append(result.model_dump())
                elif hasattr(result, 'to_dict'):
                    serializable_results.append(result.to_dict())
                else:
                    # Fallback for basic objects
                    serializable_results.append({
                        'transcript': getattr(result, 'transcript', ''),
                        'language': getattr(result, 'language', 'unknown'),
                        'timestamp': getattr(result, 'timestamp', 0.0)
                    })
            json.dump(serializable_results, f, indent=2, cls=DateTimeEncoder)
        
        logger.info(f"Speech recognition result saved to: {output_file}")
        logger.info(f"Found {len(results)} speech recognition results")
        
        # Show sample transcripts
        for i, result in enumerate(results[:3]):  # Show first 3
            transcript = getattr(result, 'transcript', 'N/A')[:50]
            logger.info(f"  Result {i+1}: '{transcript}...'")
    else:
        logger.warning("No speech recognition results found")
    
    return results


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
        # Check first result has face detection info
        first_result = results[0]
        assert hasattr(first_result, 'face_id') or hasattr(first_result, 'bbox') or hasattr(first_result, 'timestamp')
    
    # Save result to data directory
    output_file = output_dir / f"face_analysis_{video_path.stem}.json"
    with open(output_file, 'w') as f:
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, 'model_dump'):
                serializable_results.append(result.model_dump())
            elif hasattr(result, 'to_dict'):
                serializable_results.append(result.to_dict())
            else:
                # Fallback for basic objects
                serializable_results.append(str(result))
        json.dump(serializable_results, f, indent=2, cls=DateTimeEncoder)
    
    logger.info(f"Face analysis result saved to: {output_file}")
    logger.info(f"Detected {len(results)} face detections")
    
    # Show emotion summary
    emotions = {}
    for result in results:
        if hasattr(result, 'emotion') and result.emotion:
            emotion = getattr(result.emotion, 'dominant_emotion', 'unknown')
            emotions[emotion] = emotions.get(emotion, 0) + 1
    
    if emotions:
        logger.info("Emotion summary:")
        for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {emotion}: {count} detections")
    
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
        ("Audio Processing", test_audio_pipeline),
        ("Speech Recognition", test_speech_recognition_pipeline),
        ("Face Analysis", test_face_analysis_pipeline),
        ("Diarization", test_diarization_pipeline),
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
