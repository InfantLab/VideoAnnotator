#!/usr/bin/env python3
"""
Demo test script to verify VideoAnnotator functionality with real videos.
Tests both scene detection and person tracking pipelines with versioning.
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline
from src.schemas.scene_schema import SceneAnnotation, SceneSegment
from src.schemas.person_schema import PersonDetection
from src import print_version_info

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


def test_scene_detection_pipeline():
    """Test the scene detection pipeline with a demo video."""
    logger.info("Testing Scene Detection Pipeline")
    
    # Get demo videos
    demo_dir = Path("data/demovideos")
    output_dir = Path("data/demovideos.out")
    
    if not demo_dir.exists():
        logger.error(f"Demo directory not found: {demo_dir}")
        return False
    
    output_dir.mkdir(exist_ok=True)
    
    # Get first available video
    video_files = list(demo_dir.glob("*.mp4"))
    if not video_files:
        logger.error("No MP4 files found in demo directory")
        return False
    
    video_path = video_files[0]
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Available videos: {[v.name for v in video_files]}")
    
    try:
        # Initialize pipeline with minimal config
        config = {
            "threshold": 30.0,
            "min_scene_length": 1.0,
            "enabled": True
        }
        
        pipeline = SceneDetectionPipeline(config)
        pipeline.initialize()
        logger.info("Scene Detection Pipeline initialized successfully")
        
        # Test processing
        try:
            results = pipeline.process(
                video_path=str(video_path),
                start_time=0.0,
                end_time=10.0,  # Process only first 10 seconds
                output_dir=str(output_dir)
            )
            
            logger.info(f"Scene detection completed. Found {len(results)} scenes")
            
            # Check if output files were created with proper metadata
            output_files = list(output_dir.glob(f"{video_path.stem}_scenes.json"))
            if output_files:
                with open(output_files[0], 'r') as f:
                    data = json.load(f)
                
                logger.info("Scene detection output metadata:")
                logger.info(f"  - VideoAnnotator version: {data['metadata']['videoannotator']['version']}")
                logger.info(f"  - Pipeline: {data['metadata']['pipeline']['name']}")
                logger.info(f"  - Git commit: {data['metadata']['videoannotator']['git']['commit_hash'][:8]}")
                logger.info(f"  - Number of scenes: {len(data['annotations'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Scene detection processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
        finally:
            pipeline.cleanup()
            
    except Exception as e:
        logger.error(f"Scene detection pipeline initialization failed: {e}")
        return False


def test_person_tracking_pipeline():
    """Test the person tracking pipeline with a demo video."""
    logger.info("Testing Person Tracking Pipeline")
    
    # Get demo videos
    demo_dir = Path("data/demovideos")
    output_dir = Path("data/demovideos.out")
    
    if not demo_dir.exists():
        logger.error(f"Demo directory not found: {demo_dir}")
        return False
    
    output_dir.mkdir(exist_ok=True)
    
    # Get first available video
    video_files = list(demo_dir.glob("*.mp4"))
    if not video_files:
        logger.error("No MP4 files found in demo directory")
        return False
    
    video_path = video_files[0]
    logger.info(f"Processing video for person tracking: {video_path}")
    logger.info(f"Available videos: {[v.name for v in video_files]}")
    
    try:
        # Initialize pipeline with minimal config
        config = {
            "model": "yolo11n-pose.pt",
            "conf_threshold": 0.4,
            "iou_threshold": 0.7,
            "track_mode": True,
            "min_keypoint_confidence": 0.3
        }
        
        pipeline = PersonTrackingPipeline(config)
        pipeline.initialize()
        logger.info("Person Tracking Pipeline initialized successfully")
        
        # Test processing
        try:
            results = pipeline.process(
                video_path=str(video_path),
                start_time=0.0,
                end_time=5.0,  # Process only first 5 seconds
                pps=2.0,  # Low rate for testing
                output_dir=str(output_dir)
            )
            
            logger.info(f"Person tracking completed. Found {len(results)} detections")
            
            # Check if output files were created with proper metadata
            output_files = list(output_dir.glob(f"{video_path.stem}_person_detections.json"))
            if output_files:
                with open(output_files[0], 'r') as f:
                    data = json.load(f)
                
                logger.info("Person tracking output metadata:")
                logger.info(f"  - VideoAnnotator version: {data['metadata']['videoannotator']['version']}")
                logger.info(f"  - Pipeline: {data['metadata']['pipeline']['name']}")
                logger.info(f"  - Model: {data['metadata']['model']['model_name']}")
                logger.info(f"  - Git commit: {data['metadata']['videoannotator']['git']['commit_hash'][:8]}")
                logger.info(f"  - Number of detections: {len(data['detections'])}")
            else:
                logger.warning("No person detection output files found")
            
            return True
            
        except Exception as e:
            logger.error(f"Person tracking processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
        finally:
            pipeline.cleanup()
            
    except Exception as e:
        logger.error(f"Person tracking pipeline initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_schemas():
    """Test the schema classes."""
    logger.info("Testing Schema Classes")
    
    try:
        # Test SceneSegment creation
        segment = SceneSegment(
            type="scene_segment",
            video_id="test_video",
            timestamp=0.0,
            start_time=0.0,
            end_time=5.0,
            scene_id="scene_001",
            scene_type="living_room"
        )
        
        logger.info(f"SceneSegment created: {segment}")
        
        # Test serialization
        segment_dict = segment.to_dict()
        logger.info(f"SceneSegment serialized: {segment_dict}")
        
        # Test SceneAnnotation (Pydantic model)
        annotation = SceneAnnotation(
            video_id="test_video",
            timestamp=0.0,
            scene_id="scene_001",
            start_time=0.0,
            end_time=5.0,
            change_type="cut"
        )
        
        logger.info(f"SceneAnnotation created: {annotation}")
        logger.info(f"Scene duration: {annotation.duration}")
        
        return True
        
    except Exception as e:
        logger.error(f"Schema testing failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("="*60)
    logger.info("VideoAnnotator Demo Test with Real Videos")
    logger.info("="*60)
    
    # Show version information
    print("\n=== VideoAnnotator Version Information ===")
    print_version_info()
    
    logger.info("\n" + "="*60)
    logger.info("Starting Pipeline Tests")
    logger.info("="*60)
    
    # Test schemas first
    schema_success = test_schemas()
    
    # Test pipelines
    scene_success = test_scene_detection_pipeline()
    person_success = test_person_tracking_pipeline()
    
    # Summary
    logger.info("="*60)
    logger.info("Test Summary:")
    logger.info(f"  Schema Tests: {'PASSED' if schema_success else 'FAILED'}")
    logger.info(f"  Scene Detection: {'PASSED' if scene_success else 'FAILED'}")
    logger.info(f"  Person Tracking: {'PASSED' if person_success else 'FAILED'}")
    
    if schema_success and scene_success and person_success:
        logger.info("SUCCESS: All tests passed! VideoAnnotator is ready for full processing.")
        return 0
    elif schema_success and (scene_success or person_success):
        logger.info("! Some tests passed. Check logs for details.")
        return 1
    else:
        logger.info("X Multiple tests failed. Check implementation.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
