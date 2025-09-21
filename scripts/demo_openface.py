#!/usr/bin/env python3
"""
OpenFace 3.0 Comprehensive Demo Script

This script provides a complete demonstration of OpenFace 3.0 functionality,
including installation testing, parsing validation, and video processing.

Consolidates all OpenFace testing functionality into a single script.
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import cv2
    import yaml
    from src.pipelines.face_analysis.openface3_pipeline import OpenFace3Pipeline, OPENFACE3_AVAILABLE
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('openface_demo.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def test_openface_availability() -> bool:
    """Test if OpenFace 3.0 is properly installed and available."""
    logger.info("Testing OpenFace 3.0 availability...")
    
    if not OPENFACE3_AVAILABLE:
        logger.error("OpenFace 3.0 is not available")
        logger.error("Please install OpenFace 3.0 from: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0")
        return False
    
    logger.info("OpenFace 3.0 is available")
    return True


def test_pipeline_initialization() -> bool:
    """Test pipeline initialization with comprehensive configuration."""
    logger.info("Testing pipeline initialization...")
    
    try:
        # Load comprehensive configuration
        config_path = project_root / "configs" / "openface3_complete.yaml"
        
        if not config_path.exists():
            logger.warning("Complete config not found, using default config")
            config = None
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Force CPU device to avoid CUDA issues
        if config:
            config['device'] = 'cpu'
        else:
            config = {'device': 'cpu'}
        
        # Create pipeline with all features enabled
        pipeline = OpenFace3Pipeline(config)
        
        # Test initialization
        pipeline.initialize()
        logger.info("Pipeline initialized successfully")
        
        # Verify MultitaskPredictor is loaded
        if hasattr(pipeline, 'multitask_predictor') and pipeline.multitask_predictor:
            logger.info("MultitaskPredictor loaded for advanced features")
        else:
            logger.warning("MultitaskPredictor not loaded - advanced features disabled")
        
        # Test pipeline info
        info = pipeline.get_pipeline_info()
        logger.info(f"Pipeline info: {info}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        return False


def test_action_units_parsing() -> bool:
    """Test Action Units parsing functionality."""
    logger.info("Testing Action Units parsing...")
    
    try:
        pipeline = OpenFace3Pipeline({'device': 'cpu'})
        pipeline.initialize()
        
        # Test with realistic dummy tensor data (not dict)
        dummy_tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        
        parsed = pipeline._parse_action_units(dummy_tensor)
        
        if parsed and len(parsed) == 8:
            logger.info("Action Units parsing test passed")
            logger.info(f"   Sample parsed AUs: {list(parsed.keys())[:3]}...")
            return True
        else:
            logger.error("Action Units parsing returned unexpected format")
            return False
            
    except Exception as e:
        logger.error(f"Action Units parsing test failed: {e}")
        return False


def test_head_pose_parsing() -> bool:
    """Test Head Pose parsing functionality."""
    logger.info("Testing Head Pose parsing...")
    
    try:
        pipeline = OpenFace3Pipeline({'device': 'cpu'})
        pipeline.initialize()
        
        # Test with realistic dummy tensor data (not dict)
        dummy_tensor = torch.tensor([[0.1, 0.2]])
        
        parsed = pipeline._parse_head_pose(dummy_tensor)
        
        if parsed and 'pitch' in parsed and 'yaw' in parsed:
            logger.info("Head Pose parsing test passed")
            logger.info(f"   Pitch: {parsed['pitch']:.2f}, Yaw: {parsed['yaw']:.2f}")
            return True
        else:
            logger.error("Head Pose parsing returned unexpected format")
            return False
            
    except Exception as e:
        logger.error(f"Head Pose parsing test failed: {e}")
        return False


def test_emotions_parsing() -> bool:
    """Test Emotions parsing functionality."""
    logger.info("Testing Emotions parsing...")
    
    try:
        pipeline = OpenFace3Pipeline({'device': 'cpu'})
        pipeline.initialize()
        
        # Test with realistic dummy tensor data (not dict)
        dummy_tensor = torch.tensor([[0.1, 0.2, 0.05, 0.3, 0.15, 0.1, 0.05, 0.05]])
        
        parsed = pipeline._parse_emotions(dummy_tensor)
        
        if parsed and isinstance(parsed, dict) and 'dominant' in parsed and 'probabilities' in parsed:
            logger.info("Emotions parsing test passed")
            logger.info(f"   Dominant emotion: {parsed['dominant']}, confidence: {parsed['confidence']:.3f}")
            return True
        else:
            logger.error("Emotions parsing returned unexpected format")
            return False
            
    except Exception as e:
        logger.error(f"Emotions parsing test failed: {e}")
        return False


def test_gaze_parsing() -> bool:
    """Test Gaze parsing functionality."""
    logger.info("Testing Gaze parsing...")
    
    try:
        pipeline = OpenFace3Pipeline({'device': 'cpu'})
        pipeline.initialize()
        
        # Test with realistic dummy tensor data (not dict)
        dummy_tensor = torch.tensor([[0.1, 0.2, 0.3]])
        
        parsed = pipeline._parse_gaze(dummy_tensor)
        
        if parsed and 'direction_x' in parsed:
            logger.info("Gaze parsing test passed")
            logger.info(f"   Gaze direction: x={parsed['direction_x']:.3f}, y={parsed['direction_y']:.3f}, z={parsed['direction_z']:.3f}")
            return True
        else:
            logger.error("Gaze parsing returned unexpected format")
            return False
            
    except Exception as e:
        logger.error(f"Gaze parsing test failed: {e}")
        return False


def test_video_processing() -> bool:
    """Test video processing with a sample video."""
    logger.info("Testing video processing...")
    
    try:
        # Look for demo videos
        demo_video_dir = project_root / "demovideos"
        if not demo_video_dir.exists():
            logger.warning("No demo videos directory found, skipping video test")
            return True
        
        # Find first video file (including subdirectories)
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(demo_video_dir.rglob(ext))
        
        if not video_files:
            logger.warning("No video files found in demovideos, skipping video test")
            return True
        
        video_path = video_files[0]
        logger.info(f"Processing video: {video_path.name}")
        
        # Create pipeline with CPU device
        pipeline = OpenFace3Pipeline({'device': 'cpu'})
        pipeline.initialize()
        
        # Process video
        start_time = time.time()
        results = pipeline.process(str(video_path))
        processing_time = time.time() - start_time
        
        if results and len(results) > 0:
            # results is a list containing a COCO dataset dictionary
            coco_dataset = results[0]
            logger.info(f"Video processing completed in {processing_time:.2f}s")
            logger.info(f"   Annotations found: {len(coco_dataset.get('annotations', []))}")
            logger.info(f"   Dataset keys: {list(coco_dataset.keys())}")
            
            # Analyze results if we have them
            analyze_results(results)
            return True
        else:
            logger.error("Video processing returned no results")
            return False
            
    except Exception as e:
        logger.error(f"Video processing test failed: {e}")
        return False


def analyze_results(results: list) -> None:
    """Analyze and display processing results."""
    logger.info("Analyzing results...")
    
    if not results or len(results) == 0:
        logger.warning("No results to analyze")
        return
    
    # Extract COCO dataset from results list
    coco_dataset = results[0]
    annotations = coco_dataset.get('annotations', [])
    
    # Analysis summary
    total_annotations = len(annotations)
    logger.info(f"Total face annotations: {total_annotations}")
    
    if total_annotations > 0:
        # Feature availability
        features = []
        sample_annotation = annotations[0]
        openface_data = sample_annotation.get('openface3', {})
        
        if 'action_units' in openface_data:
            features.append("Action Units")
        if 'head_pose' in openface_data:
            features.append("Head Pose")
        if 'emotions' in openface_data:
            features.append("Emotions")
        if 'gaze' in openface_data:
            features.append("Gaze")
        
        logger.info(f"Available features: {', '.join(features)}")
        
        # Sample annotation analysis
        logger.info(f"Sample annotation keys: {list(sample_annotation.keys())}")
        if openface_data:
            logger.info(f"OpenFace features: {list(openface_data.keys())}")
    else:
        logger.info("No face annotations found in results")


def run_comprehensive_demo() -> bool:
    """Run the complete OpenFace 3.0 demonstration."""
    logger.info("Starting OpenFace 3.0 Comprehensive Demo")
    logger.info("=" * 60)
    
    success_count = 0
    total_tests = 7
    
    # Test 1: OpenFace availability
    if test_openface_availability():
        success_count += 1
    
    # Test 2: Pipeline initialization
    if test_pipeline_initialization():
        success_count += 1
    
    # Test 3: Action Units parsing
    if test_action_units_parsing():
        success_count += 1
    
    # Test 4: Head Pose parsing
    if test_head_pose_parsing():
        success_count += 1
    
    # Test 5: Emotions parsing
    if test_emotions_parsing():
        success_count += 1
    
    # Test 6: Gaze parsing
    if test_gaze_parsing():
        success_count += 1
    
    # Test 7: Video processing
    if test_video_processing():
        success_count += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Demo completed: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        logger.info("All tests passed! OpenFace 3.0 is working perfectly.")
        return True
    else:
        logger.warning(f"{total_tests - success_count} tests failed. Check logs for details.")
        return False


def main():
    """Main function."""
    try:
        success = run_comprehensive_demo()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Demo failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
