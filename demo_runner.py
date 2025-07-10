#!/usr/bin/env python3
"""
Simple demo runner for scene detection and person tracking.
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
from src.version import get_version_info, create_annotation_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def process_demo_videos():
    """Process all demo videos with scene detection and person tracking."""
    logger.info("Processing Demo Videos with Scene Detection and Person Tracking")
    logger.info("=" * 70)
    
    # Get demo videos and output directory
    demo_dir = Path("data/demovideos")
    output_dir = Path("data/demovideos.out")
    
    if not demo_dir.exists():
        logger.error(f"Demo directory not found: {demo_dir}")
        return False
    
    output_dir.mkdir(exist_ok=True)
    
    # Get all video files
    video_files = list(demo_dir.glob("*.mp4"))
    if not video_files:
        logger.error("No MP4 files found in demo directory")
        return False
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Initialize pipeline configs
    scene_config = {
        "threshold": 15.0,  # Lower threshold for more sensitive detection
        "min_scene_length": 0.5,  # Shorter minimum scenes for demo videos
        "scene_prompts": [
            "living room", "kitchen", "bedroom", "outdoor", "clinic", 
            "nursery", "office", "playground", "baby", "child", "toy"
        ],
        "enabled": True
    }
    
    person_config = {
        "model": "yolo11n-pose.pt",  # YOLO11 pose model
        "conf_threshold": 0.4,
        "iou_threshold": 0.7,
        "track_mode": True,
        "tracker": "bytetrack",
        "pose_format": "coco_17",
        "min_keypoint_confidence": 0.3,
        "enabled": True
    }
    
    try:
        # Initialize pipelines
        scene_pipeline = SceneDetectionPipeline(scene_config)
        logger.info("Scene detection pipeline initialized")
        
        person_pipeline = PersonTrackingPipeline(person_config)
        logger.info("Person tracking pipeline initialized")
        
        results_summary = []
        
        # Process each video
        for video_path in video_files:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing: {video_path.name}")
            logger.info(f"{'='*50}")
            
            video_results = {
                "video_path": str(video_path),
                "video_name": video_path.name,
                "processing_time": datetime.now().isoformat(),
                "pipelines": {},
                "errors": []
            }
            
            # Process with scene detection
            try:
                logger.info("Running scene detection...")
                scene_results = scene_pipeline.process(
                    video_path=str(video_path),
                    start_time=0.0,
                    end_time=None,  # Process full video
                    pps=1.0,
                    output_dir=str(output_dir)
                )
                
                logger.info(f"Found {len(scene_results)} scene segments")
                
                # Convert scene results to JSON-serializable format
                scene_data = []
                for i, result in enumerate(scene_results):
                    if hasattr(result, 'to_dict'):
                        scene_data.append(result.to_dict())
                    else:
                        scene_data.append({
                            "scene_id": f"scene_{i:03d}",
                            "start_time": getattr(result, 'start_time', 0.0),
                            "end_time": getattr(result, 'end_time', 0.0),
                            "scene_type": getattr(result, 'scene_type', 'unknown'),
                            "confidence": getattr(result, 'confidence', None)
                        })
                
                video_results["pipelines"]["scene_detection"] = {
                    "status": "success",
                    "num_scenes": len(scene_results),
                    "config": scene_config,
                    "results": scene_data
                }
                
            except Exception as e:
                logger.error(f"Scene detection failed: {e}")
                video_results["pipelines"]["scene_detection"] = {
                    "status": "error",
                    "error": str(e),
                    "config": scene_config
                }
                video_results["errors"].append(f"Scene detection: {str(e)}")
            
            # Process with person tracking
            try:
                logger.info("Running person tracking...")
                person_results = person_pipeline.process(
                    video_path=str(video_path),
                    start_time=0.0,
                    end_time=None,  # Process full video
                    pps=5.0,  # 5 predictions per second for tracking
                    output_dir=str(output_dir)
                )
                
                logger.info(f"Found {len(person_results)} person detections/trajectories")
                
                # Convert person results to JSON-serializable format
                person_data = []
                for i, result in enumerate(person_results):
                    if hasattr(result, 'to_dict'):
                        person_data.append(result.to_dict())
                    elif hasattr(result, '__dict__'):
                        # For Pydantic models, use model_dump
                        if hasattr(result, 'model_dump'):
                            person_data.append(result.model_dump())
                        else:
                            person_data.append(result.dict())
                    else:
                        person_data.append({
                            "person_id": getattr(result, 'person_id', i),
                            "timestamp": getattr(result, 'timestamp', 0.0),
                            "confidence": getattr(result, 'confidence', None)
                        })
                
                video_results["pipelines"]["person_tracking"] = {
                    "status": "success",
                    "num_detections": len(person_results),
                    "config": person_config,
                    "results": person_data
                }
                
            except Exception as e:
                logger.error(f"Person tracking failed: {e}")
                video_results["pipelines"]["person_tracking"] = {
                    "status": "error",
                    "error": str(e),
                    "config": person_config
                }
                video_results["errors"].append(f"Person tracking: {str(e)}")
            
            # Save individual video results
            output_file = output_dir / f"{video_path.stem}_combined_results.json"
            with open(output_file, 'w') as f:
                json.dump(video_results, f, indent=2)
            
            logger.info(f"Results saved to: {output_file}")
            results_summary.append(video_results)
        
        # Save comprehensive summary
        summary = {
            "metadata": create_annotation_metadata(
                pipeline_name="DemoRunner",
                processing_params={
                    "scene_config": scene_config,
                    "person_config": person_config
                }
            ),
            "processing_summary": {
                "total_videos": len(video_files),
                "successful": len([r for r in results_summary if not r.get("errors")]),
                "failed": len([r for r in results_summary if r.get("errors")]),
                "total_scenes": sum(
                    r.get("pipelines", {}).get("scene_detection", {}).get("num_scenes", 0) 
                    for r in results_summary
                ),
                "total_person_detections": sum(
                    r.get("pipelines", {}).get("person_tracking", {}).get("num_detections", 0) 
                    for r in results_summary
                ),
                "processing_time": datetime.now().isoformat(),
                "scene_config": scene_config,
                "person_config": person_config
            },
            "video_results": results_summary
        }
        
        summary_file = output_dir / "combined_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*70}")
        logger.info("Processing Summary:")
        logger.info(f"  Total videos: {summary['processing_summary']['total_videos']}")
        logger.info(f"  Successful: {summary['processing_summary']['successful']}")
        logger.info(f"  Failed: {summary['processing_summary']['failed']}")
        logger.info(f"  Total scenes detected: {summary['processing_summary']['total_scenes']}")
        logger.info(f"  Total person detections: {summary['processing_summary']['total_person_detections']}")
        logger.info(f"  Summary saved to: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        return False


if __name__ == "__main__":
    success = process_demo_videos()
    sys.exit(0 if success else 1)
