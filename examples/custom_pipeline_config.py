"""Example: Custom Pipeline Configuration.

This example shows how to create and use custom pipeline configurations
for specific use cases or experiments.

Usage:
    python examples/custom_pipeline_config.py --video_path /path/to/video.mp4
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.pipelines.face_analysis import FacePipeline, FacePipelineConfig
from src.pipelines.person_tracking import PersonPipeline, PersonPipelineConfig

# Import the modernized pipelines
from src.pipelines.scene_detection import ScenePipeline, ScenePipelineConfig


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_high_performance_config() -> dict[str, Any]:
    """Create a high-performance configuration for powerful systems."""
    return {
        "scene_detection": {
            "threshold": 0.1,  # More sensitive scene detection
            "use_adaptive_threshold": True,
            "min_scene_length": 0.5,
            "clip_model": "ViT-B/32",
            "batch_size": 8,
            "skip_frames": 1,  # Process every frame
        },
        "person_tracking": {
            "model_name": "yolo11x",  # Largest, most accurate model
            "confidence_threshold": 0.3,
            "iou_threshold": 0.5,
            "max_det": 300,
            "track_buffer": 60,
            "match_thresh": 0.7,
            "frame_rate": 30,
            "enable_pose_estimation": True,
            "enable_tracking": True,
        },
        "face_analysis": {
            "backends": ["openface", "deepface"],
            "detection_confidence": 0.7,
            "recognition_model": "VGG-Face",
            "emotion_model": "fer2013",
            "enable_landmarks": True,
            "enable_gaze_estimation": True,
            "enable_action_units": True,
            "batch_size": 16,
        },
        "audio_processing": {
            "whisper_model": "large",  # Best accuracy
            "sample_rate": 16000,
            "chunk_duration": 10.0,
            "overlap_duration": 2.0,
            "emotion_window_size": 2.0,
        },
    }


def create_lightweight_config() -> dict[str, Any]:
    """Create a lightweight configuration for limited resources."""
    return {
        "scene_detection": {
            "threshold": 0.3,
            "use_adaptive_threshold": False,
            "min_scene_length": 2.0,
            "clip_model": "ViT-B/32",
            "batch_size": 2,
            "skip_frames": 5,  # Process every 5th frame
        },
        "person_tracking": {
            "model_name": "yolo11n",  # Smallest, fastest model
            "confidence_threshold": 0.5,
            "iou_threshold": 0.7,
            "max_det": 100,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "frame_rate": 10,
            "enable_pose_estimation": False,
            "enable_tracking": True,
        },
        "face_analysis": {
            "backends": ["deepface"],  # Reliable backend only
            "detection_confidence": 0.8,
            "recognition_model": "Facenet",
            "emotion_model": "fer2013",
            "enable_landmarks": False,
            "enable_gaze_estimation": False,
            "enable_action_units": False,
            "batch_size": 4,
        },
        "audio_processing": {
            "whisper_model": "tiny",  # Fastest model
            "sample_rate": 16000,
            "chunk_duration": 30.0,
            "overlap_duration": 0.5,
            "emotion_window_size": 5.0,
        },
    }


def create_research_config() -> dict[str, Any]:
    """Create a research-oriented configuration with all features enabled."""
    return {
        "scene_detection": {
            "threshold": 0.15,
            "use_adaptive_threshold": True,
            "min_scene_length": 1.0,
            "clip_model": "ViT-L/14",  # Largest CLIP model
            "batch_size": 4,
            "skip_frames": 2,
            "enable_scene_classification": True,
            "classification_classes": ["indoor", "outdoor", "face", "object", "text"],
        },
        "person_tracking": {
            "model_name": "yolo11x",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.4,
            "max_det": 500,
            "track_buffer": 120,
            "match_thresh": 0.6,
            "frame_rate": 30,
            "enable_pose_estimation": True,
            "enable_tracking": True,
            "pose_model": "yolo11x-pose",
        },
        "face_analysis": {
            "backends": ["openface", "deepface"],
            "detection_confidence": 0.5,
            "recognition_model": "ArcFace",
            "emotion_model": "fer2013",
            "enable_landmarks": True,
            "enable_gaze_estimation": True,
            "enable_action_units": True,
            "enable_head_pose": True,
            "batch_size": 8,
        },
        "audio_processing": {
            "whisper_model": "large",
            "sample_rate": 16000,
            "chunk_duration": 15.0,
            "overlap_duration": 3.0,
            "emotion_window_size": 2.0,
            "enable_speaker_diarization": True,
            "enable_music_detection": True,
            "enable_audio_events": True,
        },
    }


def demonstrate_custom_config(video_path: Path, config_type: str):
    """Demonstrate using a custom configuration."""
    logger = logging.getLogger(__name__)

    # Create configuration based on type
    if config_type == "high_performance":
        config = create_high_performance_config()
        logger.info("Using high-performance configuration")
    elif config_type == "lightweight":
        config = create_lightweight_config()
        logger.info("Using lightweight configuration")
    elif config_type == "research":
        config = create_research_config()
        logger.info("Using research configuration")
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")

    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Create output directory
    output_dir = Path(f"output_{config_type}")
    output_dir.mkdir(exist_ok=True)

    # Process with scene detection pipeline
    logger.info("Processing with Scene Detection Pipeline")
    scene_config = ScenePipelineConfig(**config["scene_detection"])
    scene_pipeline = ScenePipeline(scene_config)

    scene_results = scene_pipeline.process_video(video_path)

    # Save results
    with open(output_dir / "scene_results.json", "w") as f:
        json.dump(scene_results, f, indent=2, default=str)

    logger.info(f"Scene detection found {len(scene_results.get('scenes', []))} scenes")

    # Process with person tracking pipeline
    logger.info("Processing with Person Tracking Pipeline")
    person_config = PersonPipelineConfig(**config["person_tracking"])
    person_pipeline = PersonPipeline(person_config)

    person_results = person_pipeline.process_video(video_path)

    # Save results
    with open(output_dir / "person_results.json", "w") as f:
        json.dump(person_results, f, indent=2, default=str)

    logger.info(f"Person tracking found {len(person_results.get('tracks', []))} tracks")

    # Process with face analysis pipeline
    logger.info("Processing with Face Analysis Pipeline")
    face_config = FacePipelineConfig(**config["face_analysis"])
    face_pipeline = FacePipeline(face_config)

    face_results = face_pipeline.process_video(video_path)

    # Save results
    with open(output_dir / "face_results.json", "w") as f:
        json.dump(face_results, f, indent=2, default=str)

    logger.info(f"Face analysis found {len(face_results.get('faces', []))} faces")

    logger.info(f"All results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate custom pipeline configurations"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--config_type",
        type=str,
        default="high_performance",
        choices=["high_performance", "lightweight", "research"],
        help="Type of configuration to use",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate inputs
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Configuration type: {args.config_type}")

    # Demonstrate custom configuration
    try:
        demonstrate_custom_config(video_path, args.config_type)
        logger.info("Custom configuration demonstration completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Custom configuration demonstration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
