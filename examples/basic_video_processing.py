"""
Example: Basic Video Processing Pipeline

This example demonstrates how to use the modernized VideoAnnotator pipeline
to process a video file and extract comprehensive annotations.

Usage:
    python examples/basic_video_processing.py --video_path /path/to/video.mp4 --output_dir /path/to/output
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any

import yaml

# Import the modernized pipelines
from src.pipelines.scene_detection import ScenePipeline, ScenePipelineConfig
from src.pipelines.person_tracking import PersonPipeline, PersonPipelineConfig
from src.pipelines.face_analysis import FacePipeline, FacePipelineConfig
from src.pipelines.audio_processing import AudioPipeline, AudioPipelineConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('video_processing.log')
        ]
    )


def process_video(video_path: Path, output_dir: Path, config: Dict[str, Any]):
    """
    Process a video file through the complete annotation pipeline.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save output files
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'video_path': str(video_path),
        'output_dir': str(output_dir),
        'processing_results': {}
    }
    
    # 1. Scene Detection Pipeline
    logger.info("Starting scene detection...")
    try:
        scene_config = ScenePipelineConfig(**config.get('scene_detection', {}))
        scene_pipeline = ScenePipeline(scene_config)
        
        scene_results = scene_pipeline.process_video(video_path)
        results['processing_results']['scene_detection'] = scene_results
        
        # Save scene detection results
        with open(output_dir / 'scene_detection.json', 'w') as f:
            json.dump(scene_results, f, indent=2, default=str)
        
        logger.info(f"Scene detection completed. Found {len(scene_results.get('scenes', []))} scenes.")
        
    except Exception as e:
        logger.error(f"Scene detection failed: {e}")
        results['processing_results']['scene_detection'] = {'error': str(e)}
    
    # 2. Person Detection and Tracking Pipeline
    logger.info("Starting person detection and tracking...")
    try:
        person_config = PersonPipelineConfig(**config.get('person_tracking', {}))
        person_pipeline = PersonPipeline(person_config)
        
        person_results = person_pipeline.process_video(video_path)
        results['processing_results']['person_tracking'] = person_results
        
        # Save person tracking results
        with open(output_dir / 'person_tracking.json', 'w') as f:
            json.dump(person_results, f, indent=2, default=str)
        
        logger.info(f"Person tracking completed. Tracked {len(person_results.get('tracks', []))} persons.")
        
    except Exception as e:
        logger.error(f"Person tracking failed: {e}")
        results['processing_results']['person_tracking'] = {'error': str(e)}
    
    # 3. Face Analysis Pipeline
    logger.info("Starting face analysis...")
    try:
        face_config = FacePipelineConfig(**config.get('face_analysis', {}))
        face_pipeline = FacePipeline(face_config)
        
        face_results = face_pipeline.process_video(video_path)
        results['processing_results']['face_analysis'] = face_results
        
        # Save face analysis results
        with open(output_dir / 'face_analysis.json', 'w') as f:
            json.dump(face_results, f, indent=2, default=str)
        
        logger.info(f"Face analysis completed. Analyzed {len(face_results.get('faces', []))} faces.")
        
    except Exception as e:
        logger.error(f"Face analysis failed: {e}")
        results['processing_results']['face_analysis'] = {'error': str(e)}
    
    # 4. Audio Processing Pipeline
    logger.info("Starting audio processing...")
    try:
        audio_config = AudioPipelineConfig(**config.get('audio_processing', {}))
        audio_pipeline = AudioPipeline(audio_config)
        
        # Extract audio from video (you might want to use ffmpeg for this)
        audio_path = output_dir / 'extracted_audio.wav'
        # TODO: Add audio extraction logic here
        
        # For now, skip audio processing if audio file doesn't exist
        if audio_path.exists():
            audio_results = audio_pipeline.process_audio(audio_path)
            results['processing_results']['audio_processing'] = audio_results
            
            # Save audio processing results
            with open(output_dir / 'audio_processing.json', 'w') as f:
                json.dump(audio_results, f, indent=2, default=str)
            
            logger.info("Audio processing completed.")
        else:
            logger.warning("Audio file not found. Skipping audio processing.")
            results['processing_results']['audio_processing'] = {'error': 'Audio file not found'}
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        results['processing_results']['audio_processing'] = {'error': str(e)}
    
    # Save complete results
    with open(output_dir / 'complete_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Video processing completed. Results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Process video through VideoAnnotator pipeline')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to configuration file')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1
    
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config_path}")
    
    # Process the video
    try:
        results = process_video(video_path, output_dir, config)
        logger.info("Video processing completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
