#!/usr/bin/env python
"""
VideoAnnotator Main Pipeline Runner

This is the main entry point for the VideoAnnotator system. It provides a unified
interface for running the complete video annotation pipeline or individual components.

Usage:
    python main.py --video_path /path/to/video.mp4 --output_dir /path/to/output
    python main.py --config configs/high_performance.yaml --video_path video.mp4
    python main.py --pipeline scene --video_path video.mp4 --output_dir output/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

# Import all pipeline modules
from src.pipelines.scene_detection import ScenePipeline, ScenePipelineConfig
from src.pipelines.person_tracking import PersonPipeline, PersonPipelineConfig
from src.pipelines.face_analysis import FacePipeline, FacePipelineConfig
from src.pipelines.audio_processing import AudioPipeline, AudioPipelineConfig


class VideoAnnotatorRunner:
    """Main runner class for the VideoAnnotator pipeline system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the VideoAnnotator runner.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._get_default_config()
        
        # Initialize pipelines
        self.pipelines = {}
        self._initialize_pipelines()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'scene_detection': {
                'enabled': True,
                'threshold': 0.3,
                'min_scene_length': 1.0
            },
            'person_tracking': {
                'enabled': True,
                'model_name': 'yolo11s',
                'confidence_threshold': 0.5
            },
            'face_analysis': {
                'enabled': True,
                'backends': ['mediapipe'],
                'detection_confidence': 0.7
            },
            'audio_processing': {
                'enabled': True,
                'whisper_model': 'base',
                'sample_rate': 16000
            }
        }
    
    def _extract_audio_from_video(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Extract audio from video file using FFmpeg.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted audio
            
        Returns:
            Path to extracted audio file or None if extraction fails
        """
        try:
            # Check if FFmpeg is available
            if not shutil.which('ffmpeg'):
                self.logger.warning("FFmpeg not found. Audio extraction disabled.")
                return None
            
            # Define output audio path
            audio_path = output_dir / f"{video_path.stem}_audio.wav"
            
            # FFmpeg command to extract audio
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                str(audio_path)
            ]
            
            self.logger.info(f"Extracting audio from {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Audio extracted to {audio_path}")
                return audio_path
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            return None
    
    def _initialize_pipelines(self):
        """Initialize all pipeline instances."""
        try:
            # Scene detection pipeline
            if self.config.get('scene_detection', {}).get('enabled', True):
                scene_config = ScenePipelineConfig(**self.config.get('scene_detection', {}))
                self.pipelines['scene'] = ScenePipeline(scene_config)
                self.logger.info("Scene detection pipeline initialized")
            
            # Person tracking pipeline
            if self.config.get('person_tracking', {}).get('enabled', True):
                person_config = PersonPipelineConfig(**self.config.get('person_tracking', {}))
                self.pipelines['person'] = PersonPipeline(person_config)
                self.logger.info("Person tracking pipeline initialized")
            
            # Face analysis pipeline
            if self.config.get('face_analysis', {}).get('enabled', True):
                face_config = FacePipelineConfig(**self.config.get('face_analysis', {}))
                self.pipelines['face'] = FacePipeline(face_config)
                self.logger.info("Face analysis pipeline initialized")
            
            # Audio processing pipeline
            if self.config.get('audio_processing', {}).get('enabled', True):
                audio_config = AudioPipelineConfig(**self.config.get('audio_processing', {}))
                self.pipelines['audio'] = AudioPipeline(audio_config)
                self.logger.info("Audio processing pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing pipelines: {e}")
            raise
    
    def process_video(self, video_path: Path, output_dir: Path, 
                     selected_pipelines: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a video through the annotation pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save results
            selected_pipelines: List of pipeline names to run (optional)
        
        Returns:
            Dictionary containing all processing results
        """
        self.logger.info(f"Starting video processing: {video_path}")
        start_time = time.time()
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            'video_path': str(video_path),
            'output_dir': str(output_dir),
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'pipeline_results': {},
            'errors': []
        }
        
        # Determine which pipelines to run
        if selected_pipelines:
            pipelines_to_run = {name: pipeline for name, pipeline in self.pipelines.items() 
                              if name in selected_pipelines}
        else:
            pipelines_to_run = self.pipelines
        
        # Extract audio if audio pipeline is enabled
        audio_path = None
        if 'audio' in pipelines_to_run:
            audio_path = self._extract_audio_from_video(video_path, output_dir)
            if not audio_path:
                self.logger.warning("Audio extraction failed. Skipping audio pipeline.")
                pipelines_to_run.pop('audio', None)
        
        # Run each pipeline
        for pipeline_name, pipeline in pipelines_to_run.items():
            self.logger.info(f"Running {pipeline_name} pipeline...")
            pipeline_start_time = time.time()
            
            try:
                # Run the pipeline
                if pipeline_name == 'audio':
                    # Use extracted audio
                    if audio_path and audio_path.exists():
                        pipeline_results = pipeline.process_audio(audio_path)
                    else:
                        pipeline_results = {
                            'error': 'Audio file not available',
                            'message': 'Audio extraction failed or disabled'
                        }
                else:
                    # For video pipelines
                    pipeline_results = pipeline.process_video(video_path)
                
                # Calculate processing time
                pipeline_duration = time.time() - pipeline_start_time
                
                # Store results
                results['pipeline_results'][pipeline_name] = {
                    'results': pipeline_results,
                    'processing_time': pipeline_duration,
                    'status': 'completed'
                }
                
                # Save individual pipeline results
                output_file = output_dir / f'{pipeline_name}_results.json'
                with open(output_file, 'w') as f:
                    json.dump(pipeline_results, f, indent=2, default=str)
                
                self.logger.info(f"Completed {pipeline_name} pipeline in {pipeline_duration:.2f}s")
                
            except Exception as e:
                error_msg = f"Error in {pipeline_name} pipeline: {str(e)}"
                self.logger.error(error_msg)
                
                results['pipeline_results'][pipeline_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
                results['errors'].append(error_msg)
        
        # Calculate total processing time
        total_duration = time.time() - start_time
        results['end_time'] = datetime.now().isoformat()
        results['total_duration'] = total_duration
        
        # Save complete results
        complete_results_file = output_dir / 'complete_results.json'
        with open(complete_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Video processing completed in {total_duration:.2f}s")
        return results
    
    def process_videos_batch(self, batch_dir: Path, output_dir: Path, 
                           selected_pipelines: Optional[List[str]] = None,
                           max_workers: int = 2) -> Dict[str, Any]:
        """
        Process multiple videos in batch mode.
        
        Args:
            batch_dir: Directory containing video files
            output_dir: Base output directory for all results
            selected_pipelines: List of pipeline names to run (optional)
            max_workers: Maximum number of parallel processes
            
        Returns:
            Dictionary containing batch processing results
        """
        self.logger.info(f"Starting batch processing from: {batch_dir}")
        
        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(batch_dir.glob(f'**/*{ext}'))
            video_files.extend(batch_dir.glob(f'**/*{ext.upper()}'))
        
        if not video_files:
            raise ValueError(f"No video files found in {batch_dir}")
        
        self.logger.info(f"Found {len(video_files)} video files to process")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize batch results
        batch_results = {
            'batch_dir': str(batch_dir),
            'output_dir': str(output_dir),
            'total_videos': len(video_files),
            'start_time': datetime.now().isoformat(),
            'video_results': {},
            'errors': [],
            'summary': {
                'completed': 0,
                'failed': 0,
                'total_processing_time': 0.0
            }
        }
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_video = {}
            
            for video_file in video_files:
                # Create unique output directory for each video
                video_output_dir = output_dir / video_file.stem
                
                # Submit processing task
                future = executor.submit(
                    self._process_single_video_safe,
                    video_file, video_output_dir, selected_pipelines
                )
                future_to_video[future] = video_file
            
            # Collect results as they complete
            for future in as_completed(future_to_video):
                video_file = future_to_video[future]
                
                try:
                    result = future.result()
                    batch_results['video_results'][str(video_file)] = result
                    
                    if result.get('errors'):
                        batch_results['summary']['failed'] += 1
                        batch_results['errors'].extend(result['errors'])
                    else:
                        batch_results['summary']['completed'] += 1
                    
                    batch_results['summary']['total_processing_time'] += result.get('total_duration', 0)
                    
                    self.logger.info(f"Completed processing: {video_file.name}")
                    
                except Exception as e:
                    error_msg = f"Failed to process {video_file}: {str(e)}"
                    self.logger.error(error_msg)
                    batch_results['errors'].append(error_msg)
                    batch_results['summary']['failed'] += 1
        
        # Finalize batch results
        batch_results['end_time'] = datetime.now().isoformat()
        
        # Save batch summary
        batch_summary_file = output_dir / 'batch_summary.json'
        with open(batch_summary_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        self.logger.info(f"Batch processing completed: {batch_results['summary']['completed']}/{batch_results['total_videos']} videos processed successfully")
        
        return batch_results
    
    def _process_single_video_safe(self, video_path: Path, output_dir: Path, 
                                  selected_pipelines: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Safely process a single video with error handling.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save results
            selected_pipelines: List of pipeline names to run
            
        Returns:
            Dictionary containing processing results
        """
        try:
            if not self.validate_video_file(video_path):
                return {
                    'error': 'Invalid video file',
                    'status': 'failed'
                }
            
            return self.process_video(video_path, output_dir, selected_pipelines)
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed',
                'video_path': str(video_path)
            }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about all initialized pipelines."""
        info = {
            'available_pipelines': list(self.pipelines.keys()),
            'pipeline_details': {}
        }
        
        for name, pipeline in self.pipelines.items():
            try:
                info['pipeline_details'][name] = pipeline.get_pipeline_info()
            except Exception as e:
                info['pipeline_details'][name] = {'error': str(e)}
        
        return info
    
    def validate_video_file(self, video_path: Path) -> bool:
        """Validate that the video file exists and is readable."""
        if not video_path.exists():
            self.logger.error(f"Video file not found: {video_path}")
            return False
        
        # Check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        if video_path.suffix.lower() not in valid_extensions:
            self.logger.warning(f"Unsupported video format: {video_path.suffix}")
        
        return True


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    """Main entry point for the VideoAnnotator system."""
    parser = argparse.ArgumentParser(
        description='VideoAnnotator - Modern Video Annotation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video with default configuration
  python main.py --video_path video.mp4 --output_dir output/

  # Use custom configuration
  python main.py --config configs/high_performance.yaml --video_path video.mp4

  # Run only specific pipelines
  python main.py --pipeline scene person --video_path video.mp4

  # Get pipeline information
  python main.py --info

  # Batch process videos
  python main.py --batch_dir /path/to/videos --output_dir /path/to/outputs
        """
    )
    
    # Input/output arguments
    parser.add_argument('--video_path', type=str, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    parser.add_argument('--batch_dir', type=str, help='Directory containing videos for batch processing')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--pipeline', nargs='+', choices=['scene', 'person', 'face', 'audio'],
                       help='Specific pipelines to run')
    
    # Batch processing arguments
    parser.add_argument('--max_workers', type=int, default=2, 
                       help='Maximum number of parallel workers for batch processing')
    parser.add_argument('--recursive', action='store_true',
                       help='Search for videos recursively in batch directory')
    
    # System arguments
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log_file', type=str, help='Log file path')
    parser.add_argument('--info', action='store_true', help='Show pipeline information and exit')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be processed without actually processing')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("VideoAnnotator - Modern Video Annotation Pipeline")
    logger.info("=" * 50)
    
    try:
        # Initialize the runner
        runner = VideoAnnotatorRunner(args.config)
        
        # Handle info request
        if args.info:
            info = runner.get_pipeline_info()
            print(json.dumps(info, indent=2))
            return 0
        
        # Validate required arguments
        if not args.video_path and not args.batch_dir:
            logger.error("Either --video_path or --batch_dir must be specified")
            return 1
        
        if not args.output_dir:
            logger.error("--output_dir must be specified")
            return 1
        
        # Single video processing
        if args.video_path:
            video_path = Path(args.video_path)
            output_dir = Path(args.output_dir)
            
            if not runner.validate_video_file(video_path):
                return 1
            
            results = runner.process_video(video_path, output_dir, args.pipeline)
            
            # Print summary
            logger.info("\nProcessing Summary:")
            logger.info(f"  Video: {video_path}")
            logger.info(f"  Output: {output_dir}")
            logger.info(f"  Duration: {results['total_duration']:.2f}s")
            logger.info(f"  Pipelines run: {len(results['pipeline_results'])}")
            logger.info(f"  Errors: {len(results['errors'])}")
            
            if results['errors']:
                logger.warning("Errors encountered:")
                for error in results['errors']:
                    logger.warning(f"  - {error}")
        
        # Batch processing
        elif args.batch_dir:
            batch_dir = Path(args.batch_dir)
            output_dir = Path(args.output_dir)
            
            if not batch_dir.exists():
                logger.error(f"Batch directory not found: {batch_dir}")
                return 1
            
            # Find video files
            video_files = runner.find_video_files(batch_dir)
            
            if not video_files:
                logger.error(f"No video files found in {batch_dir}")
                return 1
            
            logger.info(f"Found {len(video_files)} videos to process")
            
            # Process videos in batch
            batch_results = runner.process_videos_batch(
                batch_dir, output_dir, args.pipeline, max_workers=2
            )
            
            # Print batch summary
            logger.info("\nBatch Processing Summary:")
            logger.info(f"  Total videos: {batch_results['total_videos']}")
            logger.info(f"  Completed: {batch_results['summary']['completed']}")
            logger.info(f"  Failed: {batch_results['summary']['failed']}")
            logger.info(f"  Total processing time: {batch_results['summary']['total_processing_time']:.2f}s")
            
            if batch_results['errors']:
                logger.warning(f"  Errors encountered: {len(batch_results['errors'])}")
                for error in batch_results['errors'][:5]:  # Show first 5 errors
                    logger.warning(f"    - {error}")
                if len(batch_results['errors']) > 5:
                    logger.warning(f"    ... and {len(batch_results['errors']) - 5} more errors")
            
            return 0 if batch_results['summary']['failed'] == 0 else 1
        
        logger.info("VideoAnnotator processing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"VideoAnnotator processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
