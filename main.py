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
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import yaml

# Import all pipeline modules - STANDARDS-ONLY VERSIONS
from src.pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline
from src.pipelines.face_analysis import (
    FaceAnalysisPipeline,
    LAIONFacePipeline,
    OpenFace3Pipeline,
    OPENFACE3_AVAILABLE
)
from src.pipelines.audio_processing import AudioPipeline as AudioProcessingPipeline
from src.utils.model_loader import setup_download_logging, log_first_run_info


# Global variables for interrupt handling
_interrupted = threading.Event()
_executor_shutdown = False


def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C)."""
    global _interrupted, _executor_shutdown
    logger = logging.getLogger(__name__)
    logger.warning("\nInterrupt received (Ctrl+C). Gracefully shutting down...")
    _interrupted.set()
    _executor_shutdown = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)


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
                'backend': 'laion',  # 'openface3', 'laion', 'opencv'
                'detection_confidence': 0.7,
                'enable_action_units': True,
                'enable_head_pose': True,
                'enable_gaze': True,
                'max_faces': 5
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
        """Initialize all pipeline instances and pre-load models for GPU efficiency."""
        try:
            # Scene detection pipeline
            if self.config.get('scene_detection', {}).get('enabled', True):
                scene_config = self.config.get('scene_detection', {})
                self.pipelines['scene'] = SceneDetectionPipeline(scene_config)
                self.pipelines['scene'].initialize()  # Pre-initialize for batch processing
                self.logger.info("Scene detection pipeline initialized")
            
            # Person tracking pipeline
            if self.config.get('person_tracking', {}).get('enabled', True):
                person_config = self.config.get('person_tracking', {})
                self.pipelines['person'] = PersonTrackingPipeline(person_config)
                self.pipelines['person'].initialize()  # Pre-initialize for batch processing
                self.logger.info("Person tracking pipeline initialized")
            
            # Face analysis pipeline
            if self.config.get('face_analysis', {}).get('enabled', True):
                face_config = self.config.get('face_analysis', {})
                backend = face_config.get('backend', 'laion')  # Default to LAION
                
                if backend == 'openface3':
                    try:
                        if OPENFACE3_AVAILABLE:
                            self.pipelines['face'] = OpenFace3Pipeline(face_config)
                            self.pipelines['face'].initialize()  # Pre-initialize for batch processing
                            self.logger.info("OpenFace 3.0 face analysis pipeline initialized")
                        else:
                            self.logger.error("OpenFace 3.0 not available, skipping face analysis")
                            self.logger.info("Install OpenFace 3.0 from: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0")
                            # Don't add any face pipeline - gracefully skip
                    except Exception as e:
                        self.logger.error(f"Failed to initialize OpenFace 3.0: {e}")
                        self.logger.info("Face analysis will be skipped for this session")
                        # Don't add any face pipeline - gracefully skip
                elif backend == 'laion':
                    self.pipelines['face'] = LAIONFacePipeline(face_config)
                    self.pipelines['face'].initialize()  # Pre-initialize for batch processing
                    self.logger.info("LAION face analysis pipeline initialized")
                elif backend == 'opencv':
                    self.pipelines['face'] = FaceAnalysisPipeline(face_config)
                    self.pipelines['face'].initialize()  # Pre-initialize for batch processing
                    self.logger.info("OpenCV face analysis pipeline initialized")
                else:
                    self.logger.warning(f"Unknown face backend '{backend}', using LAION")
                    self.pipelines['face'] = LAIONFacePipeline(face_config)
                    self.pipelines['face'].initialize()  # Pre-initialize for batch processing
                    self.logger.info("LAION face analysis pipeline initialized (default)")
            
            # Audio processing pipeline
            if self.config.get('audio_processing', {}).get('enabled', True):
                audio_config = self.config.get('audio_processing', {})
                self.pipelines['audio'] = AudioProcessingPipeline(audio_config)
                self.pipelines['audio'].initialize()  # Pre-initialize for batch processing
                self.logger.info("Audio processing pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing pipelines: {e}")
            raise
    
    def cleanup(self):
        """Cleanup all pipeline resources to free GPU memory."""
        for name, pipeline in self.pipelines.items():
            try:
                if hasattr(pipeline, 'cleanup'):
                    pipeline.cleanup()
                    self.logger.debug(f"Cleaned up {name} pipeline")
            except Exception as e:
                self.logger.warning(f"Error cleaning up {name} pipeline: {e}")
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.debug("Cleared CUDA cache")
        except ImportError:
            pass
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        self.cleanup()
    
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
            # Check for interrupt before each pipeline
            if _interrupted.is_set():
                self.logger.warning(f"Interrupted before running {pipeline_name} pipeline")
                results['pipeline_results'][pipeline_name] = {
                    'error': 'Processing interrupted by user',
                    'status': 'cancelled'
                }
                results['errors'].append(f"{pipeline_name} pipeline cancelled due to interrupt")
                break
                
            self.logger.info(f"Running {pipeline_name} pipeline...")
            pipeline_start_time = time.time()
            
            try:
                # Run the pipeline - all standards pipelines use unified process() method
                # Call process with or without pps depending on pipeline
                if pipeline_name == 'audio':
                    pipeline_results = pipeline.process(
                        video_path=str(video_path),
                        start_time=0.0,
                        end_time=None,
                        output_dir=str(output_dir)
                    )
                else:
                    pipeline_results = pipeline.process(
                        video_path=str(video_path),
                        start_time=0.0,
                        end_time=None,
                        pps=1.0,
                        output_dir=str(output_dir)
                    )
                
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
                
            except KeyboardInterrupt:
                _interrupted.set()
                error_msg = f"Interrupted during {pipeline_name} pipeline"
                self.logger.warning(error_msg)
                results['pipeline_results'][pipeline_name] = {
                    'error': 'Processing interrupted by user',
                    'status': 'cancelled'
                }
                results['errors'].append(error_msg)
                break
                
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
        
        # Process videos in parallel with interrupt handling
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_video = {}
            
            # Submit all tasks first
            for video_file in video_files:
                if _interrupted.is_set():
                    self.logger.warning("Interrupted before submitting all tasks")
                    break
                    
                # Create unique output directory for each video
                video_output_dir = output_dir / video_file.stem
                
                # Submit processing task
                future = executor.submit(
                    self._process_single_video_safe,
                    video_file, video_output_dir, selected_pipelines
                )
                future_to_video[future] = video_file
            
            # Collect results as they complete
            try:
                for future in as_completed(future_to_video):
                    if _interrupted.is_set():
                        self.logger.warning("Interrupted during processing, cancelling remaining tasks...")
                        # Cancel remaining futures
                        for f in future_to_video:
                            if not f.done():
                                f.cancel()
                        break
                    
                    video_file = future_to_video[future]
                    
                    try:
                        # Use timeout to allow interrupt checking
                        result = future.result(timeout=1.0)
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
                        
            except KeyboardInterrupt:
                self.logger.warning("KeyboardInterrupt caught in batch processing")
                _interrupted.set()
                # Attempt to cancel remaining futures
                for future in future_to_video:
                    if not future.done():
                        future.cancel()
        
        # Check if we were interrupted
        if _interrupted.is_set():
            batch_results['interrupted'] = True
            batch_results['errors'].append("Batch processing was interrupted by user")
        
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
            # Check for interrupt before starting
            if _interrupted.is_set():
                return {
                    'error': 'Processing interrupted by user',
                    'status': 'cancelled',
                    'video_path': str(video_path)
                }
            
            if not self.validate_video_file(video_path):
                return {
                    'error': 'Invalid video file',
                    'status': 'failed',
                    'video_path': str(video_path)
                }
            
            return self.process_video(video_path, output_dir, selected_pipelines)
            
        except KeyboardInterrupt:
            _interrupted.set()
            return {
                'error': 'Processing interrupted by user',
                'status': 'cancelled',
                'video_path': str(video_path)
            }
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
    
    def find_video_files(self, directory: Path) -> List[Path]:
        """
        Find all video files in the specified directory.
        
        Args:
            directory: Directory to search for video files
            
        Returns:
            List of video file paths
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(directory.glob(f'**/*{ext}'))
            video_files.extend(directory.glob(f'**/*{ext.upper()}'))
        
        # Remove duplicates and sort
        video_files = sorted(list(set(video_files)))
        
        return video_files

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
    """Set up comprehensive logging configuration."""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler - always present for user feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    
    handlers = [console_handler]
    
    # File handler - if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        file_handler.setFormatter(detailed_formatter)
        handlers.append(file_handler)
        
        print(f"Logging to file: {log_file}")
    else:
        # Create default log file in logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"videoannotator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        handlers.append(file_handler)
        
        print(f"Logging to file: {log_file}")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set root to DEBUG, handlers control actual level
        handlers=handlers,
        force=True  # Force reconfiguration
    )
    
    # Reduce noise from some verbose libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    # Suppress ByteTrack and YOLO debug output
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('yolo').setLevel(logging.WARNING)
    logging.getLogger('bytetrack').setLevel(logging.WARNING)
    # Suppress numba debug output
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numba.core.byteflow').setLevel(logging.WARNING)
    
    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.debug(f"Log level: {log_level}")
    logger.debug(f"Handlers: {[type(h).__name__ for h in handlers]}")
    
    return str(log_file)


def main():
    """Main entry point for the VideoAnnotator system."""
    # Set up enhanced model download logging
    setup_download_logging()
    
    # Show first-run information if models directory doesn't exist or is empty
    models_dir = Path("models")
    if not models_dir.exists() or not any(models_dir.iterdir()):
        log_first_run_info()
    
    parser = argparse.ArgumentParser(
        description='VideoAnnotator - Modern Video Annotation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video with default configuration
  python main.py --video_path video.mp4 --output_dir output/

  # Process a video with auto-generated output directory
  python main.py --video_path video.mp4

  # Use custom configuration
  python main.py --config configs/high_performance.yaml --video_path video.mp4

  # Run only specific pipelines
  python main.py --pipeline scene person --video_path video.mp4

  # Smart batch processing (directory detection with auto output)
  python main.py --video_path /path/to/videos

  # Smart batch processing with custom output
  python main.py --video_path /path/to/videos --output_dir /path/to/outputs

  # Get pipeline information
  python main.py --info

  # Explicit batch processing
  python main.py --batch_dir /path/to/videos --output_dir /path/to/outputs
        """
    )
    
    # Input/output arguments
    parser.add_argument('--video_path', type=str, help='Path to input video file or directory (directory will trigger batch processing)')
    parser.add_argument('--output_dir', type=str, help='Directory to save results (optional - will auto-generate if not specified)')
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
    
    # Set up logging and get log file path
    log_file_path = setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Show startup banner
    print("=" * 60)
    print("VideoAnnotator - Modern Video Annotation Pipeline")
    print("=" * 60)
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
        
        # Smart detection: check if video_path is actually a directory
        batch_mode = False
        input_path = None
        
        if args.video_path:
            input_path = Path(args.video_path)
            if input_path.is_dir():
                logger.info(f"Detected directory input, switching to batch processing mode")
                batch_mode = True
                batch_dir = input_path
            else:
                video_path = input_path
        elif args.batch_dir:
            batch_mode = True
            batch_dir = Path(args.batch_dir)
            input_path = batch_dir
        
        # Auto-generate output directory if not specified
        if not args.output_dir:
            if batch_mode:
                # Simple rule: input_directory -> input_directory_out
                # Examples: 
                # demovideos -> demovideos_out
                # demovideos/babyjokes -> demovideos/babyjokes_out
                # /full/path/to/videos -> /full/path/to/videos_out
                output_dir = input_path.parent / f"{input_path.name}_out"
            else:
                # For single video files: video.mp4 -> video_out/
                output_dir = input_path.parent / f"{input_path.stem}_out"
            
            logger.info(f"Auto-generated output directory: {output_dir}")
        else:
            output_dir = Path(args.output_dir)
        
        # Single video processing
        if not batch_mode:
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
        else:
            if not batch_dir.exists():
                logger.error(f"Batch directory not found: {batch_dir}")
                return 1
            
            # Find video files
            video_files = runner.find_video_files(batch_dir)
            
            if not video_files:
                logger.error(f"No video files found in {batch_dir}")
                return 1
            
            logger.info(f"Found {len(video_files)} videos to process")
            logger.info(f"Using {args.max_workers} parallel workers")
            logger.info(f"Output directory: {output_dir}")
            logger.info("Press Ctrl+C to interrupt batch processing gracefully")
            print()  # Add space before processing starts
            
            try:
                # Process videos in batch
                batch_results = runner.process_videos_batch(
                    batch_dir, output_dir, args.pipeline, max_workers=args.max_workers
                )
                
                # Print batch summary
                print("\n" + "=" * 60)
                print("BATCH PROCESSING SUMMARY")
                print("=" * 60)
                logger.info("Batch Processing Summary:")
                logger.info(f"  Total videos: {batch_results['total_videos']}")
                logger.info(f"  Completed: {batch_results['summary']['completed']}")
                logger.info(f"  Failed: {batch_results['summary']['failed']}")
                
                if batch_results.get('interrupted'):
                    logger.warning("  Status: INTERRUPTED BY USER")
                    
                logger.info(f"  Total processing time: {batch_results['summary']['total_processing_time']:.2f}s")
                logger.info(f"  Log file: {log_file_path}")
                
                if batch_results['errors']:
                    logger.warning(f"  Errors encountered: {len(batch_results['errors'])}")
                    for error in batch_results['errors'][:5]:  # Show first 5 errors
                        logger.warning(f"    - {error}")
                    if len(batch_results['errors']) > 5:
                        logger.warning(f"    ... and {len(batch_results['errors']) - 5} more errors")
                        logger.warning(f"    Check log file for complete error details: {log_file_path}")
                
                # Return appropriate exit code
                if batch_results.get('interrupted'):
                    return 130  # Standard exit code for Ctrl+C
                elif batch_results['summary']['failed'] > 0:
                    return 1
                else:
                    return 0
                    
            except KeyboardInterrupt:
                logger.warning("\nBatch processing interrupted by user")
                return 130
        
        logger.info("VideoAnnotator processing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nVideoAnnotator interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"VideoAnnotator processing failed: {e}")
        logger.error(f"Check log file for details: {log_file_path}")
        
        # Log full traceback to file but show simplified error to user
        logger.debug("Full error traceback:", exc_info=True)
        
        # Show traceback if debug level
        if args.log_level.upper() == 'DEBUG':
            import traceback
            traceback.print_exc()
        else:
            logger.error("Run with --log_level DEBUG for detailed error information")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
