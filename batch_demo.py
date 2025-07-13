#!/usr/bin/env python3
"""
VideoAnnotator Batch Processing Demo

This script demonstrates the new batch processing capabilities of VideoAnnotator,
including job queue management, parallel processing, failure recovery, and progress tracking.

Usage:
    python batch_demo.py                                    # Process demo videos
    python batch_demo.py --input /path/to/videos            # Process directory
    python batch_demo.py --video /path/to/video.mp4         # Process single video
    python batch_demo.py --workers 8                        # Use 8 parallel workers
    python batch_demo.py --resume checkpoint_123.json       # Resume from checkpoint
    python batch_demo.py --pipelines scene,person          # Run specific pipelines
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.batch import BatchOrchestrator, JobStatus
from src.storage import FileStorageBackend
from src.batch.recovery import RetryStrategy
from src.version import get_version_info, print_version_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_processing.log')
    ]
)
logger = logging.getLogger(__name__)


class BatchProcessingDemo:
    """Demonstrates VideoAnnotator batch processing capabilities."""
    
    def __init__(self, args):
        """Initialize demo with command line arguments."""
        self.args = args
        self.start_time = datetime.now()
        
        # Create storage backend
        self.storage_dir = Path(args.output)
        self.storage_backend = FileStorageBackend(self.storage_dir)
        
        # Create batch orchestrator
        self.orchestrator = BatchOrchestrator(
            storage_backend=self.storage_backend,
            max_retries=args.max_retries,
            retry_strategy=RetryStrategy(args.retry_strategy),
            checkpoint_interval=args.checkpoint_interval
        )
        
        logger.info(f"Initialized batch processing demo")
        logger.info(f"Storage directory: {self.storage_dir}")
        logger.info(f"Max workers: {args.workers}")
        logger.info(f"Max retries: {args.max_retries}")
    
    def run(self):
        """Run the batch processing demonstration."""
        try:
            if self.args.resume:
                return self._resume_batch()
            elif self.args.video:
                return self._process_single_video()
            elif self.args.input:
                return self._process_directory()
            else:
                return self._process_demo_videos()
                
        except KeyboardInterrupt:
            logger.info("Batch processing interrupted by user")
            self.orchestrator.stop_batch()
            return None
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def _resume_batch(self):
        """Resume batch processing from checkpoint."""
        checkpoint_file = self.args.resume
        if not Path(checkpoint_file).exists():
            logger.error(f"Checkpoint file not found: {checkpoint_file}")
            return None
        
        logger.info(f"Resuming batch from checkpoint: {checkpoint_file}")
        report = self.orchestrator.resume_batch(checkpoint_file)
        self._print_report(report)
        return report
    
    def _process_single_video(self):
        """Process a single video file."""
        video_path = Path(self.args.video)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        logger.info(f"Processing single video: {video_path}")
        
        # Add job
        job_id = self.orchestrator.add_job(
            video_path=video_path,
            config=self._get_pipeline_config(),
            selected_pipelines=self._get_selected_pipelines()
        )
        
        # Run batch
        report = self.orchestrator.run_batch(
            max_workers=1,  # Single video, single worker
            save_checkpoints=False
        )
        
        self._print_report(report)
        return report
    
    def _process_directory(self):
        """Process all videos in a directory."""
        input_dir = Path(self.args.input)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return None
        
        logger.info(f"Processing directory: {input_dir}")
        
        # Add jobs from directory
        job_ids = self.orchestrator.add_jobs_from_directory(
            input_dir=input_dir,
            config=self._get_pipeline_config(),
            selected_pipelines=self._get_selected_pipelines(),
            extensions=self.args.extensions
        )
        
        if not job_ids:
            logger.error(f"No video files found in {input_dir}")
            return None
        
        logger.info(f"Added {len(job_ids)} jobs")
        
        # Run batch
        report = self.orchestrator.run_batch(
            max_workers=self.args.workers,
            save_checkpoints=self.args.save_checkpoints
        )
        
        self._print_report(report)
        return report
    
    def _process_demo_videos(self):
        """Process demo videos from the project."""
        # Look for demo videos in common locations
        demo_locations = [
            Path("demovideos/babyjokes"),
            Path("demovideos/VEATIC"),
            Path("demovideos"),
            Path("data/demovideos"),
        ]
        
        demo_videos = []
        for location in demo_locations:
            if location.exists():
                demo_videos.extend(location.glob("*.mp4"))
                demo_videos.extend(location.glob("*.avi"))
                demo_videos.extend(location.glob("*.mov"))
        
        if not demo_videos:
            logger.error("No demo videos found. Use --input or --video to specify videos.")
            return None
        
        # Limit to first few videos for demo
        demo_videos = demo_videos[:self.args.max_demo_videos]
        
        logger.info(f"Processing {len(demo_videos)} demo videos")
        
        # Add jobs
        for video_path in demo_videos:
            self.orchestrator.add_job(
                video_path=video_path,
                config=self._get_pipeline_config(),
                selected_pipelines=self._get_selected_pipelines()
            )
        
        # Run batch
        report = self.orchestrator.run_batch(
            max_workers=self.args.workers,
            save_checkpoints=self.args.save_checkpoints
        )
        
        self._print_report(report)
        return report
    
    def _get_pipeline_config(self):
        """Get pipeline configuration based on quality setting."""
        if self.args.fast:
            return {
                "scene_detection": {
                    "threshold": 30.0,
                    "min_scene_length": 1.0,
                },
                "person_tracking": {
                    "model": "yolo11n.pt",
                    "conf_threshold": 0.5,
                },
                "face_analysis": {
                    "model": "yolo11n-face.pt",
                    "min_face_size": 30,
                },
                "audio_processing": {
                    "whisper_model": "tiny",
                    "diarization_enabled": False,
                }
            }
        elif self.args.high_quality:
            return {
                "scene_detection": {
                    "threshold": 20.0,
                    "min_scene_length": 0.5,
                },
                "person_tracking": {
                    "model": "yolo11l.pt",
                    "conf_threshold": 0.3,
                },
                "face_analysis": {
                    "model": "yolo11l-face.pt",
                    "min_face_size": 20,
                },
                "audio_processing": {
                    "whisper_model": "base",
                    "diarization_enabled": True,
                }
            }
        else:
            # Default/balanced settings
            return {
                "scene_detection": {
                    "threshold": 25.0,
                    "min_scene_length": 1.0,
                },
                "person_tracking": {
                    "model": "yolo11s.pt",
                    "conf_threshold": 0.4,
                },
                "face_analysis": {
                    "model": "yolo11s-face.pt",
                    "min_face_size": 25,
                },
                "audio_processing": {
                    "whisper_model": "base",
                    "diarization_enabled": True,
                }
            }
    
    def _get_selected_pipelines(self):
        """Get list of selected pipelines."""
        if self.args.pipelines:
            return [p.strip() for p in self.args.pipelines.split(',')]
        return None  # All pipelines
    
    def _print_report(self, report):
        """Print formatted batch processing report."""
        print("\n" + "="*80)
        print("BATCH PROCESSING REPORT")
        print("="*80)
        
        print(f"Batch ID: {report.batch_id}")
        print(f"Duration: {report.duration:.2f} seconds" if report.duration else "Duration: In progress")
        print(f"Total Jobs: {report.total_jobs}")
        print(f"Completed: {report.completed_jobs}")
        print(f"Failed: {report.failed_jobs}")
        print(f"Cancelled: {report.cancelled_jobs}")
        print(f"Success Rate: {report.success_rate:.1f}%")
        print(f"Total Processing Time: {report.total_processing_time:.2f} seconds")
        
        if report.errors:
            print(f"\nErrors ({len(report.errors)}):")
            for error in report.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(report.errors) > 5:
                print(f"  ... and {len(report.errors) - 5} more")
        
        # Show job details
        print(f"\nJob Details:")
        for job in report.jobs:
            status_icon = "✅" if job.status == JobStatus.COMPLETED else "❌" if job.status == JobStatus.FAILED else "⏸️"
            duration_str = f"{job.duration:.1f}s" if job.duration else "N/A"
            print(f"  {status_icon} {job.video_id}: {job.status.value} ({duration_str})")
            
            # Show pipeline results
            for pipeline_name, result in job.pipeline_results.items():
                pipeline_icon = "✅" if result.status == JobStatus.COMPLETED else "❌"
                annotation_count = f"({result.annotation_count} annotations)" if result.annotation_count else ""
                print(f"    {pipeline_icon} {pipeline_name}: {result.status.value} {annotation_count}")
        
        print("="*80)
        
        # Show storage stats
        storage_stats = self.storage_backend.get_stats()
        print(f"\nStorage Statistics:")
        print(f"Backend: {storage_stats['backend_type']}")
        print(f"Location: {storage_stats['base_dir']}")
        print(f"Total Size: {storage_stats['total_size_mb']:.1f} MB")
        print(f"Jobs by Status: {storage_stats['jobs_by_status']}")
        
        total_elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\nDemo completed in {total_elapsed:.2f} seconds")


def cleanup_orphaned_reports():
    """Clean up any batch reports left in the top-level directory."""
    current_dir = Path(".")
    report_patterns = ["batch_report_*.json", "recovery_report_*.json", "checkpoint_*.json"]
    
    cleaned_files = []
    for pattern in report_patterns:
        for file_path in current_dir.glob(pattern):
            try:
                file_path.unlink()
                cleaned_files.append(str(file_path))
            except Exception as e:
                logger.warning(f"Could not remove orphaned report {file_path}: {e}")
    
    if cleaned_files:
        logger.info(f"Cleaned up {len(cleaned_files)} orphaned report files")
    
    return cleaned_files


def main():
    """Main entry point."""
    # Clean up any orphaned reports from previous runs
    cleanup_orphaned_reports()
    
    parser = argparse.ArgumentParser(description='VideoAnnotator Batch Processing Demo')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video', type=str, help='Process single video file')
    input_group.add_argument('--input', type=str, help='Process directory of videos')
    input_group.add_argument('--resume', type=str, help='Resume from checkpoint file')
    
    # Processing options
    parser.add_argument('--output', type=str, default='batch_demo_results', 
                       help='Output directory for results')
    parser.add_argument('--workers', type=int, default=4, 
                       help='Number of parallel workers')
    parser.add_argument('--pipelines', type=str, 
                       help='Comma-separated list of pipelines (scene,person,face,audio)')
    
    # Quality presets
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument('--fast', action='store_true', 
                              help='Use fastest settings (lower quality)')
    quality_group.add_argument('--high-quality', action='store_true', 
                              help='Use highest quality settings (slower)')
    
    # Retry and recovery options
    parser.add_argument('--max-retries', type=int, default=3, 
                       help='Maximum retry attempts per job')
    parser.add_argument('--retry-strategy', type=str, default='exponential_backoff',
                       choices=['none', 'fixed_delay', 'exponential_backoff', 'linear_backoff'],
                       help='Retry strategy for failed jobs')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Save checkpoint every N completed jobs')
    parser.add_argument('--no-checkpoints', dest='save_checkpoints', action='store_false',
                       help='Disable checkpoint saving')
    
    # Demo options
    parser.add_argument('--max-demo-videos', type=int, default=3,
                       help='Maximum number of demo videos to process')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
                       help='Video file extensions to process')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--version', action='store_true',
                       help='Show version information')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Show version if requested
    if args.version:
        print_version_info()
        return
    
    # Create and run demo
    demo = BatchProcessingDemo(args)
    report = demo.run()
    
    # Exit with appropriate code
    if report and report.success_rate > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
