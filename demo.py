#!/usr/bin/env python3
"""
VideoAnnotator Pipeline Demo

A simplified demo script that showcases VideoAnnotator pipelines by calling
the main pipeline runner with appropriate configurations.

This script provides a user-friendly interface with sensible defaults while
leveraging the full power of the main VideoAnnotator pipeline system.

Usage:
    python demo.py                          # Run with defaults on demo video
    python demo.py --video path/to/video    # Use specific video
    python demo.py --pipelines scene,face   # Run specific pipelines
    python demo.py --output results/        # Custom output directory
    python demo.py --fast                   # Use fastest settings
    python demo.py --high-quality           # Use best quality settings
    python demo.py --batch_dir videos/      # Batch process directory
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for version info
sys.path.insert(0, str(Path(__file__).parent))
from src.version import print_version_info
from src.utils.model_loader import setup_download_logging, log_first_run_info


def find_demo_video() -> Optional[Path]:
    """Find a demo video to use for the demonstration."""
    demo_dirs = [
        Path("demovideos"),
        Path("examples"),
        Path("data"),
        Path(".")
    ]
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    for demo_dir in demo_dirs:
        if demo_dir.exists():
            for ext in video_extensions:
                videos = list(demo_dir.glob(f'**/*{ext}'))
                videos.extend(demo_dir.glob(f'**/*{ext.upper()}'))
                if videos:
                    return videos[0]  # Return first video found
    
    return None


def map_pipeline_names(demo_pipelines: str) -> List[str]:
    """Map demo pipeline names to main.py pipeline names."""
    pipeline_mapping = {
        'scene_detection': 'scene',
        'scene': 'scene',
        'person_tracking': 'person', 
        'person': 'person',
        'face_analysis': 'face',
        'laion_face_analysis': 'face',
        'openface3_analysis': 'face',
        'face': 'face',
        'audio_processing': 'audio',
        'laion_voice_analysis': 'audio',
        'audio': 'audio'
    }
    
    pipelines = [p.strip() for p in demo_pipelines.split(',')]
    mapped_pipelines = []
    
    for pipeline in pipelines:
        if pipeline in pipeline_mapping:
            mapped_name = pipeline_mapping[pipeline]
            if mapped_name not in mapped_pipelines:  # Avoid duplicates
                mapped_pipelines.append(mapped_name)
        else:
            print(f"Warning: Unknown pipeline '{pipeline}', skipping...")
    
    return mapped_pipelines


def run_main_pipeline(args) -> int:
    """Run the main VideoAnnotator pipeline with mapped arguments."""
    main_args = ['python', 'main.py']
    
    # Map video argument
    if args.video:
        main_args.extend(['--video_path', args.video])
    elif args.batch_dir:
        main_args.extend(['--batch_dir', args.batch_dir])
    else:
        # Find a demo video
        demo_video = find_demo_video()
        if demo_video:
            print(f"No video specified, using demo video: {demo_video}")
            main_args.extend(['--video_path', str(demo_video)])
        else:
            print("Error: No video specified and no demo videos found")
            print("Please specify a video with --video or --batch_dir")
            return 1
    
    # Map output directory
    if args.output and args.output != "demo_results":
        main_args.extend(['--output_dir', args.output])
    # If output is default "demo_results", let main.py auto-generate
    
    # Map pipelines
    if args.pipelines:
        mapped_pipelines = map_pipeline_names(args.pipelines)
        if mapped_pipelines:
            main_args.extend(['--pipeline'] + mapped_pipelines)
    
    # Map quality settings to configuration files
    if args.fast:
        config_path = Path('configs/lightweight.yaml')
        if config_path.exists():
            main_args.extend(['--config', str(config_path)])
        else:
            print("Warning: Fast config not found, using defaults")
    elif hasattr(args, 'high_quality') and args.high_quality:
        config_path = Path('configs/high_performance.yaml')
        if config_path.exists():
            main_args.extend(['--config', str(config_path)])
        else:
            print("Warning: High quality config not found, using defaults")
    
    # Always use INFO logging for demo
    main_args.extend(['--log_level', 'INFO'])
    
    print("VideoAnnotator Demo - Delegating to main pipeline...")
    print(f"Command: {' '.join(main_args)}")
    print("=" * 60)
    
    # Run the main pipeline
    try:
        result = subprocess.run(main_args, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running main pipeline: {e}")
        return 1


def main():
    """Main demo function."""
    # Set up enhanced model download logging
    setup_download_logging()
    
    # Show first-run information
    models_dir = Path("models")
    if not models_dir.exists() or not any(models_dir.iterdir()):
        log_first_run_info()
    
    parser = argparse.ArgumentParser(
        description='VideoAnnotator Pipeline Demo - Simplified Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with auto-detected video
  python demo.py

  # Use specific video
  python demo.py --video my_video.mp4

  # Run specific pipelines
  python demo.py --pipelines scene,face --video my_video.mp4

  # Fast processing
  python demo.py --fast --video my_video.mp4

  # High quality processing
  python demo.py --high-quality --video my_video.mp4

  # Batch processing
  python demo.py --batch_dir /path/to/videos

  # Custom output directory
  python demo.py --video my_video.mp4 --output my_results/

Note: This demo script calls the main VideoAnnotator pipeline with 
appropriate configurations. For advanced options, use main.py directly.
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        help="Path to video file or directory (directory will trigger batch processing)"
    )
    
    parser.add_argument(
        "--pipelines", "-p",
        help="Comma-separated list of pipelines to run (scene,person,face,audio)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="demo_results",
        help="Output directory for results (default: auto-generated)"
    )
    
    parser.add_argument(
        "--batch_dir", "-b",
        help="Directory containing videos for batch processing"
    )
    
    # Quality presets
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument(
        "--fast",
        action="store_true",
        help="Use fastest settings for quick demo"
    )
    quality_group.add_argument(
        "--high-quality",
        action="store_true", 
        help="Use best quality settings (slower)"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit"
    )
    
    args = parser.parse_args()
    
    if args.version:
        print_version_info()
        return 0
    
    # Run the main pipeline
    return run_main_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
