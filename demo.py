#!/usr/bin/env python3
"""
VideoAnnotator Pipeline Demo

A single, comprehensive demo script that showcases all VideoAnnotator pipelines
with sensible defaults and simple configuration options.

Usage:
    python demo.py                          # Run with defaults
    python demo.py --video path/to/video    # Use specific video
    python demo.py --pipelines scene,face   # Run specific pipelines
    python demo.py --output results/        # Custom output directory
    python demo.py --fast                   # Use fastest settings
    python demo.py --high-quality           # Use best quality settings
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

from src.pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline  
from src.pipelines.face_analysis.face_pipeline import FaceAnalysisPipeline
from src.pipelines.audio_processing import AudioPipeline
from src.version import get_version_info, print_version_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoAnnotatorDemo:
    """Main demo class for VideoAnnotator pipelines."""
    
    def __init__(self, args):
        """Initialize demo with command line arguments."""
        self.args = args
        self.video_path = None
        self.output_dir = Path(args.output)
        self.results = {}
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Find demo video
        self._find_demo_video()
        
        # Configure pipelines based on quality setting
        self._setup_pipeline_configs()
    
    def _find_demo_video(self):
        """Find a suitable demo video."""
        if self.args.video:
            self.video_path = Path(self.args.video)
            if not self.video_path.exists():
                logger.error(f"Video file not found: {self.video_path}")
                sys.exit(1)
            return
        
        # Look for demo videos in common locations
        demo_locations = [
            Path("demovideos/babyjokes"),
            Path("demovideos/VEATIC"),
            Path("demovideos"),
            Path("data/demovideos"),
        ]
        
        for location in demo_locations:
            if location.exists():
                videos = list(location.glob("*.mp4"))
                if videos:
                    self.video_path = videos[0]
                    logger.info(f"Using demo video: {self.video_path}")
                    return
        
        logger.error("No demo video found. Use --video to specify a video file.")
        sys.exit(1)
    
    def _setup_pipeline_configs(self):
        """Setup pipeline configurations based on quality setting."""
        if self.args.fast:
            # Fastest settings for quick demos
            self.configs = {
                "scene_detection": {
                    "threshold": 30.0,
                    "min_scene_length": 1.0,
                },
                "person_tracking": {
                    "model": "yolo11n.pt",  # Nano model
                    "conf_threshold": 0.5,
                    "track_mode": True,
                },
                "face_analysis": {
                    "backend": "mediapipe",
                    "detection_confidence": 0.6,
                    "enable_emotion": True,
                    "enable_landmarks": False,
                },
                "audio_processing": {
                    "whisper_model": "tiny",
                    "word_timestamps": False,
                    "enable_diarization": False,
                }
            }
        elif self.args.high_quality:
            # Best quality settings
            self.configs = {
                "scene_detection": {
                    "threshold": 15.0,
                    "min_scene_length": 0.5,
                },
                "person_tracking": {
                    "model": "yolo11m-pose.pt",  # Medium model with pose
                    "conf_threshold": 0.3,
                    "track_mode": True,
                    "pose_format": "coco_17",
                },
                "face_analysis": {
                    "backend": "mediapipe", 
                    "detection_confidence": 0.3,
                    "enable_emotion": True,
                    "enable_landmarks": True,
                },
                "audio_processing": {
                    "whisper_model": "base",
                    "word_timestamps": True,
                    "enable_diarization": True,
                }
            }
        else:
            # Balanced default settings
            self.configs = {
                "scene_detection": {
                    "threshold": 20.0,
                    "min_scene_length": 1.0,
                },
                "person_tracking": {
                    "model": "yolo11n-pose.pt",  # Nano with pose
                    "conf_threshold": 0.4,
                    "track_mode": True,
                    "pose_format": "coco_17",
                },
                "face_analysis": {
                    "backend": "mediapipe",
                    "detection_confidence": 0.5,
                    "enable_emotion": True,
                    "enable_landmarks": True,
                },
                "audio_processing": {
                    "whisper_model": "tiny",
                    "word_timestamps": True,
                    "enable_diarization": False,  # Enable if HF token available
                }
            }
    
    def _get_enabled_pipelines(self):
        """Get list of pipelines to run based on user selection."""
        if self.args.pipelines:
            return [p.strip() for p in self.args.pipelines.split(',')]
        return ["scene_detection", "person_tracking", "face_analysis", "audio_processing"]
    
    def run_scene_detection(self):
        """Run scene detection pipeline."""
        logger.info("🎬 Running Scene Detection Pipeline...")
        try:
            pipeline = SceneDetectionPipeline(self.configs["scene_detection"])
            
            start_time = time.time()
            results = pipeline.process(str(self.video_path))
            duration = time.time() - start_time
            
            # Save results
            output_file = self.output_dir / f"scene_detection_{self.video_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            scene_count = len(results) if results else 0
            self.results["scene_detection"] = {
                "status": "success",
                "scenes_detected": scene_count,
                "processing_time": f"{duration:.2f}s",
                "output_file": str(output_file)
            }
            
            logger.info(f"   ✅ Detected {scene_count} scenes in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"   ❌ Scene detection failed: {e}")
            self.results["scene_detection"] = {"status": "failed", "error": str(e)}
    
    def run_person_tracking(self):
        """Run person tracking pipeline."""
        logger.info("👤 Running Person Tracking Pipeline...")
        try:
            pipeline = PersonTrackingPipeline(self.configs["person_tracking"])
            
            start_time = time.time()
            results = pipeline.process(str(self.video_path))
            duration = time.time() - start_time
            
            # Save results
            output_file = self.output_dir / f"person_tracking_{self.video_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            detection_count = len(results) if results else 0
            self.results["person_tracking"] = {
                "status": "success",
                "detections": detection_count,
                "processing_time": f"{duration:.2f}s",
                "output_file": str(output_file)
            }
            
            logger.info(f"   ✅ Found {detection_count} person detections in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"   ❌ Person tracking failed: {e}")
            self.results["person_tracking"] = {"status": "failed", "error": str(e)}
    
    def run_face_analysis(self):
        """Run face analysis pipeline."""
        logger.info("😊 Running Face Analysis Pipeline...")
        try:
            pipeline = FaceAnalysisPipeline(self.configs["face_analysis"])
            
            start_time = time.time()
            results = pipeline.process(str(self.video_path))
            duration = time.time() - start_time
            
            # Save results
            output_file = self.output_dir / f"face_analysis_{self.video_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            face_count = len(results) if results else 0
            self.results["face_analysis"] = {
                "status": "success",
                "faces_detected": face_count,
                "processing_time": f"{duration:.2f}s",
                "output_file": str(output_file)
            }
            
            logger.info(f"   ✅ Analyzed {face_count} faces in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"   ❌ Face analysis failed: {e}")
            self.results["face_analysis"] = {"status": "failed", "error": str(e)}
    
    def run_audio_processing(self):
        """Run audio processing pipeline."""
        logger.info("🎤 Running Audio Processing Pipeline...")
        try:
            pipeline = AudioPipeline(self.configs["audio_processing"])
            
            start_time = time.time()
            results = pipeline.process(str(self.video_path))
            duration = time.time() - start_time
            
            # Save results
            output_file = self.output_dir / f"audio_processing_{self.video_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            segment_count = len(results) if results else 0
            self.results["audio_processing"] = {
                "status": "success",
                "segments": segment_count,
                "processing_time": f"{duration:.2f}s",
                "output_file": str(output_file)
            }
            
            logger.info(f"   ✅ Processed {segment_count} audio segments in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"   ❌ Audio processing failed: {e}")
            self.results["audio_processing"] = {"status": "failed", "error": str(e)}
    
    def run_demo(self):
        """Run the complete demo."""
        logger.info("🚀 VideoAnnotator Pipeline Demo")
        logger.info("=" * 60)
        
        # Print version info
        print_version_info()
        print()
        
        # Show configuration
        logger.info(f"📁 Video: {self.video_path}")
        logger.info(f"📁 Output: {self.output_dir}")
        logger.info(f"⚙️  Quality: {'Fast' if self.args.fast else 'High Quality' if self.args.high_quality else 'Balanced'}")
        
        enabled_pipelines = self._get_enabled_pipelines()
        logger.info(f"🔧 Pipelines: {', '.join(enabled_pipelines)}")
        print()
        
        # Run selected pipelines
        pipeline_methods = {
            "scene_detection": self.run_scene_detection,
            "person_tracking": self.run_person_tracking,
            "face_analysis": self.run_face_analysis,
            "audio_processing": self.run_audio_processing,
        }
        
        total_start = time.time()
        
        for pipeline_name in enabled_pipelines:
            if pipeline_name in pipeline_methods:
                pipeline_methods[pipeline_name]()
            else:
                logger.warning(f"Unknown pipeline: {pipeline_name}")
        
        total_duration = time.time() - total_start
        
        # Print summary
        self._print_summary(total_duration)
        
        # Save summary
        self._save_summary(total_duration)
    
    def _print_summary(self, total_duration):
        """Print demo results summary."""
        print()
        logger.info("📊 Demo Results Summary")
        logger.info("=" * 60)
        
        successful = 0
        failed = 0
        
        for pipeline_name, result in self.results.items():
            status = "✅" if result["status"] == "success" else "❌"
            logger.info(f"{status} {pipeline_name.replace('_', ' ').title()}")
            
            if result["status"] == "success":
                successful += 1
                # Show key metrics
                if "scenes_detected" in result:
                    logger.info(f"    Scenes: {result['scenes_detected']}")
                elif "detections" in result:
                    logger.info(f"    Detections: {result['detections']}")
                elif "faces_detected" in result:
                    logger.info(f"    Faces: {result['faces_detected']}")
                elif "segments" in result:
                    logger.info(f"    Segments: {result['segments']}")
                
                logger.info(f"    Time: {result['processing_time']}")
                logger.info(f"    Output: {result['output_file']}")
            else:
                failed += 1
                logger.info(f"    Error: {result['error']}")
            print()
        
        logger.info(f"🎯 Overall: {successful} successful, {failed} failed")
        logger.info(f"⏱️  Total time: {total_duration:.2f}s")
        logger.info(f"📁 Results saved to: {self.output_dir}")
    
    def _save_summary(self, total_duration):
        """Save demo summary to JSON."""
        summary = {
            "demo_info": {
                "timestamp": datetime.now().isoformat(),
                "video_path": str(self.video_path),
                "output_directory": str(self.output_dir),
                "quality_setting": "fast" if self.args.fast else "high_quality" if self.args.high_quality else "balanced",
                "total_duration": f"{total_duration:.2f}s"
            },
            "videoannotator_version": get_version_info(),
            "pipeline_results": self.results
        }
        
        summary_file = self.output_dir / "demo_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Demo summary saved to: {summary_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="VideoAnnotator Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                              # Run all pipelines with default settings
  python demo.py --fast                       # Quick demo with fastest settings
  python demo.py --high-quality               # Best quality demo (slower)
  python demo.py --video my_video.mp4         # Use specific video
  python demo.py --pipelines scene,face       # Run only scene detection and face analysis
  python demo.py --output my_results/         # Save results to custom directory
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        help="Path to video file (default: auto-detect from demovideos/)"
    )
    
    parser.add_argument(
        "--pipelines", "-p",
        help="Comma-separated list of pipelines to run (scene_detection,person_tracking,face_analysis,audio_processing)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="demo_results",
        help="Output directory for results (default: demo_results)"
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
        return
    
    # Run the demo
    demo = VideoAnnotatorDemo(args)
    demo.run_demo()


if __name__ == "__main__":
    main()
