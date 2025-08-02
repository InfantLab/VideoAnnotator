#!/usr/bin/env python3
"""
OpenFace 3.0 Demo Script for VideoAnnotator

This script demonstrates how to use OpenFace 3.0 with the VideoAnnotator system
for facial analysis on video files.
"""

import sys
import os
import logging
from pathlib import Path
import yaml

# Add VideoAnnotator root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import VideoAnnotator components
try:
    from main import VideoAnnotatorRunner
    from src.pipelines.face_analysis.openface_compatibility import patch_scipy_compatibility
except ImportError as e:
    print(f"Error importing VideoAnnotator components: {e}")
    print("Make sure you're running from the VideoAnnotator root directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_openface_setup():
    """Check if OpenFace 3.0 is properly set up."""
    print("🔍 Checking OpenFace 3.0 setup...")
    
    # Apply compatibility patches
    patch_scipy_compatibility()
    
    # Check model files
    required_models = [
        "./weights/Alignment_RetinaFace.pth",
        "./weights/Landmark_98.pkl"
    ]
    
    missing_models = []
    for model_path in required_models:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        print("❌ Missing required model files:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nRun the model download command first:")
        print("python -c \"from src.pipelines.face_analysis.openface_compatibility import patch_scipy_compatibility; patch_scipy_compatibility(); from openface.cli import download; download()\"")
        return False
    
    # Test OpenFace imports
    try:
        from openface.face_detection import FaceDetector
        from openface.landmark_detection import LandmarkDetector
        print("✅ OpenFace imports successful")
        
        # Test model loading
        face_detector = FaceDetector(model_path=required_models[0])
        landmark_detector = LandmarkDetector(model_path=required_models[1])
        print("✅ OpenFace models loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenFace setup check failed: {e}")
        return False

def run_openface_demo(video_path, output_path=None):
    """
    Run OpenFace 3.0 face analysis on a video file.
    
    Args:
        video_path: Path to input video file
        output_path: Path for output results (optional)
    """
    print(f"\n🎬 Processing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    # Load OpenFace configuration
    config_path = "./configs/openface3.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    print("✅ Configuration loaded")
    
    # Set output path
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"./demo_results/{video_name}_openface3_analysis.json"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize VideoAnnotator with OpenFace configuration
    try:
        print("🚀 Initializing VideoAnnotator with OpenFace 3.0...")
        
        # Create runner with OpenFace configuration
        runner = VideoAnnotatorRunner(config_path)
        
        # Process the video
        print("🔄 Processing video...")
        results = runner.process_video(Path(video_path), Path(output_dir))
        
        # Save results
        if results:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to: {output_path}")
            
            # Print summary
            print("\n📊 Processing Summary:")
            if 'face_analysis' in results:
                face_results = results['face_analysis']
                print(f"   - Total frames processed: {len(face_results.get('frame_results', []))}")
                print(f"   - Faces detected: {face_results.get('summary', {}).get('total_faces_detected', 0)}")
                print(f"   - Average faces per frame: {face_results.get('summary', {}).get('avg_faces_per_frame', 0):.2f}")
            
            return True
        else:
            print("❌ No results returned from processing")
            return False
            
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    print("🎯 OpenFace 3.0 VideoAnnotator Demo")
    print("=" * 40)
    
    # Check OpenFace setup
    if not check_openface_setup():
        print("\n❌ OpenFace 3.0 setup check failed. Please fix the issues above and try again.")
        return
    
    print("✅ OpenFace 3.0 setup verified")
    
    # Demo video paths (you can modify these)
    demo_videos = [
        "./demovideos/babyjokes/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4"
    ]
    
    # Check for available demo videos
    available_videos = [v for v in demo_videos if os.path.exists(v)]
    
    if not available_videos:
        print("\n⚠️ No demo videos found. Available demo paths:")
        for video in demo_videos:
            print(f"   - {video}")
        print("\nTo run the demo, place a video file at one of these paths, or modify the script.")
        
        # Interactive mode
        video_path = input("\nEnter path to a video file (or press Enter to exit): ").strip()
        if video_path and os.path.exists(video_path):
            available_videos = [video_path]
        else:
            print("Exiting demo.")
            return
    
    # Process available videos
    for video_path in available_videos:
        success = run_openface_demo(video_path)
        if success:
            print(f"🎉 Successfully processed: {video_path}")
        else:
            print(f"❌ Failed to process: {video_path}")
    
    print("\n🏁 Demo completed!")

if __name__ == "__main__":
    main()
