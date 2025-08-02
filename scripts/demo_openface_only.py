#!/usr/bin/env python3
"""
OpenFace 3.0 Face Analysis Demo Script

This script demonstrates ONLY OpenFace 3.0 face analysis capabilities,
including 
98-point landmarks, 
Action Units, 
Head Pose, 
Gaze, 
Emotions.
"""

import sys
import os
import logging
import cv2
import numpy as np
from pathlib import Path
import yaml
import json
from typing import Dict, Any, List

# Add VideoAnnotator root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_openface_setup():
    """Check if OpenFace 3.0 is properly set up."""
    print("🔍 Checking OpenFace 3.0 setup...")
    
    # Apply SciPy compatibility patch
    try:
        import scipy.integrate
        if not hasattr(scipy.integrate, 'simps'):
            print("📦 Applying SciPy compatibility patch for OpenFace 3.0...")
            scipy.integrate.simps = scipy.integrate.simpson
    except ImportError:
        pass
    
    # Check if we can import OpenFace 3.0 components
    try:
        from openface.face_detection import FaceDetector
        from openface.landmark_detection import LandmarkDetector
        from openface.multitask_model import MultitaskPredictor
        print("✅ OpenFace 3.0 components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ OpenFace 3.0 setup failed: {e}")
        print("   Please install OpenFace 3.0 from: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0")
        return False

def test_openface_face_analysis(video_path: str, max_frames: int = 5):
    """
    Test OpenFace 3.0 face analysis on video frames.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process
    """
    print(f"\\n🎬 Testing OpenFace face analysis on: {video_path}")
    
    # Import OpenFace components
    try:
        from src.pipelines.face_analysis.openface3_pipeline import OpenFace3Pipeline
    except ImportError as e:
        print(f"❌ Failed to import OpenFace pipeline: {e}")
        return None
    
    # Load configuration
    config = {
        "detection_confidence": 0.7,
        "landmark_points": 98,  # Use 98-point landmarks (we have this model)
        "enable_3d_landmarks": True,
        "enable_action_units": True,
        "enable_head_pose": True,
        "enable_gaze": True,
        "enable_emotions": True,
        "device": "cpu",
        "max_faces": 5
    }
    
    # Initialize pipeline
    try:
        pipeline = OpenFace3Pipeline(config)
        pipeline.initialize()
        print("✅ OpenFace pipeline initialized")
    except Exception as e:
        print(f"❌ Failed to initialize OpenFace pipeline: {e}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None
    
    results = {
        "video_path": video_path,
        "config": config,
        "frames": []
    }
    
    frame_count = 0
    processed_frames = 0
    
    print(f"🔄 Processing up to {max_frames} frames...")
    
    while cap.isOpened() and processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 30th frame (approximately 1 FPS for 30 FPS video)
        if frame_count % 30 == 0:
            timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
            
            print(f"   📷 Processing frame {frame_count} (t={timestamp:.2f}s)")
            
            try:
                # Process frame with OpenFace
                frame_results = pipeline._process_frame(frame, timestamp)
                
                if frame_results:
                    results["frames"].append({
                        "frame_number": frame_count,
                        "timestamp": timestamp,
                        "face_analysis": frame_results
                    })
                    
                    # Print summary for this frame
                    num_faces = len(frame_results)
                    print(f"      ✅ Detected {num_faces} face(s)")
                    
                    if num_faces > 0:
                        face = frame_results[0]  # First face
                        if "landmarks" in face:
                            print(f"         📍 Landmarks: {len(face['landmarks'])//2} points")
                        if "action_units" in face:
                            active_aus = [au for au, intensity in face["action_units"].items() 
                                        if intensity > 0.5]
                            print(f"         😊 Active AUs: {active_aus}")
                        if "emotion" in face:
                            emotion = face["emotion"]
                            if "dominant" in emotion:
                                print(f"         😍 Emotion: {emotion['dominant']}")
                
                processed_frames += 1
                
            except Exception as e:
                print(f"      ❌ Error processing frame {frame_count}: {e}")
        
        frame_count += 1
    
    cap.release()
    
    try:
        pipeline.cleanup()
    except:
        pass
    
    print(f"\\n📊 Processing Summary:")
    print(f"   - Total frames in video: {frame_count}")
    print(f"   - Frames processed: {processed_frames}")
    print(f"   - Frames with faces: {len(results['frames'])}")
    
    return results

def save_openface_results(results: Dict[str, Any], output_path: str):
    """Save OpenFace-only results to JSON."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 OpenFace results saved to: {output_path}")

def main():
    """Main demo function."""
    print("🎯 OpenFace 3.0 Face Analysis Demo")
    print("=" * 50)
    print("This demo tests ONLY OpenFace 3.0 face analysis capabilities:")
    print("- 98-point facial landmarks")
    print("- Action Units analysis") 
    print("- Head pose estimation")
    print("- Gaze tracking")
    print("- Emotion recognition")
    print()
    
    # Check OpenFace setup
    if not check_openface_setup():
        print("\\n❌ OpenFace 3.0 setup check failed.")
        print("\\nTo fix this:")
        print("1. Install OpenFace 3.0 from: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0")
        print("2. Or run in development mode with compatibility layer")
        return
    
    # Demo video path
    video_path = "./demovideos/babyjokes/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4"
    
    if not os.path.exists(video_path):
        print(f"⚠️ Demo video not found: {video_path}")
        video_path = input("Enter path to a video file: ").strip()
        if not video_path or not os.path.exists(video_path):
            print("❌ Invalid video path. Exiting.")
            return
    
    # Run OpenFace face analysis
    results = test_openface_face_analysis(video_path, max_frames=10)
    
    if results:
        # Save results
        video_name = Path(video_path).stem
        output_path = f"./demo_results/{video_name}_openface_only_analysis.json"
        save_openface_results(results, output_path)
        
        print("\\n🎉 OpenFace face analysis demo completed successfully!")
        print(f"📁 Results saved to: {output_path}")
    else:
        print("\\n❌ OpenFace face analysis demo failed.")

if __name__ == "__main__":
    main()
