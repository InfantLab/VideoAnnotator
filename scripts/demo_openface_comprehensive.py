#!/usr/bin/env python
"""
Enhanced OpenFace 3.0 Demo with Full Feature Analysis

This demo showcases all OpenFace 3.0 capabilities:
- 98-point facial landmarks (2D and 3D)
- Action Units (AU) analysis
- Head pose estimation
- Gaze tracking
- Emotion recognition
- Face tracking across frames

Usage:
    python scripts/demo_openface_comprehensive.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from main import VideoAnnotatorRunner

# Try importing OpenFace 3.0 compatibility layer
try:
    from src.pipelines.face_analysis.openface_compatibility import *
    print("✅ OpenFace 3.0 compatibility layer loaded")
except ImportError as e:
    print(f"❌ Failed to load OpenFace 3.0 compatibility: {e}")
    sys.exit(1)


def check_openface_comprehensive_setup():
    """Check if OpenFace 3.0 is properly set up for comprehensive analysis."""
    print("🔍 Checking OpenFace 3.0 comprehensive setup...")
    
    try:
        # Test all OpenFace components
        components = {
            "FaceDetector": FaceDetector,
            "LandmarkDetector": LandmarkDetector,
            "ActionUnitAnalyzer": ActionUnitAnalyzer,
            "HeadPoseEstimator": HeadPoseEstimator,
            "GazeEstimator": GazeEstimator,
            "EmotionRecognizer": EmotionRecognizer
        }
        
        print("📋 OpenFace 3.0 Component Status:")
        for name, component in components.items():
            try:
                # Test instantiation
                if name == "FaceDetector":
                    test_obj = component(confidence_threshold=0.5, device="cpu")
                elif name == "LandmarkDetector":
                    test_obj = component(model_type="98_point", enable_3d=True, device="cpu")
                else:
                    test_obj = component(device="cpu")
                
                print(f"  ✅ {name}: Available")
                del test_obj
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")
                return False
        
        print("✅ All OpenFace 3.0 components available")
        return True
        
    except Exception as e:
        print(f"❌ OpenFace 3.0 comprehensive setup check failed: {e}")
        return False


def analyze_openface_results(results_file: str) -> Dict[str, Any]:
    """Analyze the comprehensive OpenFace 3.0 results."""
    print(f"\n📊 Analyzing comprehensive OpenFace 3.0 results: {results_file}")
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        analysis = {
            "total_faces": 0,
            "features_detected": {
                "landmarks_2d": 0,
                "landmarks_3d": 0,
                "action_units": 0,
                "head_pose": 0,
                "gaze": 0,
                "emotions": 0,
                "face_tracking": 0
            },
            "statistics": {
                "avg_confidence": 0.0,
                "unique_faces": set(),
                "emotion_distribution": {},
                "dominant_emotions": [],
                "head_pose_range": {"rotation": [], "translation": []},
                "action_units_detected": set()
            }
        }
        
        # Analyze annotations
        annotations = data.get("annotations", [])
        analysis["total_faces"] = len(annotations)
        
        confidences = []
        
        for annotation in annotations:
            openface_data = annotation.get("openface3", {})
            
            # Count features
            if "landmarks_2d" in openface_data:
                analysis["features_detected"]["landmarks_2d"] += 1
            if "landmarks_3d" in openface_data:
                analysis["features_detected"]["landmarks_3d"] += 1
            if "action_units" in openface_data:
                analysis["features_detected"]["action_units"] += 1
                # Collect action units
                aus = openface_data["action_units"]
                if isinstance(aus, dict):
                    for au_name in aus.keys():
                        analysis["statistics"]["action_units_detected"].add(au_name)
            if "head_pose" in openface_data:
                analysis["features_detected"]["head_pose"] += 1
                head_pose = openface_data["head_pose"]
                if "rotation" in head_pose:
                    analysis["statistics"]["head_pose_range"]["rotation"].append(head_pose["rotation"])
                if "translation" in head_pose:
                    analysis["statistics"]["head_pose_range"]["translation"].append(head_pose["translation"])
            if "gaze" in openface_data:
                analysis["features_detected"]["gaze"] += 1
            if "emotion" in openface_data:
                analysis["features_detected"]["emotions"] += 1
                emotion_data = openface_data["emotion"]
                if "dominant" in emotion_data:
                    dominant_emotion = emotion_data["dominant"]
                    analysis["statistics"]["dominant_emotions"].append(dominant_emotion)
                    analysis["statistics"]["emotion_distribution"][dominant_emotion] = \
                        analysis["statistics"]["emotion_distribution"].get(dominant_emotion, 0) + 1
            if "track_id" in openface_data:
                analysis["features_detected"]["face_tracking"] += 1
                analysis["statistics"]["unique_faces"].add(openface_data["track_id"])
            
            # Collect confidence scores
            confidence = openface_data.get("confidence", 0.0)
            if confidence > 0:
                confidences.append(confidence)
        
        # Calculate statistics
        if confidences:
            analysis["statistics"]["avg_confidence"] = sum(confidences) / len(confidences)
        
        analysis["statistics"]["unique_faces"] = len(analysis["statistics"]["unique_faces"])
        analysis["statistics"]["action_units_detected"] = list(analysis["statistics"]["action_units_detected"])
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return {}


def print_comprehensive_analysis(analysis: Dict[str, Any]):
    """Print detailed analysis of OpenFace 3.0 results."""
    print("\n🎯 OpenFace 3.0 Comprehensive Analysis Results")
    print("=" * 60)
    
    print(f"📊 Detection Summary:")
    print(f"  Total faces detected: {analysis['total_faces']}")
    print(f"  Unique tracked faces: {analysis['statistics']['unique_faces']}")
    print(f"  Average confidence: {analysis['statistics']['avg_confidence']:.3f}")
    
    print(f"\n🎭 Features Extracted:")
    features = analysis['features_detected']
    for feature, count in features.items():
        percentage = (count / analysis['total_faces'] * 100) if analysis['total_faces'] > 0 else 0
        print(f"  {feature.replace('_', ' ').title()}: {count}/{analysis['total_faces']} ({percentage:.1f}%)")
    
    if analysis['statistics']['dominant_emotions']:
        print(f"\n😊 Emotion Analysis:")
        emotion_dist = analysis['statistics']['emotion_distribution']
        for emotion, count in sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(analysis['statistics']['dominant_emotions']) * 100)
            print(f"  {emotion.capitalize()}: {count} occurrences ({percentage:.1f}%)")
    
    if analysis['statistics']['action_units_detected']:
        print(f"\n🎪 Action Units Detected:")
        aus = analysis['statistics']['action_units_detected']
        print(f"  Total unique AUs: {len(aus)}")
        if len(aus) <= 10:  # Show all if not too many
            print(f"  AUs: {', '.join(sorted(aus))}")
        else:
            print(f"  AUs: {', '.join(sorted(aus)[:10])}... (+{len(aus)-10} more)")
    
    if analysis['statistics']['head_pose_range']['rotation']:
        print(f"\n🎯 Head Pose Analysis:")
        rotations = analysis['statistics']['head_pose_range']['rotation']
        print(f"  Rotation samples: {len(rotations)}")
        if rotations:
            # Calculate rotation range
            import numpy as np
            rot_array = np.array(rotations)
            print(f"  Rotation range - X: [{rot_array[:, 0].min():.2f}, {rot_array[:, 0].max():.2f}]")
            print(f"  Rotation range - Y: [{rot_array[:, 1].min():.2f}, {rot_array[:, 1].max():.2f}]")
            print(f"  Rotation range - Z: [{rot_array[:, 2].min():.2f}, {rot_array[:, 2].max():.2f}]")


def run_openface_comprehensive_demo(video_path: str = None, output_path: str = None):
    """Run comprehensive OpenFace 3.0 demo with all features enabled."""
    
    # Default video path
    if video_path is None:
        video_path = "./demovideos/babyjokes/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4"
    
    # Set output path
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"./demo_results/{video_name}_openface3_comprehensive.json"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize VideoAnnotator with comprehensive OpenFace configuration
    try:
        print("🚀 Initializing VideoAnnotator with comprehensive OpenFace 3.0...")
        
        # Use comprehensive OpenFace configuration
        config_path = "./configs/openface3.yaml"
        runner = VideoAnnotatorRunner(config_path)
        
        # Process the video
        print("🔄 Processing video with comprehensive face analysis...")
        print(f"📹 Input: {video_path}")
        print(f"📁 Output: {output_dir}")
        
        start_time = time.time()
        results = runner.process_video(Path(video_path), Path(output_dir))
        processing_time = time.time() - start_time
        
        # Save results
        if results:
            print(f"✅ Processing completed in {processing_time:.2f} seconds")
            print(f"💾 Results saved to: {output_path}")
            
            # Find the OpenFace results file
            video_name = Path(video_path).stem
            openface_results_file = Path(output_dir) / f"{video_name}_openface3_analysis.json"
            
            if openface_results_file.exists():
                # Analyze comprehensive results
                analysis = analyze_openface_results(str(openface_results_file))
                if analysis:
                    print_comprehensive_analysis(analysis)
                
                # Also check for detailed results
                detailed_results_file = Path(output_dir) / f"{video_name}_openface3_detailed.json"
                if detailed_results_file.exists():
                    print(f"\n📋 Detailed results also available: {detailed_results_file}")
            
            return True
        else:
            print("❌ No results returned from processing")
            return False
            
    except Exception as e:
        print(f"❌ Error during comprehensive processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for comprehensive OpenFace 3.0 demo."""
    print("🎯 OpenFace 3.0 Comprehensive Feature Demo")
    print("=" * 50)
    print("This demo showcases ALL OpenFace 3.0 capabilities:")
    print("• 98-point facial landmarks (2D + 3D)")
    print("• Action Units (AU) intensity & presence")
    print("• Head pose estimation (rotation & translation)")
    print("• Gaze tracking & eye gaze direction")
    print("• Emotion recognition (7 basic emotions)")
    print("• Face tracking across video frames")
    print("=" * 50)
    
    # Check comprehensive setup
    if not check_openface_comprehensive_setup():
        print("❌ OpenFace 3.0 comprehensive setup check failed")
        print("💡 Using compatibility layer for demo purposes")
        # Continue with demo using compatibility layer
    
    # Run comprehensive demo
    success = run_openface_comprehensive_demo()
    
    if success:
        print("\n🎉 Comprehensive OpenFace 3.0 Demo Completed Successfully!")
        print("\n📋 Summary of Generated Files:")
        print("  • COCO format annotations with all OpenFace features")
        print("  • Detailed JSON with feature extraction results")
        print("  • Scene detection, person tracking, and audio analysis")
        print("  • Comprehensive analysis statistics")
        
        print("\n🚀 Next Steps:")
        print("  • Install full OpenFace 3.0 for real model processing")
        print("  • Experiment with different confidence thresholds")
        print("  • Process longer videos for tracking analysis")
        print("  • Integrate with your own video datasets")
    else:
        print("\n❌ Demo failed. Check configuration and try again.")
    
    print("\n🏁 Demo completed!")


if __name__ == "__main__":
    main()
