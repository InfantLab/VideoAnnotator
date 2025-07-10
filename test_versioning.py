#!/usr/bin/env python
"""
Test script to verify versioning and metadata are included in all outputs.
"""

import json
import tempfile
from pathlib import Path
from src.pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline
from src.version import get_version_info, print_version_info

def test_versioning():
    """Test that versioning information is included in pipeline outputs."""
    
    print("=== VideoAnnotator Version Information ===")
    print_version_info()
    print("\n")
    
    # Test video path
    test_video = "babyjokes videos/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4"
    if not Path(test_video).exists():
        print(f"Test video not found: {test_video}")
        return
    
    # Test Scene Detection Pipeline
    print("=== Testing Scene Detection Pipeline ===")
    scene_pipeline = SceneDetectionPipeline()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            scene_pipeline.initialize()
            
            # Process a short segment
            annotations = scene_pipeline.process(
                video_path=test_video,
                start_time=0.0,
                end_time=5.0,
                output_dir=temp_dir
            )
            
            # Check output file
            output_files = list(Path(temp_dir).glob("*.json"))
            if output_files:
                with open(output_files[0], 'r') as f:
                    data = json.load(f)
                
                print(f"Scene detection output saved to: {output_files[0]}")
                print("Metadata structure:")
                print(f"  - VideoAnnotator version: {data['metadata']['videoannotator']['version']}")
                print(f"  - Pipeline: {data['metadata']['pipeline']['name']}")
                if 'model' in data['metadata']:
                    print(f"  - Model: {data['metadata']['model']['model_name']}")
                else:
                    print("  - Model: No model info")
                print(f"  - Processing timestamp: {data['metadata']['pipeline']['processing_timestamp']}")
                if data['metadata']['videoannotator']['git']:
                    print(f"  - Git commit: {data['metadata']['videoannotator']['git']['commit_hash'][:8]}")
                else:
                    print("  - Git commit: N/A")
                print(f"  - Python version: {data['metadata']['system']['python_version'].split()[0]}")
                print(f"  - Number of annotations: {len(data['annotations'])}")
                print()
                
        except Exception as e:
            print(f"Scene detection test failed: {e}")
        finally:
            scene_pipeline.cleanup()
    
    # Test Person Tracking Pipeline
    print("=== Testing Person Tracking Pipeline ===")
    person_pipeline = PersonTrackingPipeline()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            person_pipeline.initialize()
            
            # Process a short segment
            annotations = person_pipeline.process(
                video_path=test_video,
                start_time=0.0,
                end_time=5.0,
                pps=1.0  # Low rate for testing
            )
            
            # Manually save to check metadata
            if annotations:
                output_path = Path(temp_dir) / "person_detections.json"
                person_pipeline.save_annotations(annotations, str(output_path))
                output_files = [output_path]
            else:
                output_files = []
            if output_files:
                with open(output_files[0], 'r') as f:
                    data = json.load(f)
                
                print(f"Person tracking output saved to: {output_files[0]}")
                print("Metadata structure:")
                print(f"  - VideoAnnotator version: {data['metadata']['videoannotator']['version']}")
                print(f"  - Pipeline: {data['metadata']['pipeline']['name']}")
                if 'model' in data['metadata']:
                    print(f"  - Model: {data['metadata']['model']['model_name']}")
                else:
                    print("  - Model: No model info")
                print(f"  - Processing timestamp: {data['metadata']['pipeline']['processing_timestamp']}")
                if data['metadata']['videoannotator']['git']:
                    print(f"  - Git commit: {data['metadata']['videoannotator']['git']['commit_hash'][:8]}")
                else:
                    print("  - Git commit: N/A")
                print(f"  - Python version: {data['metadata']['system']['python_version'].split()[0]}")
                print(f"  - Number of detections: {len(data['detections'])}")
                print()
                
        except Exception as e:
            print(f"Person tracking test failed: {e}")
        finally:
            person_pipeline.cleanup()
    
    print("=== Versioning Test Complete ===")

if __name__ == "__main__":
    test_versioning()
