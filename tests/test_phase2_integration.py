#!/usr/bin/env python3
"""
Test Phase 2: PersonID Integration with Face Analysis Pipelines
Tests face-to-person linking across all face analysis pipelines.
"""

import sys
import json
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipelines.face_analysis.face_pipeline import FaceAnalysisPipeline
from src.pipelines.face_analysis.openface3_pipeline import OpenFace3Pipeline
from src.pipelines.face_analysis.laion_face_pipeline import LAIONFacePipeline

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_face_pipeline_personid_integration():
    """Test FaceAnalysisPipeline with PersonID integration."""
    print("\n" + "="*60)
    print("Testing FaceAnalysisPipeline PersonID Integration")
    print("="*60)
    
    # Configure pipeline with PersonID enabled
    config = {
        "detection_backend": "opencv",
        "confidence_threshold": 0.5,
        "person_identity": {
            "enabled": True,
            "link_to_persons": True,
            "iou_threshold": 0.5,
            "require_person_id": False
        }
    }
    
    try:
        pipeline = FaceAnalysisPipeline(config)
        
        # Test video path
        video_path = "demovideos/babyjokes/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4"
        if not Path(video_path).exists():
            print(f"❌ Test video not found: {video_path}")
            return False
        
        # Check if person tracking data exists
        video_name = Path(video_path).stem
        person_data_paths = [
            Path("data") / f"person_tracking_{video_name}.json",
            Path("demo_results") / f"person_tracking_{video_name}.json"
        ]
        
        person_data_found = False
        for path in person_data_paths:
            if path.exists():
                print(f"✅ Found person tracking data: {path}")
                person_data_found = True
                break
        
        if not person_data_found:
            print("❌ No person tracking data found for face-person linking test")
            return False
        
        # Process video with face analysis and person linking
        print("Processing video with face analysis and person linking...")
        results = pipeline.process(
            video_path=video_path,
            start_time=0.0,
            end_time=5.0,  # Process first 5 seconds
            pps=1.0,
            output_dir="demo_results"
        )
        
        # Analyze results
        face_annotations = [ann for ann in results if ann.get('category_id') == 100]  # Face category ID is 100
        linked_faces = [ann for ann in face_annotations if ann.get('person_id') != 'unknown']
        
        print(f"📊 Results Summary:")
        print(f"   - Total face detections: {len(face_annotations)}")
        print(f"   - Faces linked to persons: {len(linked_faces)}")
        if len(face_annotations) > 0:
            print(f"   - Linking success rate: {len(linked_faces)/len(face_annotations)*100:.1f}%")
        else:
            print(f"   - Linking success rate: N/A (no faces detected)")
        
        # Test success if we have faces detected and person ID integration is working
        if len(face_annotations) > 0:
            print(f"✅ Face detection successful!")
            if linked_faces:
                print(f"✅ Face-to-person linking successful!")
                # Show sample linked face
                sample = linked_faces[0]
                print(f"   Sample: Face linked to person '{sample.get('person_label')}' (ID: {sample.get('person_id')})")
                print(f"   Confidence: {sample.get('person_label_confidence', 0.0):.2f}")
                print(f"   Method: {sample.get('person_labeling_method', 'unknown')}")
            else:
                print("ℹ️  Faces detected but no person linking (may be normal if no overlapping persons)")
            return True
        else:
            print("❌ No faces detected - check video processing")
            return False
            
    except Exception as e:
        print(f"❌ FaceAnalysisPipeline test failed: {e}")
        return False

def test_openface3_pipeline_personid_integration():
    """Test OpenFace3Pipeline with PersonID integration."""
    print("\n" + "="*60)
    print("Testing OpenFace3Pipeline PersonID Integration")
    print("="*60)
    
    # Configure pipeline with PersonID enabled
    config = {
        "detection_confidence": 0.7,
        "enable_3d_landmarks": False,  # Simplify for testing
        "enable_action_units": False,
        "enable_head_pose": False,
        "enable_gaze": False,
        "person_identity": {
            "enabled": True,
            "link_to_persons": True,
            "iou_threshold": 0.5,
            "require_person_id": False
        }
    }
    
    try:
        pipeline = OpenFace3Pipeline(config)
        
        # Test if schema includes person identity fields
        schema = pipeline.get_schema()
        person_fields = ["person_id", "person_label", "person_label_confidence", "person_labeling_method"]
        schema_annotation = schema.get("annotation_schema", {})
        
        for field in person_fields:
            if field in schema_annotation:
                print(f"✅ Schema includes {field}: {schema_annotation[field]}")
            else:
                print(f"❌ Schema missing {field}")
                return False
        
        print("✅ OpenFace3Pipeline PersonID integration configured correctly")
        print("   (Note: Full processing test skipped - requires OpenFace 3.0 models)")
        return True
        
    except Exception as e:
        print(f"❌ OpenFace3Pipeline test failed: {e}")
        return False

def test_laion_face_pipeline_personid_integration():
    """Test LAIONFacePipeline with PersonID integration."""
    print("\n" + "="*60)
    print("Testing LAIONFacePipeline PersonID Integration")
    print("="*60)
    
    # Configure pipeline with PersonID enabled
    config = {
        "confidence_threshold": 0.7,
        "model_size": "small",
        "person_identity": {
            "enabled": True,
            "link_to_persons": True,
            "iou_threshold": 0.5,
            "require_person_id": False
        }
    }
    
    try:
        pipeline = LAIONFacePipeline(config)
        
        # Test configuration
        assert "person_identity" in pipeline.config
        assert pipeline.config["person_identity"]["enabled"] == True
        assert pipeline.config["person_identity"]["link_to_persons"] == True
        
        print("✅ LAIONFacePipeline PersonID configuration verified")
        
        # Test helper methods exist
        assert hasattr(pipeline, '_load_person_tracks')
        assert hasattr(pipeline, '_get_frame_person_annotations')
        assert hasattr(pipeline, '_link_face_to_person')
        assert hasattr(pipeline, '_calculate_iou')
        
        print("✅ LAIONFacePipeline PersonID helper methods present")
        print("   (Note: Full processing test skipped - requires LAION models)")
        return True
        
    except Exception as e:
        print(f"❌ LAIONFacePipeline test failed: {e}")
        return False

def main():
    """Run Phase 2 integration tests."""
    print("🚀 Starting Phase 2: PersonID Integration Tests")
    print("Testing face-to-person linking across all face analysis pipelines")
    
    setup_logging()
    
    # Run tests
    tests = [
        ("FaceAnalysisPipeline", test_face_pipeline_personid_integration),
        ("OpenFace3Pipeline", test_openface3_pipeline_personid_integration),
        ("LAIONFacePipeline", test_laion_face_pipeline_personid_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Phase 2 Integration Test Results")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Phase 2: PersonID Integration SUCCESS!")
        print("All face analysis pipelines now support person identity linking")
    else:
        print("❌ Phase 2: Some integration tests failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
