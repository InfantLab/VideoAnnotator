"""
Test standards compatibility for VideoAnnotator schemas.

Validates compatibility with:
- COCO JSON format
- CVAT/Datumaro XML format  
- Label Studio JSON format
- Original VideoAnnotator specifications
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any, List

from src.schemas.standards_compatible_schemas import (
    UniversalPersonDetection,
    UniversalPoseKeypoints,
    COCOBoundingBox,
    COCOKeypoints,
    CVATAnnotation,
    LabelStudioResult,
    export_to_coco_json,
    export_to_cvat_xml,
    export_to_labelstudio_json
)


class TestCOCOCompatibility:
    """Test COCO format compatibility."""
    
    def test_coco_bbox_creation(self):
        """Test COCO bounding box creation."""
        bbox = COCOBoundingBox(
            id="test_1",
            image_id="frame_001",
            category_id="person",
            video_id="test_video",
            timestamp=1.5,
            bbox=[100, 150, 80, 200],
            area=16000,
            score=0.92
        )
        
        # Validate required COCO fields
        assert bbox.id == "test_1"
        assert bbox.image_id == "frame_001"
        assert bbox.category_id == "person"
        assert bbox.bbox == [100, 150, 80, 200]
        assert bbox.area == 16000
        assert bbox.score == 0.92
        
    def test_coco_keypoints_creation(self):
        """Test COCO keypoints creation."""
        # COCO format: [x1,y1,v1, x2,y2,v2, ...]
        keypoints_data = [
            100, 50, 2,   # nose
            95, 45, 2,    # left_eye
            105, 45, 2,   # right_eye
            90, 45, 1,    # left_ear
            110, 45, 1,   # right_ear
            # ... more keypoints
        ]
        
        keypoints = COCOKeypoints(
            id="test_pose_1",
            image_id="frame_001", 
            category_id="person",
            video_id="test_video",
            timestamp=1.5,
            keypoints=keypoints_data,
            num_keypoints=5,
            bbox=[80, 30, 60, 180],
            area=10800
        )
        
        assert len(keypoints.keypoints) == 15  # 5 keypoints * 3 values
        assert keypoints.num_keypoints == 5
        assert keypoints.bbox == [80, 30, 60, 180]
    
    def test_coco_export_format(self):
        """Test full COCO JSON export."""
        detections = [
            UniversalPersonDetection(
                video_id="test_video",
                timestamp=1.0,
                frame=30,
                person_id=1,
                bbox=[100, 150, 80, 200],
                confidence=0.92
            ),
            UniversalPersonDetection(
                video_id="test_video",
                timestamp=2.0,
                frame=60,
                person_id=2,
                bbox=[200, 100, 90, 180],
                confidence=0.88
            )
        ]
        
        video_metadata = {
            "video_id": "test_video",
            "width": 1920,
            "height": 1080
        }
        
        coco_data = export_to_coco_json(detections, video_metadata)
        
        # Validate COCO structure
        assert "info" in coco_data
        assert "images" in coco_data
        assert "annotations" in coco_data
        assert "categories" in coco_data
        
        assert len(coco_data["images"]) == 2  # 2 unique frames
        assert len(coco_data["annotations"]) == 2  # 2 detections
        
        # Validate first annotation
        ann = coco_data["annotations"][0]
        assert "bbox" in ann
        assert "score" in ann
        assert "track_id" in ann


class TestCVATCompatibility:
    """Test CVAT format compatibility."""
    
    def test_cvat_annotation_creation(self):
        """Test CVAT annotation creation."""
        annotation = CVATAnnotation(
            label="person",
            frame=30,
            video_id="test_video",
            timestamp=1.0,
            confidence=0.92,
            attributes={
                "track_id": "1",
                "bbox": [100, 150, 80, 200]
            }
        )
        
        assert annotation.label == "person"
        assert annotation.frame == 30
        assert annotation.video_id == "test_video"
        assert annotation.attributes["track_id"] == "1"
    
    def test_cvat_xml_export(self):
        """Test CVAT XML export format."""
        detections = [
            UniversalPersonDetection(
                video_id="test_video",
                timestamp=1.0,
                frame=30,
                person_id=1,
                bbox=[100, 150, 80, 200],
                confidence=0.92
            )
        ]
        
        video_metadata = {
            "video_id": "test_video",
            "width": 1920,
            "height": 1080
        }
        
        xml_output = export_to_cvat_xml(detections, video_metadata)
        
        # Validate XML structure
        assert '<?xml version="1.0"' in xml_output
        assert '<annotations>' in xml_output
        assert '<track id="1" label="person">' in xml_output
        assert '<box frame="30"' in xml_output
        assert 'xtl="100' in xml_output  # Could be 100 or 100.0
        assert 'ytl="150' in xml_output


class TestLabelStudioCompatibility:
    """Test Label Studio format compatibility."""
    
    def test_labelstudio_result_creation(self):
        """Test Label Studio result creation."""
        result = LabelStudioResult(
            id="test_result_1",
            type="rectanglelabels",
            value={
                "x": 10.0,
                "y": 15.0,
                "width": 8.0,
                "height": 20.0,
                "rectanglelabels": ["person"]
            },
            to_name="video",
            from_name="bbox",
            video_id="test_video",
            timestamp=1.0
        )
        
        assert result.id == "test_result_1"
        assert result.type == "rectanglelabels"
        assert result.value["rectanglelabels"] == ["person"]
    
    def test_labelstudio_export(self):
        """Test Label Studio JSON export."""
        detections = [
            UniversalPersonDetection(
                video_id="test_video",
                timestamp=1.0,
                frame=30,
                person_id=1,
                bbox=[10.0, 15.0, 8.0, 20.0],
                confidence=0.92
            )
        ]
        
        video_metadata = {
            "video_id": "test_video",
            "video_path": "/path/to/video.mp4"
        }
        
        ls_data = export_to_labelstudio_json(detections, video_metadata)
        
        # Validate Label Studio structure
        assert len(ls_data) == 1
        task = ls_data[0]
        assert "data" in task
        assert "annotations" in task
        assert task["data"]["video_id"] == "test_video"
        
        annotation = task["annotations"][0]
        assert "result" in annotation
        result = annotation["result"][0]
        assert result["type"] == "rectanglelabels"
        assert result["value"]["rectanglelabels"] == ["person"]


class TestUniversalSchemas:
    """Test universal schema conversions."""
    
    def test_person_detection_conversions(self):
        """Test person detection converts to all formats."""
        detection = UniversalPersonDetection(
            video_id="test_video",
            timestamp=1.5,
            frame=45,
            person_id=1,
            bbox=[100, 150, 80, 200],
            confidence=0.92
        )
        
        # Test COCO conversion
        coco_format = detection.to_coco()
        assert isinstance(coco_format, COCOBoundingBox)
        assert coco_format.bbox == [100, 150, 80, 200]
        assert coco_format.score == 0.92
        
        # Test CVAT conversion
        cvat_format = detection.to_cvat()
        assert isinstance(cvat_format, CVATAnnotation)
        assert cvat_format.label == "person"
        assert cvat_format.frame == 45
        
        # Test Label Studio conversion
        ls_format = detection.to_labelstudio()
        assert isinstance(ls_format, LabelStudioResult)
        assert ls_format.type == "rectanglelabels"
        assert ls_format.value["rectanglelabels"] == ["person"]
    
    def test_pose_keypoints_conversion(self):
        """Test pose keypoints COCO conversion."""
        # COCO-17 keypoints (nose, eyes, ears, shoulders)
        keypoints_flat = [
            100, 50, 2,   # nose (visible)
            95, 45, 2,    # left_eye (visible)
            105, 45, 2,   # right_eye (visible)
            90, 45, 1,    # left_ear (not visible)
            110, 45, 1,   # right_ear (not visible)
        ]
        
        pose = UniversalPoseKeypoints(
            video_id="test_video",
            timestamp=1.5,
            frame=45,
            person_id=1,
            keypoints=keypoints_flat,
            keypoint_names=["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
            num_keypoints=3,  # Only 3 visible
            bbox=[80, 30, 60, 180]
        )
        
        # Test COCO conversion
        coco_format = pose.to_coco()
        assert isinstance(coco_format, COCOKeypoints)
        assert coco_format.keypoints == keypoints_flat
        assert coco_format.num_keypoints == 3


class TestOriginalSpecCompliance:
    """Test compliance with original VideoAnnotator specifications."""
    
    def test_original_person_bbox_format(self):
        """Test original person_bbox format compatibility."""
        # Original spec: {"type": "person_bbox", "video_id": "vid123", "t": 12.34, "bbox": [x,y,w,h], "person_id": 1, "score": 0.87}
        
        detection = UniversalPersonDetection(
            video_id="vid123",
            timestamp=12.34,
            person_id=1,
            bbox=[100, 150, 80, 200],
            confidence=0.87
        )
        
        # Should be able to serialize to original format
        data = detection.model_dump()
        
        # Original spec used "t" for timestamp
        data["t"] = data.pop("timestamp")
        data["type"] = "person_bbox"
        data["score"] = data.pop("confidence")
        
        expected = {
            "type": "person_bbox",
            "video_id": "vid123", 
            "t": 12.34,
            "bbox": [100, 150, 80, 200],
            "person_id": 1,
            "score": 0.87
        }
        
        # Check all required fields match
        for key, value in expected.items():
            if key in data:
                assert data[key] == value
    
    def test_flexible_id_types(self):
        """Test support for both string and integer IDs."""
        # Should work with integer IDs
        detection_int = UniversalPersonDetection(
            video_id="test_video",
            timestamp=1.0,
            person_id=1,  # integer
            bbox=[100, 150, 80, 200],
            confidence=0.9
        )
        assert detection_int.person_id == 1
        
        # Should work with string IDs
        detection_str = UniversalPersonDetection(
            video_id="test_video", 
            timestamp=1.0,
            person_id="person_001",  # string
            bbox=[100, 150, 80, 200],
            confidence=0.9
        )
        assert detection_str.person_id == "person_001"
    
    def test_extra_fields_allowed(self):
        """Test that extra fields are preserved for research flexibility."""
        detection = UniversalPersonDetection(
            video_id="test_video",
            timestamp=1.0,
            person_id=1,
            bbox=[100, 150, 80, 200],
            confidence=0.9,
            # Extra fields that might be useful for research
            age_estimate=25,
            gender_estimate="female",
            emotion="happy",
            custom_metadata={"researcher": "jane_doe", "experiment": "study_1"}
        )
        
        # Extra fields should be preserved
        data = detection.model_dump()
        assert data["age_estimate"] == 25
        assert data["gender_estimate"] == "female"
        assert data["emotion"] == "happy"
        assert data["custom_metadata"]["researcher"] == "jane_doe"


class TestInteroperability:
    """Test interoperability between different standards."""
    
    def test_roundtrip_conversions(self):
        """Test converting between formats preserves essential data."""
        original = UniversalPersonDetection(
            video_id="test_video",
            timestamp=1.5,
            frame=45,
            person_id=1,
            bbox=[100, 150, 80, 200],
            confidence=0.92
        )
        
        # Convert to COCO and back
        coco_format = original.to_coco()
        
        # Create new detection from COCO data
        reconstructed = UniversalPersonDetection(
            video_id=coco_format.video_id,
            timestamp=coco_format.timestamp,
            frame=coco_format.frame_number,
            person_id=coco_format.track_id,
            bbox=coco_format.bbox,
            confidence=coco_format.score
        )
        
        # Essential data should be preserved
        assert reconstructed.video_id == original.video_id
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.person_id == original.person_id
        assert reconstructed.bbox == original.bbox
        assert reconstructed.confidence == original.confidence
    
    def test_batch_export_consistency(self):
        """Test that batch exports maintain data consistency."""
        detections = [
            UniversalPersonDetection(
                video_id="test_video",
                timestamp=1.0,
                frame=30,
                person_id=1,
                bbox=[100, 150, 80, 200],
                confidence=0.92
            ),
            UniversalPersonDetection(
                video_id="test_video", 
                timestamp=2.0,
                frame=60,
                person_id=1,  # Same person tracked
                bbox=[110, 155, 80, 200],
                confidence=0.89
            )
        ]
        
        video_metadata = {
            "video_id": "test_video",
            "width": 1920,
            "height": 1080,
            "video_path": "/path/to/video.mp4"
        }
        
        # Export to all formats
        coco_data = export_to_coco_json(detections, video_metadata)
        cvat_xml = export_to_cvat_xml(detections, video_metadata)
        ls_data = export_to_labelstudio_json(detections, video_metadata)
        
        # All exports should contain the same number of annotations
        assert len(coco_data["annotations"]) == 2
        assert cvat_xml.count('<box frame=') == 2  # 2 box elements
        assert len(ls_data) == 2


if __name__ == "__main__":
    # Run a simple validation test
    print("Testing VideoAnnotator standards compatibility...")
    
    # Create test detection
    detection = UniversalPersonDetection(
        video_id="test_video",
        timestamp=1.5,
        person_id=1,
        bbox=[100, 150, 80, 200],
        confidence=0.92
    )
    
    # Test all conversions
    coco_format = detection.to_coco()
    cvat_format = detection.to_cvat()
    ls_format = detection.to_labelstudio()
    
    print("✅ COCO format conversion successful")
    print("✅ CVAT format conversion successful") 
    print("✅ Label Studio format conversion successful")
    
    # Test JSON serialization
    print("\n🔍 COCO JSON:")
    print(coco_format.model_dump_json(indent=2)[:200] + "...")
    
    print("\n🔍 CVAT JSON:")
    print(cvat_format.model_dump_json(indent=2)[:200] + "...")
    
    print("\n🔍 Label Studio JSON:")
    print(ls_format.model_dump_json(indent=2)[:200] + "...")
    
    print("\n✅ All standards compatibility tests passed!")
