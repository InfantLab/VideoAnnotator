"""
Test the simplified schema approach with original specification examples.

This validates that our schemas match the JSON formats from the original specs.
"""

import pytest
import json
from src.schemas.simple_schemas import *


class TestOriginalSpecCompatibility:
    """Test compatibility with original specification examples."""
    
    def test_person_detection_original_format(self):
        """Test person detection with original spec format."""
        # From original spec: {"type": "person_bbox", "video_id": "vid123", "t": 12.34, "bbox": [x,y,w,h], "person_id": 1, "score": 0.87}
        
        person = PersonDetection(
            video_id="vid123",
            t=12.34,
            bbox=[0.1, 0.2, 0.3, 0.4],
            person_id=1,
            score=0.87
        )
        
        # Verify fields
        assert person.type == "person_bbox"
        assert person.video_id == "vid123"
        assert person.timestamp == 12.34  # timestamp property works
        assert person.bbox == [0.1, 0.2, 0.3, 0.4]
        assert person.person_id == 1
        assert person.confidence == 0.87
        
        # Verify JSON serialization matches spec
        json_output = person.model_dump(by_alias=True)
        expected_keys = {"type", "video_id", "t", "bbox", "person_id", "score"}
        assert set(json_output.keys()) >= expected_keys
    
    def test_pose_keypoints_original_format(self):
        """Test pose keypoints with original COCO format."""
        # From spec: COCO-style keypoints per frame
        
        keypoints_data = [
            {"joint": "nose", "x": 123, "y": 456, "conf": 0.98},
            {"joint": "left_eye", "x": 120, "y": 450, "conf": 0.95}
        ]
        
        pose = PoseKeypoints(
            video_id="vid123",
            t=12.34,
            person_id=1,
            keypoints=keypoints_data
        )
        
        assert pose.type == "pose_keypoints"
        assert pose.person_id == 1
        assert len(pose.keypoints) == 2
        assert pose.keypoints[0]["joint"] == "nose"
    
    def test_face_emotion_original_format(self):
        """Test facial emotion with original spec format."""
        # From spec: {"type": "facial_emotion", "video_id": "vid123", "t": 12.34, "person_id": 1, "bbox": [x,y,w,h], "emotion": "happy", "confidence": 0.91}
        
        emotion = FaceEmotion(
            video_id="vid123",
            t=12.34,
            face_id="face_001",  # String ID should work
            person_id=1,
            bbox=[0.1, 0.2, 0.15, 0.2],
            emotion="happy",
            confidence=0.91,
            # Extra emotion scores should be allowed
            happiness=0.8,
            surprise=0.1,
            neutral=0.1
        )
        
        assert emotion.type == "facial_emotion"
        assert emotion.face_id == "face_001"
        assert emotion.emotion == "happy"
        # Extra fields should be preserved
        json_data = emotion.model_dump()
        assert "happiness" in json_data
        assert "surprise" in json_data
    
    def test_speech_recognition_original_format(self):
        """Test speech recognition with original spec format."""
        # From spec: {"type": "transcript", "video_id": "vid123", "start": 12.0, "end": 14.2, "text": "Hello baby", "confidence": 0.92}
        
        speech = SpeechRecognition(
            video_id="vid123", 
            t=12.0,  # Use t as the timestamp field
            start=12.0,
            end=14.2,
            text="Hello baby",
            confidence=0.92
        )
        
        assert speech.type == "transcript"
        assert speech.start == 12.0
        assert speech.end == 14.2
        assert speech.text == "Hello baby"
        assert speech.timestamp == 12.0  # Should use t parameter
    
    def test_speaker_diarization_original_format(self):
        """Test speaker diarization with original spec format."""
        # From spec: {"type": "speaker_turn", "video_id": "vid123", "start": 12.0, "end": 14.2, "speaker": "spk_01"}
        
        diarization = SpeakerDiarization(
            video_id="vid123",
            t=12.0,  # Use t as the timestamp field
            start=12.0,
            end=14.2, 
            speaker="spk_01"
        )
        
        assert diarization.type == "speaker_turn"
        assert diarization.speaker == "spk_01"
        assert diarization.timestamp == 12.0
    
    def test_audio_event_original_format(self):
        """Test audio event with original spec format."""
        # From spec: {"type": "audio_event", "video_id": "vid123", "start": 7.0, "end": 8.4, "event": "baby_laugh", "confidence": 0.88}
        
        audio_event = AudioEvent(
            video_id="vid123",
            t=7.0,  # Use t as the timestamp field
            start=7.0,
            end=8.4,
            event="baby_laugh",
            confidence=0.88
        )
        
        assert audio_event.type == "audio_event" 
        assert audio_event.event == "baby_laugh"
        assert audio_event.timestamp == 7.0
    
    def test_scene_annotation_original_format(self):
        """Test scene annotation with original spec format."""
        # From spec: {"type": "scene_label", "video_id": "vid123", "start": 0, "end": 20, "label": "indoor-living-room", "confidence": 0.88}
        
        scene = SceneAnnotation(
            video_id="vid123",
            t=0,  # Use t as the timestamp field
            start=0,
            end=20,
            label="indoor-living-room",
            confidence=0.88
        )
        
        assert scene.type == "scene_label"
        assert scene.label == "indoor-living-room"
        assert scene.timestamp == 0
    
    def test_object_detection_original_format(self):
        """Test object detection with original spec format."""
        # From spec: {"type": "object", "video_id": "vid123", "t": 12.34, "label": "toy", "bbox": [x,y,w,h], "score": 0.91}
        
        obj = ObjectDetection(
            video_id="vid123",
            t=12.34,
            label="toy", 
            bbox=[0.2, 0.3, 0.1, 0.15],
            score=0.91
        )
        
        assert obj.type == "object"
        assert obj.label == "toy"
        assert obj.confidence == 0.91


class TestAnnotationToolCompatibility:
    """Test compatibility with annotation tools (CVAT, LabelStudio)."""
    
    def test_cvat_format_export(self):
        """Test CVAT format export."""
        annotations = [
            PersonDetection(video_id="test", t=1.0, bbox=[0.1, 0.2, 0.3, 0.4], person_id=1),
            FaceDetection(video_id="test", t=2.0, bbox=[0.15, 0.25, 0.1, 0.1], face_id="face_001")
        ]
        
        cvat_format = to_cvat_format(annotations)
        
        assert "version" in cvat_format
        assert "meta" in cvat_format
        assert "annotations" in cvat_format
        assert len(cvat_format["annotations"]) == 2
    
    def test_labelstudio_format_export(self):
        """Test LabelStudio format export."""
        annotations = [
            SpeechRecognition(video_id="test", t=1.0, start=1.0, end=2.0, text="Hello"),
            AudioEvent(video_id="test", t=3.0, start=3.0, end=4.0, event="laugh")
        ]
        
        ls_format = to_labelstudio_format(annotations)
        
        assert isinstance(ls_format, list)
        assert len(ls_format) == 2
        assert all("type" in ann for ann in ls_format)
    
    def test_from_dict_creation(self):
        """Test creating annotations from dictionaries."""
        # Test person detection
        person_data = {
            "type": "person_bbox",
            "video_id": "test",
            "t": 1.0,
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "person_id": 1
        }
        
        person = from_dict(person_data)
        assert isinstance(person, PersonDetection)
        assert person.person_id == 1
        
        # Test speech recognition
        speech_data = {
            "type": "transcript", 
            "video_id": "test",
            "t": 1.0,
            "start": 1.0,
            "end": 2.0,
            "text": "Hello world"
        }
        
        speech = from_dict(speech_data)
        assert isinstance(speech, SpeechRecognition)
        assert speech.text == "Hello world"


class TestFlexibilityAndExtensibility:
    """Test schema flexibility for real-world usage."""
    
    def test_extra_fields_allowed(self):
        """Test that extra fields are preserved."""
        person = PersonDetection(
            video_id="test",
            t=1.0,
            bbox=[0.1, 0.2, 0.3, 0.4],
            person_id=1,
            # Extra fields that might be added by different models
            track_id="track_123",
            age_estimate=25,
            gender="female",
            pose_confidence=0.95,
            metadata={"model": "yolo11", "version": "1.0"}
        )
        
        # All extra fields should be preserved
        data = person.model_dump()
        assert data["track_id"] == "track_123"
        assert data["age_estimate"] == 25
        assert data["gender"] == "female"
        assert data["pose_confidence"] == 0.95
        assert data["metadata"]["model"] == "yolo11"
    
    def test_string_and_integer_ids_both_work(self):
        """Test that both string and integer IDs work."""
        # Integer IDs
        face1 = FaceDetection(video_id="test", t=1.0, bbox=[0.1, 0.2, 0.1, 0.1], face_id=1)
        assert face1.face_id == 1
        
        # String IDs
        face2 = FaceDetection(video_id="test", t=1.0, bbox=[0.1, 0.2, 0.1, 0.1], face_id="face_001")
        assert face2.face_id == "face_001"
        
        # Both should serialize properly
        assert face1.model_dump()["face_id"] == 1
        assert face2.model_dump()["face_id"] == "face_001"
    
    def test_json_round_trip(self):
        """Test JSON serialization and deserialization."""
        original = PersonDetection(
            video_id="test_video",
            t=12.34,
            bbox=[0.1, 0.2, 0.3, 0.4],
            person_id="person_001",
            score=0.87,
            extra_field="extra_value"
        )
        
        # Serialize to JSON string
        json_str = original.model_dump_json(by_alias=True)
        
        # Parse back from JSON
        json_data = json.loads(json_str)
        reconstructed = PersonDetection(**json_data)
        
        # Should be identical
        assert reconstructed.video_id == original.video_id
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.bbox == original.bbox
        assert reconstructed.person_id == original.person_id
        assert reconstructed.model_dump()["extra_field"] == "extra_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
