"""
Schema validation tests for VideoAnnotator pipeline system.

This module contains comprehensive tests for all Pydantic model schemas
used in the VideoAnnotator system.
"""

import pytest
from datetime import datetime
from typing import List, Optional
import numpy as np

# Import schema modules
from src.schemas.base_schema import BaseAnnotation, VideoMetadata, BoundingBox, Point2D, Point3D, EmotionScores
from src.schemas.scene_schema import SceneAnnotation, SceneClassification, AudioContext, SceneSegment
from src.schemas.person_schema import PersonDetection, PersonTrajectory, PoseKeypoints, PersonDetectionLegacy
from src.schemas.face_schema import FaceDetection, FaceEmotion, FaceGaze, FaceDetectionLegacy
from src.schemas.audio_schema import AudioSegment, SpeechRecognition, SpeakerDiarizationModern, AudioClassification


class TestBaseSchema:
    """Test cases for base schema classes."""
    
    def test_video_metadata_validation(self):
        """Test VideoMetadata schema validation."""
        # Valid metadata
        metadata = VideoMetadata(
            video_id="test_video_001",
            filepath="/path/to/video.mp4",
            width=1920,
            height=1080,
            fps=30.0,
            duration=120.5,
            total_frames=3615
        )
        
        assert metadata.video_id == "test_video_001"
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.fps == 30.0
        assert metadata.duration == 120.5
        assert metadata.total_frames == 3615
    
    def test_bounding_box_validation(self):
        """Test BoundingBox schema validation."""
        # Valid bounding box
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.width == 0.3
        assert bbox.height == 0.4
        assert bbox.x2 == 0.4
        assert abs(bbox.y2 - 0.6) < 1e-10
        assert bbox.center == (0.25, 0.4)
        assert bbox.area == 0.12
    
    def test_point2d_validation(self):
        """Test Point2D schema validation."""
        point = Point2D(x=100.5, y=200.3, confidence=0.88)
        
        assert point.x == 100.5
        assert point.y == 200.3
        assert point.confidence == 0.88


class TestSceneSchema:
    """Test cases for scene detection schemas."""
    
    def test_scene_annotation_validation(self):
        """Test SceneAnnotation schema validation."""
        scene = SceneAnnotation(
            video_id="test_video_001",
            timestamp=0.0,
            scene_id="scene_001",
            start_time=0.0,
            end_time=5.0,
            change_type="cut"
        )
        
        assert scene.scene_id == "scene_001"
        assert scene.start_time == 0.0
        assert scene.end_time == 5.0
        assert scene.duration == 5.0
        assert scene.change_type == "cut"
    
    def test_scene_classification_validation(self):
        """Test SceneClassification schema validation."""
        classification = SceneClassification(
            video_id="test_video_001",
            timestamp=2.5,
            scene_id="scene_001",
            label="living_room",
            confidence=0.85,
            categories={"indoor": 0.95, "living_room": 0.85, "outdoor": 0.05}
        )
        
        assert classification.scene_id == "scene_001"
        assert classification.label == "living_room"
        assert classification.confidence == 0.85
        assert "living_room" in classification.categories
    
    def test_audio_context_validation(self):
        """Test AudioContext schema validation."""
        audio_context = AudioContext(
            video_id="test_video_001",
            timestamp=1.0,
            scene_id="scene_001",
            audio_tags=["speech", "laughter"],
            speech_detected=True,
            music_detected=False,
            noise_level=0.3
        )
        
        assert audio_context.scene_id == "scene_001"
        assert "speech" in audio_context.audio_tags
        assert audio_context.speech_detected is True
        assert audio_context.noise_level == 0.3


class TestPersonSchema:
    """Test cases for person detection and tracking schemas."""
    
    def test_person_detection_validation(self):
        """Test PersonDetection schema validation."""
        detection = PersonDetection(
            video_id="test_video_001",
            timestamp=1.0,
            person_id=1,  # Changed to int
            bbox={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},  # Changed to dict
            confidence=0.95
        )
        
        assert detection.person_id == 1
        assert detection.timestamp == 1.0
        assert detection.bbox["x"] == 0.1
    
    def test_pose_keypoints_validation(self):
        """Test PoseKeypoints schema validation."""
        keypoints = PoseKeypoints(
            video_id="test_video_001",
            timestamp=1.0,
            person_id=1,  # Changed to int
            keypoints={  # Changed to dict
                "nose": Point2D(x=150.5, y=200.3, confidence=0.88),
                "left_eye": Point2D(x=140.2, y=210.1, confidence=0.92),
                "right_eye": Point2D(x=160.8, y=210.5, confidence=0.90)
            }
        )
        
        assert keypoints.person_id == 1
        assert len(keypoints.keypoints) == 3
        assert keypoints.keypoints["nose"].x == 150.5
    
    def test_person_trajectory_validation(self):
        """Test PersonTrajectory schema validation."""
        # Create sample detections first
        detection1 = PersonDetection(
            video_id="test_video_001",
            timestamp=1.0,
            person_id=1,
            bbox={"x": 100.0, "y": 200.0, "width": 50.0, "height": 100.0},
            confidence=1.0
        )
        detection2 = PersonDetection(
            video_id="test_video_001", 
            timestamp=2.0,
            person_id=1,
            bbox={"x": 105.0, "y": 202.0, "width": 50.0, "height": 100.0},
            confidence=0.95
        )
        
        trajectory = PersonTrajectory(
            video_id="test_video_001",
            timestamp=1.0,
            person_id=1,  # Changed to int
            detections=[detection1, detection2],  # Required field
            first_seen=1.0,  # Required field  
            last_seen=2.0    # Required field
        )
        
        assert trajectory.person_id == 1
        assert trajectory.duration == 1.0
        assert len(trajectory.detections) == 2


class TestFaceSchema:
    """Test cases for face detection and analysis schemas."""
    
    def test_face_detection_validation(self):
        """Test FaceDetection schema validation."""
        face = FaceDetection(
            video_id="test_video_001",
            timestamp=1.0,
            face_id=1,  # Changed to int
            person_id=1,  # Changed to int
            bbox={"x": 0.4, "y": 0.3, "width": 0.2, "height": 0.25},  # Changed to dict
            confidence=0.95
        )
        
        assert face.face_id == 1
        assert face.person_id == 1
        assert face.bbox["width"] == 0.2
    
    def test_face_emotion_validation(self):
        """Test FaceEmotion schema validation."""
        emotion = FaceEmotion(
            video_id="test_video_001",
            timestamp=1.0,
            face_id="face_001",
            emotion_scores=EmotionScores(
                happiness=0.8,
                surprise=0.15,
                neutral=0.05
            ),
            valence=0.7,
            arousal=0.6
        )
        
        assert emotion.face_id == "face_001"
        assert emotion.emotion_scores.happiness == 0.8
        assert emotion.valence == 0.7
        assert emotion.arousal == 0.6
    
    def test_face_gaze_validation(self):
        """Test FaceGaze schema validation."""
        gaze = FaceGaze(
            video_id="test_video_001",
            timestamp=1.0,
            face_id="face_001",
            gaze_direction=Point2D(x=0.2, y=-0.1, confidence=0.85),
            head_pose=Point3D(x=10.0, y=5.0, z=-2.0, confidence=0.9),
            eye_openness=0.8
        )
        
        assert gaze.face_id == "face_001"
        assert gaze.gaze_direction.x == 0.2
        assert gaze.head_pose.x == 10.0
        assert gaze.eye_openness == 0.8


class TestAudioSchema:
    """Test cases for audio processing schemas."""
    
    def test_audio_segment_validation(self):
        """Test AudioSegment schema validation."""
        audio_data = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        
        segment = AudioSegment(
            video_id="test_video_001",
            timestamp=0.0,
            start_time=0.0,
            end_time=1.0,
            audio_data=audio_data.tolist(),
            sample_rate=16000
        )
        
        assert segment.start_time == 0.0
        assert segment.end_time == 1.0
        assert segment.sample_rate == 16000
        assert len(segment.audio_data) == 16000
    
    def test_speech_recognition_validation(self):
        """Test SpeechRecognition schema validation."""
        speech = SpeechRecognition(
            video_id="test_video_001",
            timestamp=1.0,
            text="Hello world this is a test",
            language="en",
            speaker_id="speaker_001",
            words=[
                {"word": "Hello", "start": 1.0, "end": 1.5, "confidence": 0.95},
                {"word": "world", "start": 1.5, "end": 2.0, "confidence": 0.92}
            ]
        )
        
        assert speech.text == "Hello world this is a test"
        assert speech.language == "en"
        assert speech.speaker_id == "speaker_001"
        assert len(speech.words) == 2
    
    def test_speaker_diarization_validation(self):
        """Test SpeakerDiarizationModern schema validation."""
        diarization = SpeakerDiarizationModern(
            video_id="test_video_001",
            timestamp=0.0,
            num_speakers=2,
            speaker_segments=[
                {"speaker_id": "speaker_001", "start_time": 0.0, "end_time": 5.0},
                {"speaker_id": "speaker_002", "start_time": 5.0, "end_time": 10.0}
            ]
        )
        
        assert diarization.num_speakers == 2
        assert len(diarization.speaker_segments) == 2
        assert diarization.speaker_segments[0]["speaker_id"] == "speaker_001"


class TestSchemaValidation:
    """Test schema validation and error handling."""
    
    def test_invalid_confidence_scores(self):
        """Test validation of confidence scores."""
        # Test confidence out of range
        with pytest.raises(ValueError):
            Point2D(x=100.0, y=200.0, confidence=1.5)
        
        with pytest.raises(ValueError):
            Point2D(x=100.0, y=200.0, confidence=-0.1)
    
    def test_required_field_validation(self):
        """Test required field validation."""
        # Missing required fields should raise validation error
        with pytest.raises(ValueError):
            SceneAnnotation(
                video_id="test_video_001",
                timestamp=0.0
                # Missing scene_id, start_time, end_time
            )


class TestSchemaSerializationDeserialization:
    """Test schema serialization and deserialization."""
    
    def test_scene_annotation_json_serialization(self):
        """Test Scene annotation JSON serialization/deserialization."""
        original_scene = SceneAnnotation(
            video_id="test_video_001",
            timestamp=0.0,
            scene_id="scene_001",
            start_time=0.0,
            end_time=5.0,
            change_type="cut"
        )
        
        # Serialize to JSON
        scene_json = original_scene.model_dump_json()
        
        # Deserialize from JSON
        reconstructed_scene = SceneAnnotation.model_validate_json(scene_json)
        
        assert reconstructed_scene.scene_id == original_scene.scene_id
        assert reconstructed_scene.start_time == original_scene.start_time
        assert reconstructed_scene.end_time == original_scene.end_time
    
    def test_schema_validation_with_invalid_data(self):
        """Test schema validation with invalid data."""
        # Test with invalid data types
        with pytest.raises(ValueError):
            SceneAnnotation(
                video_id="test_video_001",
                timestamp="invalid_timestamp",  # Should be float
                scene_id="scene_001",
                start_time=0.0,
                end_time=5.0
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
