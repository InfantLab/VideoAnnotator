"""
Unit tests for Face Analysis Pipeline - Standards-Only Version.

Tests cover face detection, emotion analysis, landmarks using COCO format output.
All outputs comply with official COCO annotation standards.
"""

import pytest
import tempfile
import numpy as np
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.pipelines.face_analysis.face_pipeline import FaceAnalysisPipeline
from src.exporters.native_formats import create_coco_annotation, validate_coco_json


@pytest.mark.unit
class TestFaceAnalysisPipeline:
    """Test cases for standards-only face analysis pipeline."""
    
    def test_face_pipeline_initialization(self):
        """Test face pipeline initialization with custom config."""
        config = {
            "detection_backend": "opencv",
            "emotion_backend": "deepface", 
            "landmark_backend": "mediapipe",
            "coco_categories": [{"id": 1, "name": "face"}]
        }
        
        pipeline = FaceAnalysisPipeline(config)
        assert pipeline.config["detection_backend"] == "opencv"
        assert pipeline.config["emotion_backend"] == "deepface"
    
    def test_face_pipeline_default_config(self):
        """Test face pipeline with default configuration."""
        pipeline = FaceAnalysisPipeline()
        
        # Should have default backends
        assert "detection_backend" in pipeline.config
        assert "coco_categories" in pipeline.config
        assert isinstance(pipeline.config["coco_categories"], list)
    
    @patch('cv2.VideoCapture')
    def test_face_detection_returns_coco_format(self, mock_cv2):
        """Test that face detection returns COCO format annotations."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,  # FPS
            3: 640,   # WIDTH
            4: 480,   # HEIGHT
            7: 900    # FRAME_COUNT
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        
        pipeline = FaceAnalysisPipeline()
        
        # Mock face detection to return bounding boxes
        with patch.object(pipeline, '_detect_faces_in_frame') as mock_detect:
            mock_detect.return_value = [
                {
                    'id': 1,
                    'image_id': 'test_video_frame_0',
                    'category_id': 1,
                    'bbox': [100, 100, 50, 60],  # [x, y, width, height]
                    'area': 3000,
                    'score': 0.95,
                    'timestamp': 0.0,
                    'frame_number': 0
                }
            ]
            
            results = pipeline.process("test_video.mp4", output_dir=None)
            
            # Should return list of COCO annotations
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Check COCO format compliance
            annotation = results[0]
            assert 'id' in annotation
            assert 'image_id' in annotation  
            assert 'category_id' in annotation
            assert 'bbox' in annotation
            assert 'area' in annotation
            assert 'score' in annotation
            
            # Check bbox format [x, y, width, height]
            bbox = annotation['bbox']
            assert len(bbox) == 4
            assert all(isinstance(coord, (int, float)) for coord in bbox)
    
    def test_coco_annotation_creation(self):
        """Test creation of COCO annotation with face detection data."""
        annotation = create_coco_annotation(
            annotation_id=1,
            image_id="test_frame_001",
            category_id=1,  # Face category
            bbox=[100, 150, 80, 90],  # [x, y, width, height]
            score=0.92,
            # VideoAnnotator extensions
            timestamp=1.5,
            frame_number=45,
            emotions={"happy": 0.8, "neutral": 0.2}
        )
        
        # Verify COCO compliance
        assert annotation['id'] == 1
        assert annotation['image_id'] == "test_frame_001"
        assert annotation['category_id'] == 1
        assert annotation['bbox'] == [100, 150, 80, 90]
        assert annotation['area'] == 80 * 90  # width * height
        assert annotation['score'] == 0.92
        
        # Verify VideoAnnotator extensions
        assert annotation['timestamp'] == 1.5
        assert annotation['frame_number'] == 45
        assert annotation['emotions'] == {"happy": 0.8, "neutral": 0.2}
    
    @patch('cv2.VideoCapture')
    def test_face_pipeline_with_emotions(self, mock_cv2):
        """Test face pipeline with emotion analysis enabled."""
        # Setup mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0, 3: 640, 4: 480, 7: 30
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        
        config = {"enable_emotions": True, "emotion_backend": "deepface"}
        pipeline = FaceAnalysisPipeline(config)
        
        # Mock face detection with emotions
        with patch.object(pipeline, '_detect_faces_in_frame') as mock_detect:
            mock_detect.return_value = [
                {
                    'id': 1,
                    'image_id': 'test_video_frame_0',
                    'category_id': 1,
                    'bbox': [50, 60, 40, 50],
                    'area': 2000,
                    'score': 0.88,
                    'emotions': {
                        'happy': 0.7,
                        'neutral': 0.2,
                        'sad': 0.1
                    },
                    'timestamp': 0.0,
                    'frame_number': 0
                }
            ]
            
            results = pipeline.process("test_video.mp4", output_dir=None)
            
            # Check emotions are included
            assert len(results) > 0
            annotation = results[0]
            assert 'emotions' in annotation
            assert isinstance(annotation['emotions'], dict)
            assert 'happy' in annotation['emotions']
    
    @patch('cv2.VideoCapture')
    def test_face_pipeline_with_landmarks(self, mock_cv2):
        """Test face pipeline with facial landmark detection."""
        # Setup mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0, 3: 640, 4: 480, 7: 30
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        
        config = {"enable_landmarks": True, "landmark_backend": "mediapipe"}
        pipeline = FaceAnalysisPipeline(config)
        
        # Mock face detection with landmarks
        with patch.object(pipeline, '_detect_faces_in_frame') as mock_detect:
            mock_detect.return_value = [
                {
                    'id': 1,
                    'image_id': 'test_video_frame_0',
                    'category_id': 1,
                    'bbox': [75, 85, 60, 70],
                    'area': 4200,
                    'score': 0.93,
                    'keypoints': [
                        # Simplified: [x1, y1, v1, x2, y2, v2, ...]
                        100, 110, 2,  # Left eye
                        120, 110, 2,  # Right eye
                        110, 125, 2,  # Nose
                        105, 140, 2,  # Left mouth corner
                        115, 140, 2   # Right mouth corner
                    ],
                    'num_keypoints': 5,
                    'timestamp': 0.0,
                    'frame_number': 0
                }
            ]
            
            results = pipeline.process("test_video.mp4", output_dir=None)
            
            # Check landmarks are included
            assert len(results) > 0
            annotation = results[0]
            assert 'keypoints' in annotation
            assert 'num_keypoints' in annotation
            assert isinstance(annotation['keypoints'], list)
            assert annotation['num_keypoints'] == 5
    
    def test_coco_validation_integration(self):
        """Test COCO format validation with face annotations."""
        # Create sample annotations
        annotations = [
            create_coco_annotation(
                annotation_id=1,
                image_id="frame_001",
                category_id=1,
                bbox=[10, 20, 30, 40],
                score=0.9
            ),
            create_coco_annotation(
                annotation_id=2,
                image_id="frame_002", 
                category_id=1,
                bbox=[50, 60, 35, 45],
                score=0.85
            )
        ]
        
        # Create images list
        images = [
            {
                'id': 'frame_001',
                'width': 640,
                'height': 480,
                'file_name': 'frame_001.jpg'
            },
            {
                'id': 'frame_002',
                'width': 640,
                'height': 480,
                'file_name': 'frame_002.jpg'
            }
        ]
        
        # Test COCO JSON export and validation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            coco_data = {
                'images': images,
                'annotations': annotations,
                'categories': [{'id': 1, 'name': 'face'}]
            }
            json.dump(coco_data, f)
            temp_path = f.name
        
        try:
            # Validate COCO format
            validation_result = validate_coco_json(temp_path, "face_detection")
            assert validation_result.is_valid or len(validation_result.warnings) == 0
            
        finally:
            os.unlink(temp_path)
    
    @patch('cv2.VideoCapture')
    def test_pipeline_output_directory_creation(self, mock_cv2):
        """Test that pipeline creates output directory and saves results."""
        # Setup mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0, 3: 640, 4: 480, 7: 30
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        
        pipeline = FaceAnalysisPipeline()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "face_output"
            
            # Mock face detection
            with patch.object(pipeline, '_detect_faces_in_frame') as mock_detect:
                mock_detect.return_value = [
                    create_coco_annotation(
                        annotation_id=1,
                        image_id="test_video_frame_0",
                        category_id=1,
                        bbox=[10, 20, 30, 40],
                        score=0.9
                    )
                ]
                
                results = pipeline.process(
                    "test_video.mp4", 
                    output_dir=str(output_dir)
                )
                
                # Check output directory exists
                assert output_dir.exists()
                
                # Check COCO JSON file was created
                coco_files = list(output_dir.glob("*.json"))
                assert len(coco_files) > 0
    
    def test_video_metadata_extraction(self):
        """Test video metadata extraction for COCO image entries."""
        pipeline = FaceAnalysisPipeline()
        
        # Mock video file
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                0: 29.97,  # FPS
                3: 1920,   # WIDTH
                4: 1080,   # HEIGHT
                7: 1500    # FRAME_COUNT
            }.get(prop, 0)
            mock_cv2.return_value = mock_cap
            
            metadata = pipeline._get_video_metadata("test_video.mp4")
            
            assert metadata['video_id'] == 'test_video'
            assert metadata['width'] == 1920
            assert metadata['height'] == 1080
            assert metadata['fps'] == 29.97
            assert metadata['total_frames'] == 1500
            
            mock_cap.release.assert_called_once()
    
    def test_pipeline_cleanup(self):
        """Test pipeline resource cleanup."""
        pipeline = FaceAnalysisPipeline()
        
        # Simulate loaded models
        pipeline.face_detector = Mock()
        pipeline.emotion_analyzer = Mock()
        pipeline.landmark_detector = Mock()
        
        # Cleanup should not raise errors
        pipeline.cleanup()
        
        # Models should be cleared
        assert pipeline.face_detector is None
        assert pipeline.emotion_analyzer is None
        assert pipeline.landmark_detector is None
