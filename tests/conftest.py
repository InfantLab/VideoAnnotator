"""
Test configuration for VideoAnnotator pipeline system.

This module contains pytest configuration and fixtures for testing
the VideoAnnotator system.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock
import cv2


@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        # Create a simple test video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))
        
        # Write a few frames
        for i in range(90):  # 3 seconds at 30fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some variation to frames
            frame[:, :, i % 3] = (i * 5) % 255
            out.write(frame)
        
        out.release()
        
        yield Path(f.name)
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Create a simple test audio file
        import scipy.io.wavfile as wavfile
        
        sample_rate = 16000
        duration = 3.0  # 3 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate a simple sine wave
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to int16
        audio_data = (audio_data * 32767).astype(np.int16)
        
        wavfile.write(f.name, sample_rate, audio_data)
        
        yield Path(f.name)
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing."""
    model = Mock()
    model.predict.return_value = [Mock()]
    model.predict.return_value[0].boxes = Mock()
    model.predict.return_value[0].boxes.data = np.array([
        [100, 100, 200, 200, 0.9, 0]  # x1, y1, x2, y2, confidence, class
    ])
    return model


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing."""
    model = Mock()
    model.transcribe.return_value = {
        'text': 'Hello world this is a test',
        'language': 'en',
        'segments': [
            {
                'start': 0.0,
                'end': 2.0,
                'text': 'Hello world'
            },
            {
                'start': 2.0,
                'end': 4.0,
                'text': 'this is a test'
            }
        ]
    }
    return model


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'scene_detection': {
            'threshold': 0.3,
            'min_scene_length': 1.0,
            'use_adaptive_threshold': True
        },
        'person_tracking': {
            'model_name': 'yolo11s',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.7
        },
        'face_analysis': {
            'backends': ['mediapipe'],
            'detection_confidence': 0.7,
            'enable_landmarks': True
        },
        'audio_processing': {
            'whisper_model': 'base',
            'sample_rate': 16000,
            'chunk_duration': 30.0
        }
    }


@pytest.fixture
def sample_scene_results():
    """Sample scene detection results for testing."""
    return {
        'scenes': [
            {
                'scene_id': 'scene_001',
                'start_time': 0.0,
                'end_time': 5.0,
                'scene_type': 'indoor',
                'confidence': 0.85
            },
            {
                'scene_id': 'scene_002',
                'start_time': 5.0,
                'end_time': 10.0,
                'scene_type': 'outdoor',
                'confidence': 0.92
            }
        ],
        'total_duration': 10.0,
        'total_scenes': 2
    }


@pytest.fixture
def sample_person_results():
    """Sample person tracking results for testing."""
    return {
        'tracks': [
            {
                'track_id': 'person_001',
                'duration': 8.0,
                'detections': [
                    {
                        'timestamp': 1.0,
                        'bounding_box': [100, 100, 200, 200],
                        'confidence': 0.9
                    }
                ]
            }
        ],
        'total_tracks': 1,
        'total_detections': 240
    }


@pytest.fixture
def sample_face_results():
    """Sample face analysis results for testing."""
    return {
        'faces': [
            {
                'face_id': 'face_001',
                'timestamp': 1.0,
                'bounding_box': [50, 50, 100, 100],
                'confidence': 0.8
            }
        ],
        'face_tracks': [
            {
                'track_id': 'face_track_001',
                'duration': 5.0,
                'faces': ['face_001']
            }
        ],
        'total_faces': 1
    }


@pytest.fixture
def sample_audio_results():
    """Sample audio processing results for testing."""
    return {
        'duration': 10.0,
        'sample_rate': 16000,
        'speech_transcription': {
            'text': 'Hello world this is a test',
            'language': 'en',
            'confidence': 0.9
        },
        'speaker_diarization': {
            'num_speakers': 2,
            'segments': [
                {
                    'start_time': 0.0,
                    'end_time': 5.0,
                    'speaker_id': 'speaker_001'
                },
                {
                    'start_time': 5.0,
                    'end_time': 10.0,
                    'speaker_id': 'speaker_002'
                }
            ]
        }
    }


# Test markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit marker to unit tests
        if "unit" in item.nodeid or "Unit" in item.name:
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid or "Integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in item.nodeid or "Performance" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add slow marker to slow tests
        if "slow" in item.nodeid or any(keyword in item.name.lower() 
                                       for keyword in ["batch", "large", "stress"]):
            item.add_marker(pytest.mark.slow)
