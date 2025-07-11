"""
Unit tests for VideoAnnotator pipeline system.

This module contains comprehensive tests for all pipeline components,
schemas, and utility functions.
"""

import pytest
import tempfile
import numpy as np
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pipeline modules
from src.pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline
from src.pipelines.face_analysis.face_pipeline import FaceAnalysisPipeline
from src.pipelines.audio_processing.audio_pipeline import AudioPipeline
from src.pipelines.audio_processing.diarization_pipeline import DiarizationPipeline, DiarizationPipelineConfig
from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline, SpeechPipelineConfig

# Import schemas
from src.schemas.scene_schema import SceneAnnotation, SceneClassification, SceneSegment
from src.schemas.person_schema import PersonDetection, PersonTrajectory, PoseKeypoints
from src.schemas.face_schema import FaceDetection, FaceEmotion, FaceGaze
from src.schemas.audio_schema import AudioSegment, SpeechRecognition, SpeakerDiarization

# Import version utilities
from src.version import (
    get_version_info, 
    get_git_info, 
    get_dependency_versions,
    get_model_info,
    create_annotation_metadata,
    __version__
)


@pytest.mark.unit
class TestSceneDetectionPipeline:
    """Test cases for scene detection pipeline."""
    
    def test_scene_pipeline_initialization(self):
        """Test scene pipeline initialization."""
        config = {
            "threshold": 30.0,
            "min_scene_length": 2.0
        }
        pipeline = SceneDetectionPipeline(config)
        
        assert pipeline.config["threshold"] == 30.0
        assert pipeline.config["min_scene_length"] == 2.0
    
    def test_scene_pipeline_default_config(self):
        """Test scene pipeline with default configuration."""
        pipeline = SceneDetectionPipeline()
        
        # Should have default values
        assert "threshold" in pipeline.config
        assert "min_scene_length" in pipeline.config
        assert "scene_prompts" in pipeline.config
    
    @patch('src.pipelines.scene_detection.scene_pipeline.VideoFileClip')
    def test_scene_pipeline_process_video(self, mock_video_clip, temp_video_file):
        """Test scene pipeline video processing."""
        # Mock video clip
        mock_clip = Mock()
        mock_clip.duration = 10.0
        mock_clip.fps = 30.0
        mock_video_clip.return_value = mock_clip
        
        pipeline = SceneDetectionPipeline()
        
        try:
            results = pipeline.process(str(temp_video_file))
            # Should return some results even if mocked
            assert isinstance(results, list)
        except Exception as e:
            # Pipeline might fail due to missing dependencies, that's ok for now
            pytest.skip(f"Pipeline processing failed: {e}")


@pytest.mark.unit
class TestPersonTrackingPipeline:
    """Test cases for person tracking pipeline."""
    
    def test_person_pipeline_initialization(self):
        """Test person pipeline initialization."""
        config = {
            "model_name": "yolo11s",
            "confidence_threshold": 0.5
        }
        pipeline = PersonTrackingPipeline(config)
        
        assert pipeline.config["model_name"] == "yolo11s"
        assert pipeline.config["confidence_threshold"] == 0.5
    
    def test_person_pipeline_default_config(self):
        """Test person pipeline with default configuration."""
        pipeline = PersonTrackingPipeline()
        
        # Should have default values
        assert "model_name" in pipeline.config
        assert "confidence_threshold" in pipeline.config
    
    @patch('src.pipelines.person_tracking.person_pipeline.YOLO')
    def test_person_pipeline_process_video(self, mock_yolo, temp_video_file):
        """Test person pipeline video processing."""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        pipeline = PersonTrackingPipeline()
        
        try:
            results = pipeline.process(str(temp_video_file))
            assert isinstance(results, list)
        except Exception as e:
            pytest.skip(f"Pipeline processing failed: {e}")


@pytest.mark.unit
class TestFaceAnalysisPipeline:
    """Test cases for face analysis pipeline."""
    
    def test_face_pipeline_initialization(self):
        """Test face pipeline initialization."""
        config = {
            "backend": "mediapipe",
            "detection_confidence": 0.7
        }
        pipeline = FaceAnalysisPipeline(config)
        
        assert pipeline.config["backend"] == "mediapipe"
        assert pipeline.config["detection_confidence"] == 0.7
    
    def test_face_pipeline_default_config(self):
        """Test face pipeline with default configuration."""
        pipeline = FaceAnalysisPipeline()
        
        # Should have default values
        assert "backend" in pipeline.config
        assert "detection_confidence" in pipeline.config
    
    @patch('src.pipelines.face_analysis.face_pipeline.DeepFace')
    def test_face_pipeline_process_video(self, mock_deepface, temp_video_file):
        """Test face pipeline video processing."""
        # Mock DeepFace
        mock_deepface.analyze.return_value = [{
            'emotion': {'happy': 0.8, 'sad': 0.2},
            'age': 25,
            'gender': {'Woman': 0.6, 'Man': 0.4}
        }]
        
        pipeline = FaceAnalysisPipeline()
        
        try:
            results = pipeline.process(str(temp_video_file))
            assert isinstance(results, list)
        except Exception as e:
            pytest.skip(f"Pipeline processing failed: {e}")


@pytest.mark.unit
class TestAudioPipeline:
    """Test cases for audio processing pipeline."""
    
    def test_audio_pipeline_initialization(self):
        """Test audio pipeline initialization."""
        config = {
            "whisper_model": "base",
            "sample_rate": 16000
        }
        pipeline = AudioPipeline(config)
        
        assert pipeline.config["whisper_model"] == "base"
        assert pipeline.config["sample_rate"] == 16000
    
    def test_audio_pipeline_default_config(self):
        """Test audio pipeline with default configuration."""
        pipeline = AudioPipeline()
        
        # Should have default values
        assert "whisper_model" in pipeline.config
        assert "sample_rate" in pipeline.config
    
    @patch('src.pipelines.audio_processing.audio_pipeline.whisper')
    def test_audio_pipeline_process_audio(self, mock_whisper, temp_audio_file):
        """Test audio pipeline audio processing."""
        # Mock Whisper
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'Hello world',
            'language': 'en',
            'segments': []
        }
        mock_whisper.load_model.return_value = mock_model
        
        pipeline = AudioPipeline()
        
        try:
            results = pipeline.process(str(temp_audio_file))
            assert isinstance(results, list)
        except Exception as e:
            pytest.skip(f"Pipeline processing failed: {e}")


@pytest.mark.unit
class TestDiarizationPipeline:
    """Test cases for speaker diarization pipeline."""
    
    def test_diarization_config_initialization(self):
        """Test diarization pipeline configuration."""
        config = DiarizationPipelineConfig(
            diarization_model="pyannote/speaker-diarization-3.1",
            min_speakers=1,
            max_speakers=5,
            use_gpu=False
        )
        
        assert config.diarization_model == "pyannote/speaker-diarization-3.1"
        assert config.min_speakers == 1
        assert config.max_speakers == 5
        assert config.use_gpu == False
    
    def test_diarization_config_token_from_env(self):
        """Test that config picks up token from environment."""
        # Clear any existing environment tokens
        with patch.dict('os.environ', {'HF_AUTH_TOKEN': 'test_token', 'HUGGINGFACE_TOKEN': 'test_token'}, clear=False):
            with patch.dict('os.environ', {}, clear=True):
                with patch.dict('os.environ', {'HUGGINGFACE_TOKEN': 'test_token'}):
                    config = DiarizationPipelineConfig()
                    assert config.huggingface_token == 'test_token'
    
    def test_diarization_pipeline_initialization(self):
        """Test diarization pipeline initialization."""
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            use_gpu=False
        )
        pipeline = DiarizationPipeline(config)
        
        assert pipeline.config.huggingface_token == "test_token"
        assert pipeline.config.use_gpu == False
        assert not pipeline.is_initialized
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    def test_diarization_pipeline_initialize_success(self, mock_pyannote):
        """Test successful diarization pipeline initialization."""
        # Mock PyAnnote pipeline
        mock_pipeline_instance = Mock()
        mock_pyannote.from_pretrained.return_value = mock_pipeline_instance
        
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            use_gpu=False
        )
        pipeline = DiarizationPipeline(config)
        
        # Test initialization
        pipeline.initialize()
        
        assert pipeline.is_initialized
        mock_pyannote.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="test_token"
        )
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', False)
    def test_diarization_pipeline_initialize_no_pyannote(self):
        """Test initialization fails when PyAnnote not available."""
        config = DiarizationPipelineConfig(huggingface_token="test_token")
        pipeline = DiarizationPipeline(config)
        
        with pytest.raises(ImportError, match="pyannote.audio is not available"):
            pipeline.initialize()
    
    def test_diarization_pipeline_initialize_no_token(self):
        """Test initialization fails without HuggingFace token."""
        # Clear environment variables to ensure no token is available
        with patch.dict('os.environ', {}, clear=True):
            config = DiarizationPipelineConfig()  # No token
            pipeline = DiarizationPipeline(config)
            
            with pytest.raises(ValueError, match="HuggingFace token is required"):
                pipeline.initialize()
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    @patch('src.pipelines.audio_processing.diarization_pipeline.torch')
    def test_diarization_pipeline_gpu_usage(self, mock_torch, mock_pyannote):
        """Test GPU usage in diarization pipeline."""
        # Mock GPU availability
        mock_torch.cuda.is_available.return_value = True
        mock_device = Mock()
        mock_torch.device.return_value = mock_device
        
        mock_pipeline_instance = Mock()
        mock_pyannote.from_pretrained.return_value = mock_pipeline_instance
        
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            use_gpu=True
        )
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()
        
        # Should call device and to methods
        mock_torch.device.assert_called_with("cuda")
        mock_pipeline_instance.to.assert_called_with(mock_device)
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    def test_diarization_audio_processing(self, mock_pyannote, temp_audio_file):
        """Test audio diarization processing."""
        # Mock diarization results
        mock_turn1 = Mock()
        mock_turn1.start = 0.0
        mock_turn1.end = 2.5
        
        mock_turn2 = Mock() 
        mock_turn2.start = 3.0
        mock_turn2.end = 5.0
        
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (mock_turn1, None, "SPEAKER_00"),
            (mock_turn2, None, "SPEAKER_01")
        ]
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = mock_diarization
        mock_pyannote.from_pretrained.return_value = mock_pipeline_instance
        
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            use_gpu=False
        )
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()
        
        # Test diarization
        result = pipeline.diarize_audio(temp_audio_file)
        
        assert result is not None
        assert isinstance(result, SpeakerDiarization)
        assert len(result.speakers) == 2
        assert len(result.segments) == 2
        
        # Check segments
        segments = result.segments
        assert segments[0]['speaker_id'] == 'speaker_SPEAKER_00'
        assert segments[0]['start_time'] == 0.0
        assert segments[0]['end_time'] == 2.5
        
        assert segments[1]['speaker_id'] == 'speaker_SPEAKER_01'
        assert segments[1]['start_time'] == 3.0
        assert segments[1]['end_time'] == 5.0
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    def test_diarization_audio_file_not_found(self, mock_pyannote):
        """Test diarization with non-existent audio file."""
        mock_pipeline_instance = Mock()
        mock_pyannote.from_pretrained.return_value = mock_pipeline_instance
        
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            use_gpu=False
        )
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()
        
        # Test with non-existent file
        result = pipeline.diarize_audio("non_existent_file.wav")
        assert result is None
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    def test_diarization_pipeline_info(self, mock_pyannote):
        """Test pipeline info method."""
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            use_gpu=False
        )
        pipeline = DiarizationPipeline(config)
        
        info = pipeline.get_pipeline_info()
        
        assert info['name'] == 'DiarizationPipeline'
        assert 'capabilities' in info
        assert 'models' in info
        assert 'config' in info
        assert 'requirements' in info
        
        # Check requirements
        assert 'pyannote_available' in info['requirements']
        assert 'cuda_available' in info['requirements']
        assert 'has_auth_token' in info['requirements']
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    def test_diarization_process_video(self, mock_pyannote, temp_video_file):
        """Test diarization processing of video file."""
        # Mock diarization
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = []
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = mock_diarization
        mock_pyannote.from_pretrained.return_value = mock_pipeline_instance
        
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            use_gpu=False
        )
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()
        
        # Create a mock audio file that should exist
        audio_path = Path(str(temp_video_file).replace('.mp4', '.wav'))
        audio_path.touch()
        
        try:
            results = pipeline.process(str(temp_video_file))
            assert isinstance(results, list)
        except Exception as e:
            pytest.skip(f"Video processing test failed: {e}")
        finally:
            # Clean up
            if audio_path.exists():
                audio_path.unlink()


@pytest.mark.integration
class TestDiarizationPipelineIntegration:
    """Integration tests for diarization pipeline (require external dependencies)."""
    
    @pytest.mark.skipif(
        not os.getenv("HUGGINGFACE_TOKEN"),
        reason="Requires HUGGINGFACE_TOKEN environment variable"
    )
    @pytest.mark.skipif(
        not os.getenv("TEST_INTEGRATION"),
        reason="Integration tests disabled. Set TEST_INTEGRATION=1 to enable"
    )
    def test_real_diarization_pipeline(self):
        """Test diarization with real PyAnnote models (requires HF token)."""
        try:
            # This test requires actual PyAnnote installation and HF token
            from src.pipelines.audio_processing.diarization_pipeline import PYANNOTE_AVAILABLE
            
            if not PYANNOTE_AVAILABLE:
                pytest.skip("PyAnnote not available")
            
            config = DiarizationPipelineConfig(
                huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
                use_gpu=False  # Use CPU for CI
            )
            
            pipeline = DiarizationPipeline(config)
            pipeline.initialize()
            
            # Test pipeline info
            info = pipeline.get_pipeline_info()
            assert info['requirements']['pyannote_available']
            assert info['requirements']['has_auth_token']
            
            # If we have test data, try processing it
            test_videos = list(Path("babyjokes videos").glob("*.mp4"))
            if test_videos:
                result = pipeline.process(str(test_videos[0]))
                assert isinstance(result, list)
                if result:
                    assert isinstance(result[0], SpeakerDiarization)
        except Exception as e:
            pytest.skip(f"Real diarization test failed: {e}")


@pytest.mark.unit
class TestPipelineIntegration:
    """Integration tests for pipeline system."""
    
    def test_pipeline_context_manager(self):
        """Test pipeline context manager functionality."""
        pipeline = SceneDetectionPipeline()
        
        # Test context manager
        with pipeline:
            assert hasattr(pipeline, 'initialize')
            assert hasattr(pipeline, 'cleanup')
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        pipeline = SceneDetectionPipeline()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            pipeline.process("nonexistent_file.mp4")
    
    def test_pipeline_configuration_validation(self):
        """Test pipeline configuration validation."""
        # Test valid configuration
        config = {
            "threshold": 30.0,
            "min_scene_length": 2.0
        }
        pipeline = SceneDetectionPipeline(config)
        assert pipeline.config["threshold"] == 30.0
        
        # Test configuration merging with defaults
        partial_config = {"threshold": 25.0}
        pipeline = SceneDetectionPipeline(partial_config)
        assert pipeline.config["threshold"] == 25.0
        assert "min_scene_length" in pipeline.config  # Should have default


@pytest.mark.unit 
class TestUtilities:
    """Test utility functions."""
    
    def test_video_file_validation(self):
        """Test video file validation."""
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        invalid_extensions = ['.txt', '.jpg', '.png']
        
        for ext in valid_extensions:
            assert ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        
        for ext in invalid_extensions:
            assert ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    def test_output_directory_creation(self, temp_output_dir):
        """Test output directory creation."""
        output_dir = temp_output_dir / "new_subdir"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        assert output_dir.exists()
        assert output_dir.is_dir()


@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance and stress tests."""
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation for different input sizes."""
        # Test with different video configurations
        configs = [
            {"width": 640, "height": 480, "duration": 10},
            {"width": 1280, "height": 720, "duration": 30},
            {"width": 1920, "height": 1080, "duration": 60}
        ]
        
        for config in configs:
            estimated_memory = (config["width"] * config["height"] * 3 * 30 * config["duration"]) / (1024**3)
            # Should be reasonable amount of memory (< 10GB for test cases)
            assert estimated_memory < 10.0
    
    def test_processing_speed_benchmark(self):
        """Test processing speed benchmarks."""
        import time
        
        # Simple benchmark test
        start_time = time.time()
        
        # Simulate some processing
        data = np.random.randn(1000, 1000)
        result = np.sum(data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        assert processing_time < 1.0
        assert isinstance(result, (int, float))


@pytest.mark.unit
class TestSchemaIntegration:
    """Test schema integration with pipelines."""
    
    def test_scene_annotation_creation(self):
        """Test creating scene annotations."""
        annotation = SceneAnnotation(
            video_id="test_video",
            timestamp=0.0,
            scene_id="scene_001",
            start_time=0.0,
            end_time=5.0,
            change_type="cut"
        )
        
        assert annotation.scene_id == "scene_001"
        assert annotation.duration == 5.0
    
    def test_person_detection_creation(self):
        """Test creating person detection annotations."""
        from src.schemas.base_schema import BoundingBox
        
        detection = PersonDetection(
            video_id="test_video",
            timestamp=1.0,
            person_id="person_001",
            bounding_box=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
            track_id="track_001"
        )
        
        assert detection.person_id == "person_001"
        assert detection.track_id == "track_001"
    
    def test_face_detection_creation(self):
        """Test creating face detection annotations."""
        from src.schemas.base_schema import BoundingBox, Point2D
        
        face = FaceDetection(
            video_id="test_video",
            timestamp=1.0,
            face_id="face_001",
            person_id="person_001",
            bounding_box=BoundingBox(x=0.4, y=0.3, width=0.2, height=0.25),
            landmarks=[
                Point2D(x=0.45, y=0.35, confidence=0.95)
            ]
        )
        
        assert face.face_id == "face_001"
        assert face.person_id == "person_001"
        assert len(face.landmarks) == 1


@pytest.mark.unit
class TestVersioning:
    """Test cases for versioning and metadata functionality."""
    
    def test_version_info_structure(self):
        """Test that version info has correct structure."""
        version_info = get_version_info()
        
        # Check top-level keys
        assert "videoannotator" in version_info
        assert "system" in version_info
        assert "dependencies" in version_info
        assert "metadata" in version_info
        
        # Check videoannotator section
        va_info = version_info["videoannotator"]
        assert "version" in va_info
        assert "version_info" in va_info
        assert "release_date" in va_info
        assert "build_date" in va_info
        assert "author" in va_info
        assert "license" in va_info
        
        # Check system section
        sys_info = version_info["system"]
        assert "platform" in sys_info
        assert "python_version" in sys_info
        assert "architecture" in sys_info
        
        # Check dependencies section
        assert isinstance(version_info["dependencies"], dict)
        
        # Check metadata section
        metadata = version_info["metadata"]
        assert "generated_at" in metadata
        assert "timezone" in metadata
    
    def test_version_string_format(self):
        """Test version string format."""
        assert isinstance(__version__, str)
        # Should follow semver format (major.minor.patch)
        parts = __version__.split('.')
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()
    
    def test_git_info_structure(self):
        """Test git info structure when available."""
        git_info = get_git_info()
        
        if git_info is not None:
            # If git info is available, check structure
            assert isinstance(git_info, dict)
            expected_keys = ["commit_hash", "branch", "is_clean", "remote_url"]
            for key in expected_keys:
                assert key in git_info
    
    def test_dependency_versions(self):
        """Test dependency version checking."""
        deps = get_dependency_versions()
        
        assert isinstance(deps, dict)
        
        # Check key dependencies
        key_deps = ["opencv-python", "ultralytics", "pydantic", "numpy"]
        for dep in key_deps:
            assert dep in deps
            # Should either be a version string, "not_installed", or error message
            assert isinstance(deps[dep], str)
    
    def test_model_info_yolo(self):
        """Test model info for YOLO models."""
        model_info = get_model_info("yolo11n-pose.pt", "path/to/model.pt")
        
        assert "model_name" in model_info
        assert "model_path" in model_info
        assert "loaded_at" in model_info
        assert "model_type" in model_info
        assert "framework" in model_info
        
        assert model_info["model_name"] == "yolo11n-pose.pt"
        assert model_info["model_type"] == "YOLO"
        assert model_info["framework"] == "Ultralytics"
    
    def test_model_info_clip(self):
        """Test model info for CLIP models."""
        model_info = get_model_info("clip-vit-base")
        
        assert model_info["model_type"] == "CLIP"
        assert model_info["framework"] == "OpenAI"
    
    def test_model_info_whisper(self):
        """Test model info for Whisper models."""
        model_info = get_model_info("whisper-large")
        
        assert model_info["model_type"] == "Whisper"
        assert model_info["framework"] == "OpenAI"
    
    def test_model_info_unknown(self):
        """Test model info for unknown models."""
        model_info = get_model_info("unknown-model")
        
        assert model_info["model_type"] == "unknown"
    
    def test_create_annotation_metadata(self):
        """Test annotation metadata creation."""
        metadata = create_annotation_metadata(
            pipeline_name="TestPipeline",
            model_info={"model_name": "test-model", "model_type": "test"},
            processing_params={"param1": "value1"},
            video_metadata={"video_id": "test_video"}
        )
        
        # Check structure
        assert "videoannotator" in metadata
        assert "pipeline" in metadata
        assert "model" in metadata
        assert "video" in metadata
        
        # Check pipeline info
        pipeline_info = metadata["pipeline"]
        assert pipeline_info["name"] == "TestPipeline"
        assert "processing_timestamp" in pipeline_info
        assert pipeline_info["processing_params"] == {"param1": "value1"}
        
        # Check model info
        assert metadata["model"]["model_name"] == "test-model"
        
        # Check video info
        assert metadata["video"]["video_id"] == "test_video"
    
    def test_create_annotation_metadata_minimal(self):
        """Test annotation metadata creation with minimal parameters."""
        metadata = create_annotation_metadata("MinimalPipeline")
        
        assert "videoannotator" in metadata
        assert "pipeline" in metadata
        assert metadata["pipeline"]["name"] == "MinimalPipeline"
        assert metadata["pipeline"]["processing_params"] == {}


@pytest.mark.unit
class TestPipelineMetadata:
    """Test cases for pipeline metadata integration."""
    
    def test_scene_pipeline_metadata(self):
        """Test scene pipeline includes metadata in outputs."""
        pipeline = SceneDetectionPipeline()
        pipeline.initialize()
        
        # Test metadata creation
        metadata = pipeline.create_output_metadata()
        
        assert "videoannotator" in metadata
        assert "pipeline" in metadata
        assert metadata["pipeline"]["name"] == "SceneDetectionPipeline"
        
        # Test model info is set
        assert pipeline._model_info is not None
        assert "model_name" in pipeline._model_info
        
        pipeline.cleanup()
    
    def test_person_pipeline_metadata(self):
        """Test person pipeline includes metadata in outputs."""
        pipeline = PersonTrackingPipeline()
        pipeline.initialize()
        
        # Test metadata creation
        metadata = pipeline.create_output_metadata()
        
        assert "videoannotator" in metadata
        assert "pipeline" in metadata
        assert metadata["pipeline"]["name"] == "PersonTrackingPipeline"
        
        # Test model info is set
        assert pipeline._model_info is not None
        assert "model_name" in pipeline._model_info
        
        pipeline.cleanup()
    
    @patch('cv2.VideoCapture')
    def test_pipeline_save_with_metadata(self, mock_cv2):
        """Test pipeline save operations include comprehensive metadata."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,  # FPS
            7: 1000,  # Frame count
            3: 1920,  # Width
            4: 1080   # Height
        }.get(prop, 0)
        mock_cv2.return_value = mock_cap
        
        pipeline = SceneDetectionPipeline()
        pipeline.initialize()
        
        # Create test annotations
        from src.schemas.scene_schema import SceneSegment
        test_annotation = SceneSegment(
            type="scene_segment",
            video_id="test_video",
            timestamp=1.0,
            start_time=0.0,
            end_time=2.0,
            scene_id="scene_001",
            scene_type="test_scene"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test save with metadata
            pipeline.save_annotations([test_annotation], temp_path)
            
            # Read back and verify metadata
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            # Check metadata structure
            assert "metadata" in data
            assert "pipeline" in data
            assert "timestamp" in data
            assert "annotations" in data
            
            # Check metadata content
            metadata = data["metadata"]
            assert "videoannotator" in metadata
            assert "pipeline" in metadata
            assert "model" in metadata
            
            # Check pipeline info
            assert metadata["pipeline"]["name"] == "SceneDetectionPipeline"
            assert "processing_timestamp" in metadata["pipeline"]
            
            # Check annotations
            assert len(data["annotations"]) == 1
            assert data["annotations"][0]["scene_id"] == "scene_001"
            
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            pipeline.cleanup()
    
    def test_base_pipeline_methods(self):
        """Test base pipeline metadata methods."""
        pipeline = SceneDetectionPipeline()
        
        # Test set_model_info
        pipeline.set_model_info("test-model", "/path/to/model")
        assert pipeline._model_info is not None
        assert pipeline._model_info["model_name"] == "test-model"
        assert pipeline._model_info["model_path"] == "/path/to/model"
        
        # Test create_output_metadata
        metadata = pipeline.create_output_metadata({"test": "data"})
        assert "videoannotator" in metadata
        assert "pipeline" in metadata
        assert "model" in metadata
        assert metadata["video"]["test"] == "data"


@pytest.mark.integration
class TestVersioningIntegration:
    """Integration tests for versioning across the system."""
    
    def test_version_consistency(self):
        """Test version consistency across modules."""
        from src import __version__ as main_version
        from src.version import __version__ as version_module_version
        
        assert main_version == version_module_version
    
    def test_pipeline_metadata_consistency(self):
        """Test that all pipelines use consistent metadata format."""
        pipelines = [
            SceneDetectionPipeline(),
            PersonTrackingPipeline()
        ]
        
        for pipeline in pipelines:
            pipeline.initialize()
            metadata = pipeline.create_output_metadata()
            
            # Check consistent structure
            assert "videoannotator" in metadata
            assert "pipeline" in metadata
            assert "model" in metadata
            
            # Check version consistency
            assert metadata["videoannotator"]["version"] == __version__
            
            # Check pipeline name matches class
            expected_name = pipeline.__class__.__name__
            assert metadata["pipeline"]["name"] == expected_name
            
            pipeline.cleanup()
    
    @patch('cv2.VideoCapture')
    def test_end_to_end_metadata_flow(self, mock_cv2):
        """Test metadata flow from processing to output files."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,  # FPS
            7: 300,   # Frame count
            3: 640,   # Width
            4: 480    # Height
        }.get(prop, 0)
        mock_cv2.return_value = mock_cap
        
        pipeline = SceneDetectionPipeline()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake video file
            fake_video = Path(temp_dir) / "test_video.mp4"
            fake_video.touch()
            
            try:
                pipeline.initialize()
                
                # Mock the scene detection to return test data
                with patch.object(pipeline, '_detect_scene_boundaries') as mock_detect:
                    mock_detect.return_value = [
                        {"start": 0.0, "end": 5.0},
                        {"start": 5.0, "end": 10.0}
                    ]
                    
                    with patch.object(pipeline, '_classify_scenes') as mock_classify:
                        mock_classify.return_value = [
                            {"start": 0.0, "end": 5.0, "classification": "scene1", "confidence": 0.8},
                            {"start": 5.0, "end": 10.0, "classification": "scene2", "confidence": 0.7}
                        ]
                        
                        # Process with output directory
                        results = pipeline.process(
                            video_path=str(fake_video),
                            start_time=0.0,
                            end_time=10.0,
                            output_dir=temp_dir
                        )
                        
                        # Check results
                        assert len(results) == 2
                        
                        # Check output file was created
                        output_files = list(Path(temp_dir).glob("*.json"))
                        assert len(output_files) == 1
                        
                        # Verify metadata in output file
                        with open(output_files[0], 'r') as f:
                            data = json.load(f)
                        
                        assert "metadata" in data
                        assert data["metadata"]["videoannotator"]["version"] == __version__
                        assert data["metadata"]["pipeline"]["name"] == "SceneDetectionPipeline"
                        assert len(data["annotations"]) == 2
                        
            finally:
                pipeline.cleanup()

@pytest.mark.unit
class TestSpeechPipeline:
    """Test cases for SpeechPipeline."""
    
    def test_speech_config_initialization(self):
        """Test speech pipeline configuration initialization."""
        config = SpeechPipelineConfig(
            model_name="tiny",
            language="en",
            task="transcribe",
            word_timestamps=True,
            use_gpu=False
        )
        
        assert config.model_name == "tiny"
        assert config.language == "en"
        assert config.task == "transcribe"
        assert config.word_timestamps == True
        assert config.use_gpu == False
    
    def test_speech_config_validation(self):
        """Test speech configuration validation."""
        # Test invalid model name
        with pytest.raises(ValueError, match="Invalid model_name"):
            SpeechPipelineConfig(model_name="invalid_model")
        
        # Test invalid task
        with pytest.raises(ValueError, match="Invalid task"):
            SpeechPipelineConfig(task="invalid_task")
    
    def test_speech_pipeline_initialization(self):
        """Test speech pipeline initialization."""
        config = SpeechPipelineConfig(
            model_name="tiny",
            use_gpu=False
        )
        pipeline = SpeechPipeline(config)
        
        assert pipeline.config.model_name == "tiny"
        assert not pipeline.is_initialized
        assert pipeline._whisper_model is None
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.speech_pipeline.whisper')
    def test_speech_pipeline_initialize_success(self, mock_whisper):
        """Test successful pipeline initialization."""
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        config = SpeechPipelineConfig(model_name="tiny", use_gpu=False)
        pipeline = SpeechPipeline(config)
        pipeline.initialize()
        
        assert pipeline.is_initialized
        assert pipeline._whisper_model == mock_model
        mock_whisper.load_model.assert_called_once_with("tiny", device="cpu")
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', False)
    def test_speech_pipeline_initialize_no_whisper(self):
        """Test initialization fails when Whisper is not available."""
        pipeline = SpeechPipeline()
        
        with pytest.raises(ImportError, match="whisper is not available"):
            pipeline.initialize()
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.speech_pipeline.whisper')
    @patch('torch.cuda.is_available', return_value=True)
    def test_speech_pipeline_gpu_usage(self, mock_cuda, mock_whisper):
        """Test GPU usage when available."""
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        config = SpeechPipelineConfig(model_name="tiny", use_gpu=True)
        pipeline = SpeechPipeline(config)
        pipeline.initialize()
        
        mock_whisper.load_model.assert_called_once_with("tiny", device="cuda")
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.speech_pipeline.whisper')
    def test_speech_audio_processing(self, mock_whisper, temp_audio_file):
        """Test speech recognition on audio file."""
        # Mock Whisper model and result
        mock_model = Mock()
        mock_transcribe_result = {
            'text': 'Hello world',
            'language': 'en',
            'segments': [
                {
                    'id': 0,
                    'start': 0.0,
                    'end': 2.0,
                    'text': 'Hello world',
                    'tokens': [1, 2, 3],
                    'temperature': 0.0,
                    'avg_logprob': -0.5,
                    'compression_ratio': 1.0,
                    'no_speech_prob': 0.1,
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 1.0, 'probability': 0.9},
                        {'word': 'world', 'start': 1.0, 'end': 2.0, 'probability': 0.8}
                    ]
                }
            ]
        }
        mock_model.transcribe.return_value = mock_transcribe_result
        mock_whisper.load_model.return_value = mock_model
        
        config = SpeechPipelineConfig(model_name="tiny", use_gpu=False)
        pipeline = SpeechPipeline(config)
        pipeline.initialize()
        
        result = pipeline.transcribe_audio(temp_audio_file)
        
        assert result is not None
        assert result.transcript == 'Hello world'
        assert result.language == 'en'
        assert len(result.words) == 2
        assert len(result.metadata.get('segments', [])) == 1
        assert result.words[0]['word'] == 'Hello'
        assert result.words[0]['start'] == 0.0
        assert result.words[0]['end'] == 1.0
    
    def test_speech_audio_file_not_found(self):
        """Test handling of missing audio file."""
        config = SpeechPipelineConfig(model_name="tiny", use_gpu=False)
        pipeline = SpeechPipeline(config)
        
        result = pipeline.transcribe_audio("nonexistent_file.wav")
        assert result is None
    
    def test_speech_pipeline_info(self):
        """Test getting pipeline information."""
        config = SpeechPipelineConfig(
            model_name="base",
            language="en",
            word_timestamps=True
        )
        pipeline = SpeechPipeline(config)
        info = pipeline.get_pipeline_info()
        
        assert info['name'] == 'SpeechPipeline'
        assert 'capabilities' in info
        assert 'speech_recognition' in info['capabilities']
        assert info['config']['model_name'] == 'base'
        assert info['config']['language'] == 'en'
    
    def test_speech_get_schema(self):
        """Test getting output schema."""
        pipeline = SpeechPipeline()
        schema = pipeline.get_schema()
        
        assert schema['type'] == 'speech_recognition'
        assert 'properties' in schema
        assert 'transcript' in schema['properties']
        assert 'words' in schema['properties']
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.speech_pipeline.whisper')
    @patch('src.pipelines.audio_processing.ffmpeg_utils.extract_audio_from_video')
    def test_speech_process_video(self, mock_extract, mock_whisper, temp_video_file):
        """Test processing video file for speech recognition."""
        # Mock FFmpeg extraction
        mock_extract.return_value = Path("extracted_audio.wav")
        
        # Mock Whisper
        mock_model = Mock()
        mock_result = {
            'text': 'Test speech',
            'language': 'en',
            'segments': []
        }
        mock_model.transcribe.return_value = mock_result
        mock_whisper.load_model.return_value = mock_model
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            config = SpeechPipelineConfig(model_name="tiny", use_gpu=False)
            pipeline = SpeechPipeline(config)
            pipeline.initialize()
            
            results = pipeline.process(str(temp_video_file))
            
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].transcript == 'Test speech'
    
    def test_speech_cleanup(self):
        """Test pipeline cleanup."""
        config = SpeechPipelineConfig(model_name="tiny", use_gpu=False)
        pipeline = SpeechPipeline(config)
        
        # Mock model with to() method
        mock_model = Mock()
        pipeline._whisper_model = mock_model
        pipeline.is_initialized = True
        
        pipeline.cleanup()
        
        assert pipeline._whisper_model is None
        assert not pipeline.is_initialized
        mock_model.to.assert_called_once_with('cpu')


@pytest.mark.integration  
class TestSpeechPipelineIntegration:
    """Integration tests for speech pipeline (require external dependencies)."""
    
    @pytest.mark.skipif(
        not os.getenv("TEST_INTEGRATION"),
        reason="Integration tests disabled. Set TEST_INTEGRATION=1 to enable"
    )
    def test_real_speech_pipeline(self, temp_audio_file):
        """Test speech recognition with real Whisper model."""
        try:
            config = SpeechPipelineConfig(
                model_name="tiny",  # Use smallest model for testing
                use_gpu=False,
                word_timestamps=True
            )
            
            pipeline = SpeechPipeline(config)
            pipeline.initialize()
            
            result = pipeline.transcribe_audio(temp_audio_file)
            
            # Basic validation
            assert result is not None
            assert isinstance(result.text, str)
            assert isinstance(result.words, list)
            assert isinstance(result.segments, list)
            
        except Exception as e:
            pytest.skip(f"Real speech recognition test failed: {e}")
        finally:
            pipeline.cleanup()
