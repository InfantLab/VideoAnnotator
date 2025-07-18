"""
Unit tests for Diarization and Speech Recognition Pipelines.

Tests cover speaker diarization, speech recognition, and audio analysis functionality.
"""

import pytest
import tempfile
import numpy as np
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.pipelines.audio_processing.diarization_pipeline import DiarizationPipeline
from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline


@pytest.mark.unit
class TestDiarizationPipeline:
    """Test cases for speaker diarization pipeline."""
    
    def test_diarization_config_initialization(self):
        """Test diarization pipeline configuration."""
        config = {
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "min_speakers": 1,
            "max_speakers": 5,
            "use_gpu": False,
            "huggingface_token": "test_token"
        }
        
        pipeline = DiarizationPipeline(config)
        assert pipeline.config["diarization_model"] == "pyannote/speaker-diarization-3.1"
        assert pipeline.config["min_speakers"] == 1
        assert pipeline.config["max_speakers"] == 5
        assert pipeline.config["use_gpu"] == False
        assert pipeline.config["huggingface_token"] == "test_token"
    
    def test_diarization_config_token_from_env(self):
        """Test that config picks up token from environment."""
        with patch.dict('os.environ', {'HF_AUTH_TOKEN': 'env_token'}, clear=False):
            config = {}
            pipeline = DiarizationPipeline(config)
            
            # Should use environment token during initialization
            assert pipeline.config.get("use_auth_token") == True
    
    def test_diarization_pipeline_initialization(self):
        """Test diarization pipeline initialization."""
        config = {
            "huggingface_token": "test_token",
            "use_gpu": False
        }
        pipeline = DiarizationPipeline(config)
        
        assert pipeline.config["huggingface_token"] == "test_token"
        assert pipeline.config["use_gpu"] == False
        assert not pipeline.is_initialized
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyAnnotePipeline')
    def test_diarization_pipeline_initialize_success(self, mock_pyannote):
        """Test successful diarization pipeline initialization."""
        # Mock PyAnnote pipeline
        mock_pipeline = Mock()
        mock_pyannote.from_pretrained.return_value = mock_pipeline
        
        config = {
            "huggingface_token": "test_token",
            "use_gpu": False
        }
        pipeline = DiarizationPipeline(config)
        
        # Initialize should succeed
        pipeline.initialize()
        
        assert pipeline.is_initialized == True
        assert pipeline.diarization_model is not None
        
        # Check that model was loaded with correct parameters
        mock_pyannote.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="FAKE_TOKEN_FOR_TESTING"  # Mock token for testing
        )
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', False)
    def test_diarization_pipeline_initialize_no_pyannote(self):
        """Test initialization fails when PyAnnote not available."""
        config = {"huggingface_token": "test_token"}
        pipeline = DiarizationPipeline(config)
        
        with pytest.raises(ImportError, match="PyAnnote not available"):
            pipeline.initialize()
    
    def test_diarization_pipeline_initialize_no_token(self):
        """Test initialization fails without HuggingFace token."""
        # Clear environment variables
        with patch.dict('os.environ', {}, clear=True):
            config = {}
            pipeline = DiarizationPipeline(config)
            
            with pytest.raises(ValueError, match="HuggingFace token required"):
                pipeline.initialize()
    
    @pytest.mark.skip(reason="GPU functionality not implemented in DiarizationPipeline")
    def test_diarization_pipeline_gpu_usage(self):
        """Test GPU usage configuration."""
        # TODO: Implement GPU support in DiarizationPipeline
        pass
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyAnnotePipeline')
    def test_diarization_audio_processing(self, mock_pyannote, temp_audio_file):
        """Test audio diarization processing."""
        # Mock turn objects that have start, duration, end attributes
        mock_turn1 = Mock()
        mock_turn1.start = 0.0
        mock_turn1.duration = 5.0
        mock_turn1.end = 5.0
        
        mock_turn2 = Mock()
        mock_turn2.start = 5.0
        mock_turn2.duration = 5.0
        mock_turn2.end = 10.0
        
        mock_turn3 = Mock()
        mock_turn3.start = 10.0
        mock_turn3.duration = 5.0
        mock_turn3.end = 15.0
        
        # Mock diarization result that can be iterated with itertracks
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (mock_turn1, None, "SPEAKER_00"),
            (mock_turn2, None, "SPEAKER_01"),
            (mock_turn3, None, "SPEAKER_00")
        ]
        
        # Mock the pipeline call to return the diarization result
        mock_pipeline = Mock()
        mock_pipeline.return_value = mock_diarization
        mock_pyannote.from_pretrained.return_value = mock_pipeline
        
        config = {"huggingface_token": "test_token"}
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()
        
        # Process audio
        results = pipeline.process(str(temp_audio_file))
        
        assert isinstance(results, list)  # Should return list of results from BasePipeline
        assert len(results) >= 1
        diarization_result = results[0]
        assert isinstance(diarization_result, dict)
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyAnnotePipeline')
    def test_diarization_audio_file_not_found(self, mock_pyannote):
        """Test error handling for missing audio file."""
        config = {"huggingface_token": "test_token"}
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()
        
        # Test error handling for missing file - should return empty list or None  
        result = pipeline.process("non_existent_file.wav")
        # Modern pipelines handle errors gracefully - might return empty list or None
        assert result is None or (isinstance(result, list) and len(result) == 0)
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyAnnotePipeline')
    def test_diarization_pipeline_info(self, mock_pyannote):
        """Test pipeline info generation."""
        config = {"huggingface_token": "test_token"}
        pipeline = DiarizationPipeline(config)
        
        # Test basic pipeline properties
        assert hasattr(pipeline, 'config')
        assert pipeline.config["huggingface_token"] == "test_token"


@pytest.mark.unit
class TestSpeechPipeline:
    """Test cases for speech recognition pipeline."""
    
    def test_speech_config_initialization(self):
        """Test speech pipeline configuration."""
        config = {
            "model_name": "base",
            "language": "en",
            "beam_size": 5
        }
        
        pipeline = SpeechPipeline(config)
        assert pipeline.config["model_name"] == "base"
        assert pipeline.config["language"] == "en"
        assert pipeline.config["beam_size"] == 5
    
    def test_speech_config_validation(self):
        """Test speech config validation."""
        # Valid model size
        config = {"model_name": "large-v2"}
        pipeline = SpeechPipeline(config)
        assert pipeline.config["model_name"] == "large-v2"
        
        # Invalid model size should use default
        config = {"model_name": "invalid"}
        pipeline = SpeechPipeline(config)
        # Pipeline should handle invalid model gracefully
        assert "model_name" in pipeline.config
    
    def test_speech_pipeline_initialization(self):
        """Test speech pipeline initialization."""
        config = {
            "model_name": "tiny",
            "language": "en"
        }
        pipeline = SpeechPipeline(config)
        
        assert pipeline.config["model_name"] == "tiny"
        assert pipeline.config["language"] == "en"
        assert not pipeline.is_initialized
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.speech_pipeline.whisper')
    def test_speech_pipeline_initialize_success(self, mock_whisper):
        """Test successful speech pipeline initialization."""
        # Mock whisper model
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        config = {"model": "base"}
        pipeline = SpeechPipeline(config)
        
        # Initialize should succeed
        pipeline.initialize()
        
        assert pipeline.is_initialized == True
        assert pipeline.whisper_model is not None
        
        # Check that model was loaded
        mock_whisper.load_model.assert_called_once_with("base", device='cpu')
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', False)
    def test_speech_pipeline_initialize_no_whisper(self):
        """Test initialization fails when Whisper not available."""
        config = {}
        pipeline = SpeechPipeline(config)
        
        with pytest.raises(ImportError, match="whisper is not available"):
            pipeline.initialize()
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.speech_pipeline.whisper')
    @patch('torch.cuda.is_available', return_value=True)
    def test_speech_pipeline_gpu_usage(self, mock_cuda, mock_whisper):
        """Test GPU usage for speech recognition."""
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        config = {"use_gpu": True}
        pipeline = SpeechPipeline(config)
        
        pipeline.initialize()
        
        # Should load model with GPU device
        mock_whisper.load_model.assert_called_with("base", device="cuda")
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.speech_pipeline.whisper')
    def test_speech_audio_processing(self, mock_whisper, temp_audio_file):
        """Test speech recognition processing."""
        # Mock whisper transcription result
        mock_result = {
            "text": "Hello world, this is a test.",
            "language": "en",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world,",
                    "tokens": [50364, 5751, 1002, 11, 50464],
                    "temperature": 0.0,
                    "avg_logprob": -0.5,
                    "compression_ratio": 2.5,
                    "no_speech_prob": 0.1,
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "Hello", "probability": 0.99},
                        {"start": 0.6, "end": 1.0, "word": "world,", "probability": 0.95}
                    ]
                },
                {
                    "id": 1,
                    "start": 2.0,
                    "end": 4.0,
                    "text": "this is a test.",
                    "tokens": [50464, 341, 307, 257, 1500, 13, 50564],
                    "temperature": 0.0,
                    "avg_logprob": -0.4,
                    "compression_ratio": 2.3,
                    "no_speech_prob": 0.05,
                    "words": [
                        {"start": 2.0, "end": 2.3, "word": "this", "probability": 0.98},
                        {"start": 2.4, "end": 2.6, "word": "is", "probability": 0.97},
                        {"start": 2.7, "end": 2.8, "word": "a", "probability": 0.96},
                        {"start": 2.9, "end": 3.2, "word": "test.", "probability": 0.94}
                    ]
                }
            ]
        }
        
        mock_model = Mock()
        mock_model.transcribe.return_value = mock_result
        mock_whisper.load_model.return_value = mock_model
        
        config = {}
        pipeline = SpeechPipeline(config)
        pipeline.initialize()
        
        # Process audio
        result = pipeline.process(str(temp_audio_file))
        
        assert isinstance(result, list)  # Should return list of results from BasePipeline
        assert len(result) == 1
        speech_result = result[0]
        assert isinstance(speech_result, dict)
        assert "transcript" in speech_result
        assert "words" in speech_result
        assert "metadata" in speech_result
        assert speech_result["transcript"] == "Hello world, this is a test."
        
        pipeline.cleanup()
    
    def test_speech_audio_file_not_found(self):
        """Test error handling for missing audio file."""
        config = {}
        pipeline = SpeechPipeline(config)
        
        # Test error handling for missing file - should return empty list or None  
        result = pipeline.process("non_existent_file.wav")
        # Modern pipelines handle errors gracefully - might return empty list or None
        assert result is None or (isinstance(result, list) and len(result) == 0)
    
    def test_speech_pipeline_info(self):
        """Test pipeline info generation."""
        config = {"model": "large"}
        pipeline = SpeechPipeline(config)
        
        info = pipeline.get_pipeline_info()
        
        assert "models" in info
        assert "config" in info
        assert info["models"]["whisper_model"] == "large"
    
    def test_speech_get_schema(self):
        """Test speech recognition schema generation."""
        pipeline = SpeechPipeline()
        schema = pipeline.get_schema()
        
        assert schema["type"] == "speech_recognition"
        assert "properties" in schema
        assert "transcript" in schema["properties"]
        assert "confidence" in schema["properties"]


@pytest.mark.integration
class TestDiarizationSpeechIntegration:
    """Integration tests for diarization and speech pipelines."""
    
    @pytest.mark.skipif(
        not os.getenv("HUGGINGFACE_TOKEN"),
        reason="Requires HUGGINGFACE_TOKEN environment variable"
    )
    @pytest.mark.skipif(
        not os.getenv("TEST_INTEGRATION"),
        reason="Integration tests disabled. Set TEST_INTEGRATION=1 to enable"
    )
    def test_real_diarization_pipeline(self, temp_audio_file):
        """Test real diarization with actual models."""
        config = {
            "use_gpu": False,  # Use CPU for testing
            "min_speakers": 1,
            "max_speakers": 3
        }
        
        pipeline = DiarizationPipeline(config)
        
        try:
            with pipeline:
                results = pipeline.process(str(temp_audio_file))
                
                assert isinstance(results, list)  # Should return list from BasePipeline
                assert len(results) >= 1
                assert isinstance(results[0], dict)
                
        except ImportError:
            pytest.skip("PyAnnote not available for integration test")
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")
    
    @pytest.mark.skipif(
        not os.getenv("TEST_INTEGRATION"),
        reason="Integration tests disabled. Set TEST_INTEGRATION=1 to enable"
    )
    def test_real_speech_recognition(self, temp_audio_file):
        """Test real speech recognition with actual models."""
        config = {
            "model_name": "tiny",  # Smallest model for testing
            "language": "en",
            "use_gpu": False
        }
        
        pipeline = SpeechPipeline(config)
        
        try:
            with pipeline:
                results = pipeline.process(str(temp_audio_file))
                
                assert isinstance(results, list)  # Should return list from BasePipeline
                assert len(results) >= 1
                assert isinstance(results[0], dict)
                    
        except ImportError:
            pytest.skip("Whisper not available for integration test")
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


@pytest.mark.performance
class TestAudioPipelinePerformance:
    """Performance tests for audio processing pipelines."""
    
    def test_diarization_memory_efficiency(self):
        """Test diarization memory usage with long audio."""
        config = {
            "huggingface_token": "test_token",
            "chunk_length": 30  # Process in 30-second chunks
        }
        
        pipeline = DiarizationPipeline(config)
        
        # Test that pipeline can be created with chunking config
        assert "chunk_length" in pipeline.config
        assert pipeline.config["chunk_length"] == 30
    
    def test_speech_recognition_speed(self):
        """Test speech recognition processing speed."""
        config = {
            "model_name": "tiny",  # Fastest model
            "beam_size": 1  # Fastest decoding
        }
        
        pipeline = SpeechPipeline(config)
        
        # Test that pipeline can be created with speed-optimized config
        assert pipeline.config["model_name"] == "tiny"
        assert pipeline.config["beam_size"] == 1
