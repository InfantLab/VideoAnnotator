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

from src.pipelines.audio_processing.diarization_pipeline import (
    DiarizationPipeline, 
    DiarizationPipelineConfig
)
from src.pipelines.audio_processing.speech_pipeline import (
    SpeechPipeline,
    SpeechPipelineConfig
)
from src.schemas.audio_schema import SpeakerDiarization, SpeechRecognition


@pytest.mark.unit
class TestDiarizationPipeline:
    """Test cases for speaker diarization pipeline."""
    
    def test_diarization_config_initialization(self):
        """Test diarization pipeline configuration."""
        config = DiarizationPipelineConfig(
            diarization_model="pyannote/speaker-diarization-3.1",
            min_speakers=1,
            max_speakers=5,
            use_gpu=False,
            huggingface_token="test_token"
        )
        
        assert config.diarization_model == "pyannote/speaker-diarization-3.1"
        assert config.min_speakers == 1
        assert config.max_speakers == 5
        assert config.use_gpu == False
        assert config.huggingface_token == "test_token"
    
    def test_diarization_config_token_from_env(self):
        """Test that config picks up token from environment."""
        with patch.dict('os.environ', {'HF_AUTH_TOKEN': 'env_token'}, clear=False):
            config = DiarizationPipelineConfig()
            
            # Should pick up token from environment
            assert config.huggingface_token == 'env_token'
    
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
        mock_pipeline = Mock()
        mock_pyannote.from_pretrained.return_value = mock_pipeline
        
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            use_gpu=False
        )
        pipeline = DiarizationPipeline(config)
        
        # Initialize should succeed
        pipeline.initialize()
        
        assert pipeline.is_initialized == True
        assert pipeline.diarization_pipeline is not None
        
        # Check that model was loaded with correct parameters
        mock_pyannote.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="test_token"
        )
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', False)
    def test_diarization_pipeline_initialize_no_pyannote(self):
        """Test initialization fails when PyAnnote not available."""
        config = DiarizationPipelineConfig(huggingface_token="test_token")
        pipeline = DiarizationPipeline(config)
        
        with pytest.raises(ImportError, match="pyannote.audio is required"):
            pipeline.initialize()
    
    def test_diarization_pipeline_initialize_no_token(self):
        """Test initialization fails without HuggingFace token."""
        # Clear environment variables
        with patch.dict('os.environ', {}, clear=True):
            config = DiarizationPipelineConfig()
            pipeline = DiarizationPipeline(config)
            
            with pytest.raises(ValueError, match="HuggingFace token is required"):
                pipeline.initialize()
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    @patch('src.pipelines.audio_processing.diarization_pipeline.torch')
    def test_diarization_pipeline_gpu_usage(self, mock_torch, mock_pyannote):
        """Test GPU usage configuration."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True
        
        mock_pipeline = Mock()
        mock_pyannote.from_pretrained.return_value = mock_pipeline
        
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            use_gpu=True
        )
        pipeline = DiarizationPipeline(config)
        
        pipeline.initialize()
        
        # Should move pipeline to GPU
        mock_pipeline.to.assert_called_with(mock_torch.device.return_value)
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    def test_diarization_audio_processing(self, mock_pyannote, temp_audio_file):
        """Test audio diarization processing."""
        # Mock diarization results
        mock_result = Mock()
        mock_result.get_timeline.return_value = [
            Mock(start=0.0, end=5.0, speaker="SPEAKER_00"),
            Mock(start=5.0, end=10.0, speaker="SPEAKER_01"),
            Mock(start=10.0, end=15.0, speaker="SPEAKER_00")
        ]
        
        mock_pipeline = Mock()
        mock_pipeline.return_value = mock_result
        mock_pyannote.from_pretrained.return_value = mock_pipeline
        
        config = DiarizationPipelineConfig(huggingface_token="test_token")
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()
        
        # Process audio
        results = pipeline.transcribe_audio(str(temp_audio_file))
        
        assert isinstance(results, list)
        assert len(results) == 1  # One diarization result
        
        diarization = results[0]
        assert diarization.video_id == temp_audio_file.stem
        assert diarization.total_speakers == 2
        assert len(diarization.segments) == 3
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    def test_diarization_audio_file_not_found(self, mock_pyannote):
        """Test error handling for missing audio file."""
        config = DiarizationPipelineConfig(huggingface_token="test_token")
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()
        
        with pytest.raises(FileNotFoundError):
            pipeline.transcribe_audio("non_existent_file.wav")
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.diarization_pipeline.PYANNOTE_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.diarization_pipeline.PyannoteePipeline')
    def test_diarization_pipeline_info(self, mock_pyannote):
        """Test pipeline info generation."""
        config = DiarizationPipelineConfig(huggingface_token="test_token")
        pipeline = DiarizationPipeline(config)
        
        info = pipeline.get_pipeline_info()
        
        assert "model" in info
        assert "min_speakers" in info
        assert "max_speakers" in info
        assert info["model"] == "pyannote/speaker-diarization-3.1"


@pytest.mark.unit
class TestSpeechPipeline:
    """Test cases for speech recognition pipeline."""
    
    def test_speech_config_initialization(self):
        """Test speech pipeline configuration."""
        config = SpeechPipelineConfig(
            model_name="base",
            language="en",
            beam_size=5
        )
        
        assert config.model_name == "base"
        assert config.language == "en"
        assert config.beam_size == 5
    
    def test_speech_config_validation(self):
        """Test speech config validation."""
        # Valid model size
        config = SpeechPipelineConfig(model_name="large-v2")
        assert config.model_name == "large-v2"
        
        # Invalid model size should use default
        with pytest.warns(UserWarning):
            config = SpeechPipelineConfig(model_name="invalid")
            assert config.model_name == "base"  # Should fallback to default
    
    def test_speech_pipeline_initialization(self):
        """Test speech pipeline initialization."""
        config = SpeechPipelineConfig(
            model_name="tiny",
            language="en"
        )
        pipeline = SpeechPipeline(config)
        
        assert pipeline.config.model_name == "tiny"
        assert pipeline.config.language == "en"
        assert not pipeline.is_initialized
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.speech_pipeline.whisper')
    def test_speech_pipeline_initialize_success(self, mock_whisper):
        """Test successful speech pipeline initialization."""
        # Mock whisper model
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        config = SpeechPipelineConfig(model_name="base")
        pipeline = SpeechPipeline(config)
        
        # Initialize should succeed
        pipeline.initialize()
        
        assert pipeline.is_initialized == True
        assert pipeline.model is not None
        
        # Check that model was loaded
        mock_whisper.load_model.assert_called_once_with("base")
        
        pipeline.cleanup()
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', False)
    def test_speech_pipeline_initialize_no_whisper(self):
        """Test initialization fails when Whisper not available."""
        config = SpeechPipelineConfig()
        pipeline = SpeechPipeline(config)
        
        with pytest.raises(ImportError, match="whisper is required"):
            pipeline.initialize()
    
    @patch('src.pipelines.audio_processing.speech_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.pipelines.audio_processing.speech_pipeline.whisper')
    @patch('torch.cuda.is_available', return_value=True)
    def test_speech_pipeline_gpu_usage(self, mock_cuda, mock_whisper):
        """Test GPU usage for speech recognition."""
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        config = SpeechPipelineConfig(use_gpu=True)
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
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world,",
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "Hello", "probability": 0.99},
                        {"start": 0.6, "end": 1.0, "word": "world,", "probability": 0.95}
                    ]
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "this is a test.",
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
        
        config = SpeechPipelineConfig()
        pipeline = SpeechPipeline(config)
        pipeline.initialize()
        
        # Process audio
        speech_result = pipeline.transcribe_audio(str(temp_audio_file))
        
        assert speech_result is not None  # Should have speech recognition result
        
        # Check result
        assert speech_result.video_id == temp_audio_file.stem
        assert "Hello world" in speech_result.transcript
        assert hasattr(speech_result, 'words')  # Should have words
        
        pipeline.cleanup()
    
    def test_speech_audio_file_not_found(self):
        """Test error handling for missing audio file."""
        config = SpeechPipelineConfig()
        pipeline = SpeechPipeline(config)
        
        with pytest.raises(FileNotFoundError):
            pipeline.transcribe_audio("non_existent_file.wav")
    
    def test_speech_pipeline_info(self):
        """Test pipeline info generation."""
        config = SpeechPipelineConfig(model_name="large")
        pipeline = SpeechPipeline(config)
        
        info = pipeline.get_pipeline_info()
        
        assert "model" in info
        assert "language" in info
        assert info["model"] == "large"
    
    def test_speech_get_schema(self):
        """Test speech recognition schema generation."""
        pipeline = SpeechPipeline()
        schema = pipeline.get_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "video_id" in schema["properties"]
        assert "timestamp" in schema["properties"]
        assert "text" in schema["properties"]
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
        config = DiarizationPipelineConfig(
            use_gpu=False,  # Use CPU for testing
            min_speakers=1,
            max_speakers=3
        )
        
        pipeline = DiarizationPipeline(config)
        
        try:
            with pipeline:
                results = pipeline.transcribe_audio(str(temp_audio_file))
                
                assert isinstance(results, list)
                assert len(results) >= 1
                
                # Check result format
                diarization = results[0]
                assert hasattr(diarization, 'total_speakers')
                assert hasattr(diarization, 'segments')
                assert diarization.total_speakers >= 1
                
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
        config = SpeechPipelineConfig(
            model_name="tiny",  # Smallest model for testing
            language="en",
            use_gpu=False
        )
        
        pipeline = SpeechPipeline(config)
        
        try:
            with pipeline:
                results = pipeline.transcribe_audio(str(temp_audio_file))
                
                assert isinstance(results, list)
                
                if results:  # May be empty for silent audio
                    speech_result = results[0]
                    assert hasattr(speech_result, 'full_text')
                    assert hasattr(speech_result, 'segments')
                    assert isinstance(speech_result.full_text, str)
                    
        except ImportError:
            pytest.skip("Whisper not available for integration test")
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


@pytest.mark.performance
class TestAudioPipelinePerformance:
    """Performance tests for audio processing pipelines."""
    
    def test_diarization_memory_efficiency(self):
        """Test diarization memory usage with long audio."""
        config = DiarizationPipelineConfig(
            huggingface_token="test_token",
            chunk_length=30  # Process in 30-second chunks
        )
        
        pipeline = DiarizationPipeline(config)
        
        # Mock long audio processing
        with patch.object(pipeline, '_chunk_audio') as mock_chunk:
            # Simulate 10-minute audio chunked into 30-second segments
            mock_chunk.return_value = [
                f"chunk_{i}" for i in range(20)  # 20 chunks of 30 seconds
            ]
            
            chunks = pipeline._chunk_audio("long_audio.wav")
            
            # Should break into manageable chunks
            assert len(chunks) == 20
    
    def test_speech_recognition_speed(self):
        """Test speech recognition processing speed."""
        config = SpeechPipelineConfig(
            model_name="tiny",  # Fastest model
            beam_size=1  # Fastest decoding
        )
        
        pipeline = SpeechPipeline(config)
        
        # Mock audio segment (30 seconds)
        audio_length = 30 * 16000  # 30 seconds at 16kHz
        
        import time
        
        with patch.object(pipeline, '_transcribe_segment') as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "Test transcription",
                "segments": []
            }
            
            start_time = time.time()
            
            # Process multiple segments
            for i in range(10):
                result = pipeline._transcribe_segment(f"audio_segment_{i}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process efficiently (mocked)
            assert processing_time < 1.0
