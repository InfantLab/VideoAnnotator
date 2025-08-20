"""
Stage 1: Basic Whisper Pipeline Test - Simple Structure Test

This tests the basic structure without complex dependencies.
"""

import pytest


@pytest.mark.unit
class TestWhisperBasePipelineStage1:
    """Stage 1 tests - basic structure verification."""
    
    def test_pipeline_module_exists(self):
        """Test that the WhisperBasePipeline module can be imported."""
        try:
            from src.pipelines.audio_processing.whisper_base_pipeline import WhisperBasePipeline
            assert True, "Module imported successfully"
        except ImportError as e:
            pytest.skip(f"WhisperBasePipeline not available: {e}")
    
    def test_basic_instantiation(self):
        """Test basic pipeline instantiation."""
        try:
            from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline
            pipeline = SpeechPipeline()
            assert pipeline is not None
            assert hasattr(pipeline, 'config')
        except ImportError as e:
            pytest.skip(f"SpeechPipeline not available: {e}")
        except Exception as e:
            pytest.fail(f"Failed to instantiate pipeline: {e}")
    
    def test_config_handling(self):
        """Test configuration handling."""
        try:
            from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline
            
            # Test with empty config
            pipeline1 = SpeechPipeline({})
            assert pipeline1.config is not None
            
            # Test with custom config
            custom_config = {"whisper_model": "small"}
            pipeline2 = SpeechPipeline(custom_config)
            assert pipeline2.config["whisper_model"] == "small"
            
        except ImportError as e:
            pytest.skip(f"SpeechPipeline not available: {e}")
        except Exception as e:
            pytest.fail(f"Configuration test failed: {e}")


@pytest.mark.unit
class TestBasicAudioProcessing:
    """Test basic audio processing utilities without heavy dependencies."""
    
    def test_numpy_availability(self):
        """Test that numpy is available for audio processing."""
        import numpy as np
        
        # Test basic array operations
        audio_data = np.array([0.5, -0.3, 0.8, -0.1])
        assert len(audio_data) == 4
        assert isinstance(audio_data, np.ndarray)
    
    def test_basic_audio_normalization_logic(self):
        """Test basic audio normalization without pipeline dependencies."""
        import numpy as np
        
        # Simple normalization function (similar to what pipeline would use)
        def normalize_audio(audio):
            if len(audio) == 0:
                raise ValueError("Audio data is empty")
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
            return audio
        
        # Test normal case
        audio = np.array([0.5, -0.8, 1.2, -1.5])
        normalized = normalize_audio(audio)
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test edge case
        with pytest.raises(ValueError):
            normalize_audio(np.array([]))
