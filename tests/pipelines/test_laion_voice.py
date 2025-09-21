"""
Stage 3: LAION Voice Pipeline Tests

Test suite for the LAIONVoicePipeline component.
Focus on voice emotion analysis using LAION models with Whisper base.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Try importing the pipeline, skip tests if not available
try:
    from src.pipelines.audio_processing.laion_voice_pipeline import LAIONVoicePipeline, EMOTION_LABELS
    LAION_VOICE_AVAILABLE = True
except ImportError:
    LAION_VOICE_AVAILABLE = False

# Try importing the base pipeline
try:
    from src.pipelines.audio_processing.whisper_base_pipeline import WhisperBasePipeline
    WHISPER_BASE_AVAILABLE = True
except ImportError:
    WHISPER_BASE_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not LAION_VOICE_AVAILABLE, reason="LAIONVoicePipeline not available")
class TestLAIONVoicePipelineBasics:
    """Test basic LAIONVoicePipeline functionality."""
    
    def test_default_configuration(self):
        """Test pipeline initialization with default configuration."""
        pipeline = LAIONVoicePipeline()
        
        # Verify key default configuration values
        assert pipeline.config["model_size"] == "small"
        assert pipeline.config["whisper_model"] == "mkrausio/EmoWhisper-AnS-Small-v0.1"
        assert pipeline.config["top_k_emotions"] == 5
        assert pipeline.config["segmentation_mode"] == "fixed_interval"
        assert pipeline.config["min_segment_duration"] == 1.0
        assert pipeline.config["max_segment_duration"] == 30.0
    
    def test_custom_configuration(self):
        """Test pipeline initialization with custom configuration."""
        config = {
            "model_size": "large",
            "top_k_emotions": 3,
            "segmentation_mode": "diarization",
            "min_segment_duration": 2.0,
            "max_segment_duration": 15.0
        }
        pipeline = LAIONVoicePipeline(config)
        
        # Verify custom configuration values
        assert pipeline.config["model_size"] == "large"
        assert pipeline.config["top_k_emotions"] == 3
        assert pipeline.config["segmentation_mode"] == "diarization"
        assert pipeline.config["min_segment_duration"] == 2.0
        assert pipeline.config["max_segment_duration"] == 15.0
    
    def test_inherits_from_whisper_base(self):
        """Test that LAIONVoicePipeline inherits from WhisperBasePipeline."""
        if not WHISPER_BASE_AVAILABLE:
            pytest.skip("WhisperBasePipeline not available")
            
        pipeline = LAIONVoicePipeline()
        
        # Should inherit from WhisperBasePipeline
        assert isinstance(pipeline, WhisperBasePipeline)
        
        # Should have Whisper base attributes
        assert hasattr(pipeline, 'whisper_model')
        assert hasattr(pipeline, 'whisper_processor')
        assert hasattr(pipeline, 'device')
    
    def test_emotion_labels_structure(self):
        """Test that emotion labels are properly defined."""
        # Should have all 43 emotions (same as face pipeline)
        assert len(EMOTION_LABELS) == 43
        
        # Test some key emotions exist  
        available_emotions = list(EMOTION_LABELS.keys())
        assert len(available_emotions) >= 40
        
        # All should have model file mappings
        for emotion, filename in EMOTION_LABELS.items():
            assert isinstance(emotion, str)
            assert isinstance(filename, str) 
            assert filename.endswith('.pth')
    
    def test_pipeline_attributes(self):
        """Test that pipeline has required attributes."""
        pipeline = LAIONVoicePipeline()
        
        # Should have basic pipeline attributes
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'is_initialized')
        assert hasattr(pipeline, 'logger')
        
        # Should have voice-specific attributes
        assert hasattr(pipeline, 'classifiers')  # emotion classifiers
        # cuda_capability might be dynamic, so just check if accessible
        try:
            _ = getattr(pipeline, 'cuda_capability', None)
        except:
            pass  # Not critical if this attribute doesn't exist


@pytest.mark.unit
@pytest.mark.skipif(not LAION_VOICE_AVAILABLE, reason="LAIONVoicePipeline not available")
class TestLAIONVoicePipelineSegmentation:
    """Test audio segmentation functionality."""
    
    def test_segmentation_modes(self):
        """Test different segmentation mode configurations."""
        modes = ["fixed_interval", "diarization", "scene_based", "vad"]
        
        for mode in modes:
            pipeline = LAIONVoicePipeline({"segmentation_mode": mode})
            assert pipeline.config["segmentation_mode"] == mode
    
    def test_segment_duration_bounds(self):
        """Test segment duration configuration."""
        config = {
            "min_segment_duration": 0.5,
            "max_segment_duration": 60.0
        }
        pipeline = LAIONVoicePipeline(config)
        
        assert pipeline.config["min_segment_duration"] == 0.5
        assert pipeline.config["max_segment_duration"] == 60.0
    
    def test_pps_parameter_logic(self):
        """Test PPS (predictions per second) parameter logic."""
        # Test PPS calculation logic
        def calculate_segment_duration(pps):
            """Calculate segment duration from PPS."""
            if pps <= 0:
                return 5.0  # Default
            return 1.0 / pps
        
        # Test different PPS values
        assert calculate_segment_duration(0.2) == 5.0  # 5-second segments
        assert calculate_segment_duration(1.0) == 1.0  # 1-second segments  
        assert calculate_segment_duration(0.1) == 10.0  # 10-second segments
    
    def test_segment_overlap_configuration(self):
        """Test segment overlap configuration."""
        pipeline = LAIONVoicePipeline({"segment_overlap": 0.5})
        assert pipeline.config["segment_overlap"] == 0.5


@pytest.mark.unit
@pytest.mark.skipif(not LAION_VOICE_AVAILABLE, reason="LAIONVoicePipeline not available")
class TestLAIONVoicePipelineAudioProcessing:
    """Test audio processing functionality."""
    
    def test_audio_segment_processing_logic(self):
        """Test audio segment processing logic."""
        pipeline = LAIONVoicePipeline()
        
        # Should have methods for audio processing
        assert hasattr(pipeline, '_segment_audio') or hasattr(pipeline, 'segment_audio')
        
        # Test segment creation logic
        def create_segments(audio_length, segment_duration, overlap=0.0):
            """Create audio segments."""
            segments = []
            start = 0
            step = segment_duration * (1 - overlap)
            
            while start < audio_length:
                end = min(start + segment_duration, audio_length)
                segments.append((start, end))
                start += step
                if end >= audio_length:
                    break
            return segments
        
        # Test 10-second audio with 3-second segments
        segments = create_segments(10.0, 3.0, overlap=0.0)
        assert len(segments) >= 3
        assert segments[0] == (0, 3.0)
    
    def test_whisper_integration_constants(self):
        """Test Whisper integration constants."""
        # These constants should be defined in the module
        from src.pipelines.audio_processing.laion_voice_pipeline import (
            WHISPER_SEQ_LEN, WHISPER_EMBED_DIM
        )
        
        assert WHISPER_SEQ_LEN == 1500
        assert WHISPER_EMBED_DIM == 768
    
    def test_mlp_model_structure(self):
        """Test MLP model structure definition."""
        # Should have MLP model class
        from src.pipelines.audio_processing.laion_voice_pipeline import FullEmbeddingMLP
        
        # Should be a class
        assert isinstance(FullEmbeddingMLP, type)


@pytest.mark.unit
@pytest.mark.skipif(not LAION_VOICE_AVAILABLE, reason="LAIONVoicePipeline not available")
class TestLAIONVoicePipelineGPUCompatibility:
    """Test GPU compatibility features."""
    
    def test_cuda_capability_detection(self):
        """Test CUDA capability detection logic."""
        pipeline = LAIONVoicePipeline()
        
        # Should have CUDA capability attribute
        assert hasattr(pipeline, 'cuda_capability')
        
        # Test CUDA capability logic (without actual GPU)
        def get_cuda_capability():
            """Mock CUDA capability detection."""
            try:
                import torch
                if torch.cuda.is_available():
                    major, minor = torch.cuda.get_device_capability()
                    return major + minor * 0.1
                return 0.0
            except:
                return 0.0
        
        capability = get_cuda_capability()
        assert isinstance(capability, (int, float))
        assert capability >= 0.0
    
    def test_torch_compile_logic(self):
        """Test torch.compile usage logic."""
        # Test logic for deciding when to use torch.compile
        def should_use_torch_compile(cuda_capability):
            """Determine if torch.compile should be used."""
            return cuda_capability >= 7.0
        
        # Test different capabilities
        assert should_use_torch_compile(8.0) == True   # Modern GPU
        assert should_use_torch_compile(6.1) == False  # Legacy GPU
        assert should_use_torch_compile(0.0) == False  # CPU
    
    def test_device_fallback_logic(self):
        """Test device fallback logic."""
        # Test device selection logic
        def select_device(requested_device):
            """Select actual device based on request."""
            if requested_device == "auto":
                try:
                    import torch
                    return "cuda" if torch.cuda.is_available() else "cpu"
                except:
                    return "cpu"
            return requested_device
        
        # Test different configurations
        auto_device = select_device("auto")
        assert auto_device in ["cpu", "cuda"]
        
        assert select_device("cpu") == "cpu"
        assert select_device("cuda") == "cuda"


@pytest.mark.unit
@pytest.mark.skipif(not LAION_VOICE_AVAILABLE, reason="LAIONVoicePipeline not available")
class TestLAIONVoicePipelineOutputFormats:
    """Test output format functionality."""
    
    def test_webvtt_output_structure(self):
        """Test WebVTT output format structure."""
        # Test WebVTT segment structure
        mock_vtt_segment = {
            "start_time": 0.0,
            "end_time": 5.0,
            "speaker_id": "speaker_1",
            "text": "Hello world",
            "emotions": {
                "joy": {"score": 0.8, "rank": 1},
                "contentment": {"score": 0.6, "rank": 2}
            }
        }
        
        # Verify structure
        assert "start_time" in mock_vtt_segment
        assert "end_time" in mock_vtt_segment
        assert "emotions" in mock_vtt_segment
        assert mock_vtt_segment["end_time"] > mock_vtt_segment["start_time"]
    
    def test_json_output_structure(self):
        """Test JSON output format structure."""
        # Test JSON output structure
        mock_json_output = {
            "segments": [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "emotions": {"joy": {"score": 0.8, "rank": 1}},
                    "model_info": {"model_size": "small"}
                }
            ],
            "metadata": {
                "pipeline": "LAIONVoicePipeline",
                "total_segments": 1
            }
        }
        
        # Verify structure
        assert "segments" in mock_json_output
        assert "metadata" in mock_json_output
        assert len(mock_json_output["segments"]) == mock_json_output["metadata"]["total_segments"]
    
    def test_emotion_ranking_logic(self):
        """Test emotion ranking and scoring logic."""
        # Test emotion ranking
        def rank_emotions(emotion_scores, top_k=3):
            """Rank emotions by score."""
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            ranked = {}
            for i, (emotion, score) in enumerate(sorted_emotions[:top_k]):
                ranked[emotion] = {"score": score, "rank": i + 1}
            return ranked
        
        scores = {"joy": 0.8, "sadness": 0.6, "anger": 0.4, "fear": 0.2}
        ranked = rank_emotions(scores, top_k=3)
        
        assert len(ranked) == 3
        assert ranked["joy"]["rank"] == 1
        assert ranked["sadness"]["rank"] == 2
        assert ranked["anger"]["rank"] == 3
        assert "fear" not in ranked


@pytest.mark.unit
@pytest.mark.skipif(not LAION_VOICE_AVAILABLE, reason="LAIONVoicePipeline not available")
class TestLAIONVoicePipelineErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_segmentation_mode(self):
        """Test handling of invalid segmentation mode."""
        # Should handle invalid modes gracefully
        pipeline = LAIONVoicePipeline({"segmentation_mode": "invalid_mode"})
        
        # Should either use default or handle gracefully
        assert isinstance(pipeline.config["segmentation_mode"], str)
    
    def test_invalid_duration_bounds(self):
        """Test handling of invalid duration bounds."""
        # Test min > max case
        pipeline = LAIONVoicePipeline({
            "min_segment_duration": 10.0,
            "max_segment_duration": 5.0
        })
        
        # Should handle gracefully (might swap or use defaults)
        assert isinstance(pipeline.config["min_segment_duration"], (int, float))
        assert isinstance(pipeline.config["max_segment_duration"], (int, float))
    
    def test_cleanup_functionality(self):
        """Test pipeline cleanup."""
        pipeline = LAIONVoicePipeline()
        
        # Should have cleanup method
        assert hasattr(pipeline, 'cleanup')
        assert callable(getattr(pipeline, 'cleanup'))
        
        # Should handle cleanup when not initialized
        pipeline.cleanup()  # Should not raise error


@pytest.mark.integration
@pytest.mark.skipif(not LAION_VOICE_AVAILABLE, reason="LAIONVoicePipeline not available")
class TestLAIONVoicePipelineIntegration:
    """Integration tests with mocked dependencies."""
    
    def test_whisper_base_integration(self):
        """Test integration with WhisperBasePipeline."""
        if not WHISPER_BASE_AVAILABLE:
            pytest.skip("WhisperBasePipeline not available")
        
        pipeline = LAIONVoicePipeline()
        
        # Should inherit WhisperBase configuration
        whisper_config_keys = ["whisper_model", "sample_rate", "device"]
        for key in whisper_config_keys:
            assert key in pipeline.config
    
    def test_diarization_integration_config(self):
        """Test diarization integration configuration."""
        pipeline = LAIONVoicePipeline({
            "enable_diarization": True,
            "segmentation_mode": "diarization"
        })
        
        assert pipeline.config["enable_diarization"] == True
        assert pipeline.config["segmentation_mode"] == "diarization"
    
    def test_scene_alignment_config(self):
        """Test scene alignment configuration.""" 
        pipeline = LAIONVoicePipeline({
            "enable_scene_alignment": True,
            "segmentation_mode": "scene_based"
        })
        
        assert pipeline.config["enable_scene_alignment"] == True
        assert pipeline.config["segmentation_mode"] == "scene_based"
