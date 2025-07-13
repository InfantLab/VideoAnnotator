"""
DEPRECATED: Pipeline tests have been moved to individual files.

This module is maintained for backward compatibility. 
New tests and improvements are in the individual pipeline test files:

- test_face_pipeline.py: Face Analysis Pipeline tests
- test_audio_pipeline.py: Audio Processing Pipeline tests  
- test_scene_pipeline.py: Scene Detection Pipeline tests
- test_person_pipeline.py: Person Tracking Pipeline tests
- test_audio_speech_pipeline.py: Diarization and Speech Recognition tests
- test_all_pipelines.py: Test runner for all pipeline tests

Please use the new individual test files for better organization and maintainability.
The tests below include fixes applied during development for:
1. Face Analysis Pipeline missing methods and JSON serialization
2. Unicode logging issues on Windows
3. Audio Processing validation errors
"""

import pytest
import warnings

# Import from new test modules for backward compatibility
from .test_face_pipeline import TestFaceAnalysisPipeline
from .test_audio_pipeline import TestAudioPipeline  
from .test_scene_pipeline import TestSceneDetectionPipeline
from .test_person_pipeline import TestPersonTrackingPipeline
from .test_audio_speech_pipeline import TestDiarizationPipeline, TestSpeechPipeline

# Issue deprecation warning
warnings.warn(
    "test_pipelines.py is deprecated. Please use individual pipeline test files: "
    "test_face_pipeline.py, test_audio_pipeline.py, test_scene_pipeline.py, "
    "test_person_pipeline.py, test_audio_speech_pipeline.py",
    DeprecationWarning,
    stacklevel=2
)


# Legacy test classes for backward compatibility
class TestSceneDetectionPipelineLegacy(TestSceneDetectionPipeline):
    """Legacy wrapper for scene detection tests."""
    pass


class TestPersonTrackingPipelineLegacy(TestPersonTrackingPipeline):
    """Legacy wrapper for person tracking tests."""
    pass


class TestFaceAnalysisPipelineLegacy(TestFaceAnalysisPipeline):
    """Legacy wrapper for face analysis tests."""
    pass


class TestAudioPipelineLegacy(TestAudioPipeline):
    """Legacy wrapper for audio pipeline tests."""
    pass


class TestDiarizationPipelineLegacy(TestDiarizationPipeline):
    """Legacy wrapper for diarization tests."""
    pass


class TestSpeechPipelineLegacy(TestSpeechPipeline):
    """Legacy wrapper for speech recognition tests."""
    pass


# Test runner function for backward compatibility
def run_all_pipeline_tests():
    """
    Run all pipeline tests using the new modular structure.
    
    This function provides backward compatibility for existing test runners
    while leveraging the improved modular test organization.
    """
    from .test_all_pipelines import run_all_tests
    return run_all_tests()


if __name__ == "__main__":
    # Run tests when module is executed directly
    pytest.main([__file__])
