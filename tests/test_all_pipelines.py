"""
Comprehensive pipeline test suite runner.

This file imports and runs all pipeline tests from the individual test modules.
"""

import pytest

# Import all pipeline test modules
try:
    # Try relative imports first (when run as part of package)
    from .test_face_pipeline import TestFaceAnalysisPipeline, TestFaceAnalysisPipelineIntegration, TestFaceAnalysisPipelinePerformance
    from .test_audio_pipeline import TestAudioPipeline, TestAudioPipelineIntegration, TestAudioPipelinePerformance
    from .test_scene_pipeline import TestSceneDetectionPipeline, TestSceneDetectionPipelineIntegration, TestSceneDetectionPipelinePerformance
    from .test_person_pipeline import TestPersonTrackingPipeline, TestPersonTrackingPipelineIntegration, TestPersonTrackingPipelinePerformance
    from .test_audio_speech_pipeline import TestDiarizationPipeline, TestSpeechPipeline, TestDiarizationSpeechIntegration, TestAudioPipelinePerformance as TestAudioSpeechPerformance
except ImportError:
    # Fall back to absolute imports (when run directly)
    from test_face_pipeline import TestFaceAnalysisPipeline, TestFaceAnalysisPipelineIntegration, TestFaceAnalysisPipelinePerformance
    from test_audio_pipeline import TestAudioPipeline, TestAudioPipelineIntegration, TestAudioPipelinePerformance
    from test_scene_pipeline import TestSceneDetectionPipeline, TestSceneDetectionPipelineIntegration, TestSceneDetectionPipelinePerformance
    from test_person_pipeline import TestPersonTrackingPipeline, TestPersonTrackingPipelineIntegration, TestPersonTrackingPipelinePerformance
    from test_audio_speech_pipeline import TestDiarizationPipeline, TestSpeechPipeline, TestDiarizationSpeechIntegration, TestAudioPipelinePerformance as TestAudioSpeechPerformance


# Test collections for easy organization
UNIT_TEST_CLASSES = [
    TestFaceAnalysisPipeline,
    TestAudioPipeline,
    TestSceneDetectionPipeline,
    TestPersonTrackingPipeline,
    TestDiarizationPipeline,
    TestSpeechPipeline
]

INTEGRATION_TEST_CLASSES = [
    TestFaceAnalysisPipelineIntegration,
    TestAudioPipelineIntegration,
    TestSceneDetectionPipelineIntegration,
    TestPersonTrackingPipelineIntegration,
    TestDiarizationSpeechIntegration
]

PERFORMANCE_TEST_CLASSES = [
    TestFaceAnalysisPipelinePerformance,
    TestAudioPipelinePerformance,
    TestSceneDetectionPipelinePerformance,
    TestPersonTrackingPipelinePerformance,
    TestAudioSpeechPerformance
]


def run_unit_tests():
    """Run all unit tests for pipelines."""
    pytest.main([
        "-v",
        "-m", "unit",
        "tests/test_face_pipeline.py",
        "tests/test_audio_pipeline.py", 
        "tests/test_scene_pipeline.py",
        "tests/test_person_pipeline.py",
        "tests/test_audio_speech_pipeline.py"
    ])


def run_integration_tests():
    """Run all integration tests for pipelines."""
    pytest.main([
        "-v", 
        "-m", "integration",
        "tests/test_face_pipeline.py",
        "tests/test_audio_pipeline.py",
        "tests/test_scene_pipeline.py", 
        "tests/test_person_pipeline.py",
        "tests/test_audio_speech_pipeline.py"
    ])


def run_performance_tests():
    """Run all performance tests for pipelines."""
    pytest.main([
        "-v",
        "-m", "performance", 
        "tests/test_face_pipeline.py",
        "tests/test_audio_pipeline.py",
        "tests/test_scene_pipeline.py",
        "tests/test_person_pipeline.py", 
        "tests/test_audio_speech_pipeline.py"
    ])


def run_all_pipeline_tests():
    """Run all pipeline tests."""
    pytest.main([
        "-v",
        "tests/test_face_pipeline.py",
        "tests/test_audio_pipeline.py",
        "tests/test_scene_pipeline.py",
        "tests/test_person_pipeline.py",
        "tests/test_audio_speech_pipeline.py"
    ])


if __name__ == "__main__":
    # Run all tests by default
    run_all_pipeline_tests()
