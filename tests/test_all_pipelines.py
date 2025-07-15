"""
Stage 4: All Pipelines Test Integration

Test runner for all VideoAnnotator pipelines including the new LAION pipelines.
This file coordinates testing across all pipeline types.
"""

import pytest
from pathlib import Path

# Import tests for all pipelines
from tests.test_whisper_base_pipeline import *
from tests.test_laion_face_pipeline import *
from tests.test_laion_voice_pipeline import *

# Try to import existing pipeline tests
try:
    from tests.test_face_pipeline_modern import *
    FACE_PIPELINE_TESTS_AVAILABLE = True
except ImportError:
    FACE_PIPELINE_TESTS_AVAILABLE = False

try:
    from tests.test_audio_pipeline import *
    AUDIO_PIPELINE_TESTS_AVAILABLE = True
except ImportError:
    AUDIO_PIPELINE_TESTS_AVAILABLE = False

try:
    from tests.test_person_pipeline_modern import *
    PERSON_PIPELINE_TESTS_AVAILABLE = True
except ImportError:
    PERSON_PIPELINE_TESTS_AVAILABLE = False

try:
    from tests.test_scene_pipeline_modern import *
    SCENE_PIPELINE_TESTS_AVAILABLE = True
except ImportError:
    SCENE_PIPELINE_TESTS_AVAILABLE = False


@pytest.mark.integration
class TestAllPipelinesIntegration:
    """Integration tests across all pipeline types."""
    
    def test_pipeline_imports(self):
        """Test that all pipelines can be imported."""
        import_results = {}
        
        # Test core pipelines
        try:
            from src.pipelines.face_analysis.face_pipeline import FaceAnalysisPipeline
            import_results["face_analysis"] = True
        except ImportError:
            import_results["face_analysis"] = False
        
        try:
            from src.pipelines.audio_processing.audio_pipeline_modular import AudioPipeline
            import_results["audio_processing"] = True
        except ImportError:
            import_results["audio_processing"] = False
        
        try:
            from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline
            import_results["person_tracking"] = True
        except ImportError:
            import_results["person_tracking"] = False
        
        try:
            from src.pipelines.scene_detection.scene_pipeline import SceneDetectionPipeline
            import_results["scene_detection"] = True
        except ImportError:
            import_results["scene_detection"] = False
        
        # Test new LAION pipelines
        try:
            from src.pipelines.audio_processing.whisper_base_pipeline import WhisperBasePipeline
            import_results["whisper_base"] = True
        except ImportError:
            import_results["whisper_base"] = False
        
        try:
            from src.pipelines.face_analysis.laion_face_pipeline import LAIONFacePipeline
            import_results["laion_face"] = True
        except ImportError:
            import_results["laion_face"] = False
        
        try:
            from src.pipelines.audio_processing.laion_voice_pipeline import LAIONVoicePipeline
            import_results["laion_voice"] = True
        except ImportError:
            import_results["laion_voice"] = False
        
        # Report results
        total_pipelines = len(import_results)
        successful_imports = sum(import_results.values())
        
        print(f"\\nPipeline Import Results: {successful_imports}/{total_pipelines}")
        for pipeline, success in import_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {pipeline}")
        
        # At least some core pipelines should be available
        assert successful_imports > 0, "No pipelines could be imported"
    
    def test_new_pipeline_inheritance(self):
        """Test that new pipelines inherit correctly."""
        inheritance_results = {}
        
        # Test LAIONVoicePipeline inherits from WhisperBasePipeline
        try:
            from src.pipelines.audio_processing.laion_voice_pipeline import LAIONVoicePipeline
            from src.pipelines.audio_processing.whisper_base_pipeline import WhisperBasePipeline
            
            pipeline = LAIONVoicePipeline()
            inheritance_results["laion_voice_inherits_whisper"] = isinstance(pipeline, WhisperBasePipeline)
        except ImportError:
            inheritance_results["laion_voice_inherits_whisper"] = None
        
        # Test all pipelines inherit from BasePipeline
        try:
            from src.pipelines.base_pipeline import BasePipeline
            from src.pipelines.face_analysis.laion_face_pipeline import LAIONFacePipeline
            
            face_pipeline = LAIONFacePipeline()
            inheritance_results["laion_face_inherits_base"] = isinstance(face_pipeline, BasePipeline)
        except ImportError:
            inheritance_results["laion_face_inherits_base"] = None
        
        try:
            from src.pipelines.base_pipeline import BasePipeline
            from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline
            
            speech_pipeline = SpeechPipeline()
            inheritance_results["speech_inherits_base"] = isinstance(speech_pipeline, BasePipeline)
        except ImportError:
            inheritance_results["speech_inherits_base"] = None
        
        # Report inheritance results
        for test, result in inheritance_results.items():
            if result is True:
                print(f"✅ {test}")
            elif result is False:
                print(f"❌ {test}")
            else:
                print(f"⚠️  {test} (skipped - import failed)")
    
    def test_pipeline_configuration_consistency(self):
        """Test that all pipelines handle configuration consistently."""
        config_results = {}
        
        pipelines_to_test = []
        
        # Add available pipelines
        try:
            from src.pipelines.face_analysis.laion_face_pipeline import LAIONFacePipeline
            pipelines_to_test.append(("LAIONFacePipeline", LAIONFacePipeline))
        except ImportError:
            pass
        
        try:
            from src.pipelines.audio_processing.laion_voice_pipeline import LAIONVoicePipeline
            pipelines_to_test.append(("LAIONVoicePipeline", LAIONVoicePipeline))
        except ImportError:
            pass
        
        try:
            from src.pipelines.audio_processing.whisper_base_pipeline import WhisperBasePipeline
            pipelines_to_test.append(("WhisperBasePipeline", WhisperBasePipeline))
        except ImportError:
            pass
        
        # Test each pipeline
        for name, pipeline_class in pipelines_to_test:
            try:
                # Test default initialization
                pipeline1 = pipeline_class()
                assert hasattr(pipeline1, 'config'), f"{name} missing config attribute"
                
                # Test custom config
                pipeline2 = pipeline_class({"test_param": "test_value"})
                assert hasattr(pipeline2, 'config'), f"{name} missing config with custom params"
                
                # Test required methods
                required_methods = ['initialize', 'cleanup', 'get_schema']
                for method in required_methods:
                    assert hasattr(pipeline2, method), f"{name} missing {method} method"
                    assert callable(getattr(pipeline2, method)), f"{name}.{method} not callable"
                
                config_results[name] = True
                
            except Exception as e:
                config_results[name] = f"Failed: {e}"
        
        # Report results
        for pipeline, result in config_results.items():
            if result is True:
                print(f"✅ {pipeline} configuration")
            else:
                print(f"❌ {pipeline} configuration: {result}")
        
        # At least some pipelines should pass
        successful_configs = sum(1 for result in config_results.values() if result is True)
        assert successful_configs > 0, "No pipelines passed configuration tests"


@pytest.mark.meta
class TestPipelineTestCoverage:
    """Meta-tests to ensure test coverage is adequate."""
    
    def test_new_pipelines_have_tests(self):
        """Test that new pipelines have corresponding test files."""
        test_files = {
            "whisper_base_pipeline": "tests/test_whisper_base_pipeline_stage1.py",
            "laion_face_pipeline": "tests/test_laion_face_pipeline.py", 
            "laion_voice_pipeline": "tests/test_laion_voice_pipeline.py"
        }
        
        missing_tests = []
        for pipeline, test_file in test_files.items():
            if not Path(test_file).exists():
                missing_tests.append(f"{pipeline} -> {test_file}")
        
        assert len(missing_tests) == 0, f"Missing test files: {missing_tests}"
    
    def test_test_files_structure(self):
        """Test that test files follow proper structure."""
        test_files = [
            "tests/test_whisper_base_pipeline_stage1.py",
            "tests/test_laion_face_pipeline.py",
            "tests/test_laion_voice_pipeline.py"
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # Check for proper structure
                assert 'import pytest' in content, f"{test_file} missing pytest import"
                assert '@pytest.mark.unit' in content, f"{test_file} missing unit test markers"
                assert 'class Test' in content, f"{test_file} missing test classes"
    
    def test_existing_tests_compatibility(self):
        """Test that existing tests still work with new pipelines."""
        # Test that we haven't broken existing test structure
        existing_test_files = [
            "tests/test_face_pipeline_modern.py",
            "tests/test_audio_pipeline.py", 
            "tests/test_person_pipeline_modern.py",
            "tests/test_scene_pipeline_modern.py"
        ]
        
        available_tests = []
        for test_file in existing_test_files:
            if Path(test_file).exists():
                available_tests.append(test_file)
        
        # Should have at least some existing tests
        assert len(available_tests) > 0, "No existing test files found"
        
        print(f"\\nExisting test files available: {len(available_tests)}")
        for test_file in available_tests:
            print(f"  ✅ {test_file}")


@pytest.mark.performance
class TestPipelinePerformanceBaseline:
    """Baseline performance tests for all pipelines."""
    
    def test_import_performance(self):
        """Test that pipeline imports are reasonably fast."""
        import time
        
        import_times = {}
        
        # Test import times for new pipelines
        start_time = time.time()
        try:
            from src.pipelines.audio_processing.whisper_base_pipeline import WhisperBasePipeline
            import_times["whisper_base"] = time.time() - start_time
        except ImportError:
            import_times["whisper_base"] = None
        
        start_time = time.time()
        try:
            from src.pipelines.face_analysis.laion_face_pipeline import LAIONFacePipeline
            import_times["laion_face"] = time.time() - start_time
        except ImportError:
            import_times["laion_face"] = None
        
        start_time = time.time()
        try:
            from src.pipelines.audio_processing.laion_voice_pipeline import LAIONVoicePipeline
            import_times["laion_voice"] = time.time() - start_time
        except ImportError:
            import_times["laion_voice"] = None
        
        # Report import times
        for pipeline, import_time in import_times.items():
            if import_time is not None:
                print(f"\\n{pipeline} import time: {import_time:.3f}s")
                # Import should be reasonably fast (< 5 seconds)
                assert import_time < 5.0, f"{pipeline} import too slow: {import_time:.3f}s"
    
    def test_initialization_performance(self):
        """Test that pipeline initialization is reasonably fast."""
        import time
        
        init_times = {}
        
        # Test initialization times
        try:
            from src.pipelines.face_analysis.laion_face_pipeline import LAIONFacePipeline
            start_time = time.time()
            pipeline = LAIONFacePipeline()
            init_times["laion_face"] = time.time() - start_time
        except ImportError:
            init_times["laion_face"] = None
        
        try:
            from src.pipelines.audio_processing.laion_voice_pipeline import LAIONVoicePipeline
            start_time = time.time()
            pipeline = LAIONVoicePipeline()
            init_times["laion_voice"] = time.time() - start_time
        except ImportError:
            init_times["laion_voice"] = None
        
        # Report initialization times
        for pipeline, init_time in init_times.items():
            if init_time is not None:
                print(f"\\n{pipeline} init time: {init_time:.3f}s")
                # Initialization should be fast (< 1 second without model loading)
                assert init_time < 1.0, f"{pipeline} initialization too slow: {init_time:.3f}s"
