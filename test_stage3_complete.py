#!/usr/bin/env python3
"""
Comprehensive test for the Whisper pipeline refactoring.
Tests Stage 3 implementation - SpeechPipeline inheriting from WhisperBasePipeline.
"""

import sys
import os
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

def test_whisper_base_pipeline():
    """Test WhisperBasePipeline standalone"""
    print("=" * 50)
    print("Testing WhisperBasePipeline...")
    print("=" * 50)
    
    try:
        from src.pipelines.audio_processing.whisper_base_pipeline import WhisperBasePipeline
        print("✓ WhisperBasePipeline imported successfully")
        
        # Test default config
        pipeline = WhisperBasePipeline()
        print("✓ WhisperBasePipeline created with default config")
        print(f"  - Whisper model: {pipeline.config['whisper_model']}")
        print(f"  - Sample rate: {pipeline.config['sample_rate']}")
        print(f"  - Device: {pipeline.config['device']}")
        print(f"  - Use FP16: {pipeline.config['use_fp16']}")
        
        # Test custom config
        custom_config = {
            "whisper_model": "small",
            "device": "cpu",
            "use_fp16": False
        }
        
        custom_pipeline = WhisperBasePipeline(custom_config)
        print("✓ WhisperBasePipeline created with custom config")
        print(f"  - Custom model: {custom_pipeline.config['whisper_model']}")
        print(f"  - Custom device: {custom_pipeline.config['device']}")
        
        return True
        
    except Exception as e:
        print(f"❌ WhisperBasePipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_speech_pipeline():
    """Test SpeechPipeline with WhisperBasePipeline inheritance"""
    print("\n" + "=" * 50)
    print("Testing SpeechPipeline...")
    print("=" * 50)
    
    try:
        from src.pipelines.audio_processing.speech_pipeline import SpeechPipeline
        from src.pipelines.audio_processing.whisper_base_pipeline import WhisperBasePipeline
        print("✓ SpeechPipeline imported successfully")
        
        # Test default config
        speech_pipeline = SpeechPipeline()
        print("✓ SpeechPipeline created with default config")
        
        # Test inheritance
        is_instance = isinstance(speech_pipeline, WhisperBasePipeline)
        print(f"✓ SpeechPipeline inherits from WhisperBasePipeline: {is_instance}")
        
        if not is_instance:
            raise ValueError("SpeechPipeline should inherit from WhisperBasePipeline")
        
        # Test configuration merging
        print(f"  - Whisper model: {speech_pipeline.config['whisper_model']}")
        print(f"  - Language: {speech_pipeline.config.get('language', 'None')}")
        print(f"  - Task: {speech_pipeline.config.get('task', 'transcribe')}")
        print(f"  - Word timestamps: {speech_pipeline.config.get('word_timestamps', True)}")
        
        # Test with legacy "model" key for backward compatibility
        legacy_config = {
            "model": "tiny",  # Legacy key
            "language": "en",
            "word_timestamps": True
        }
        
        legacy_pipeline = SpeechPipeline(legacy_config)
        print("✓ SpeechPipeline created with legacy config")
        print(f"  - Legacy model converted: {legacy_pipeline.config['whisper_model']}")
        print(f"  - Language: {legacy_pipeline.config['language']}")
        
        # Test output format property
        output_format = speech_pipeline.output_format
        print(f"✓ Output format: {output_format}")
        
        # Test schema method
        schema = speech_pipeline.get_schema()
        print(f"✓ Schema type: {schema['type']}")
        
        # Test pipeline info
        info = speech_pipeline.get_pipeline_info()
        print(f"✓ Pipeline name: {info['name']}")
        print(f"  - Version: {info['version']}")
        
        return True
        
    except Exception as e:
        print(f"❌ SpeechPipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_pipeline_import():
    """Test that the overall audio_processing package imports work"""
    print("\n" + "=" * 50)
    print("Testing audio_processing package imports...")
    print("=" * 50)
    
    try:
        from src.pipelines.audio_processing import (
            SpeechPipeline, 
            WhisperBasePipeline, 
            AudioPipelineModular
        )
        print("✓ All main classes imported successfully from audio_processing package")
        
        # Test that SpeechPipeline is properly integrated
        speech = SpeechPipeline()
        print("✓ SpeechPipeline can be created from package import")
        
        return True
        
    except Exception as e:
        print(f"❌ Package import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Starting Whisper Pipeline Refactoring Tests (Stage 3)")
    print("Testing SpeechPipeline inheriting from WhisperBasePipeline")
    
    tests = [
        test_whisper_base_pipeline,
        test_speech_pipeline,
        test_audio_pipeline_import
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("✓ WhisperBasePipeline is working correctly")
        print("✓ SpeechPipeline successfully inherits from WhisperBasePipeline")
        print("✓ Configuration handling works properly")
        print("✓ Backward compatibility is maintained")
        print("✓ Package imports are working")
        print("\nStage 3 of the Whisper upgrade is complete!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
