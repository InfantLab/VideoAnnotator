#!/usr/bin/env python3
"""
Simple test script to verify the refactored SpeechPipeline works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.pipelines.audio_processing import SpeechPipeline, WhisperBasePipeline
    print("✓ Successfully imported SpeechPipeline and WhisperBasePipeline")
    
    # Test SpeechPipeline creation
    speech_pipeline = SpeechPipeline()
    print("✓ Successfully created SpeechPipeline instance")
    
    # Check inheritance
    print(f"✓ SpeechPipeline inherits from WhisperBasePipeline: {isinstance(speech_pipeline, WhisperBasePipeline)}")
    
    # Check configuration
    print(f"✓ Whisper model: {speech_pipeline.config['whisper_model']}")
    print(f"✓ Sample rate: {speech_pipeline.config['sample_rate']}")
    print(f"✓ Device: {speech_pipeline.config['device']}")
    
    # Test with custom config
    custom_config = {
        "whisper_model": "small",
        "language": "en",
        "word_timestamps": True
    }
    
    custom_pipeline = SpeechPipeline(custom_config)
    print("✓ Successfully created SpeechPipeline with custom config")
    print(f"✓ Custom model: {custom_pipeline.config['whisper_model']}")
    print(f"✓ Custom language: {custom_pipeline.config['language']}")
    
    print("\n🎉 All tests passed! SpeechPipeline refactoring is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
