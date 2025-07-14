#!/usr/bin/env python3
"""Simple test for WhisperBasePipeline"""

try:
    from src.pipelines.audio_processing.whisper_base_pipeline import WhisperBasePipeline
    print("WhisperBasePipeline imported successfully")
    
    pipeline = WhisperBasePipeline()
    print("WhisperBasePipeline created successfully")
    print(f"Config: {pipeline.config}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
