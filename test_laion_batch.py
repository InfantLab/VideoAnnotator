#!/usr/bin/env python3
"""
Test script for LAION pipelines in batch processing.

This script demonstrates how to use the LAION face and voice emotion analysis pipelines
in the VideoAnnotator batch processing system.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.batch.batch_orchestrator import BatchOrchestrator
from src.storage.file_backend import FileStorageBackend

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('batch_laion_test.log')
        ]
    )

def main():
    """Test LAION pipelines in batch processing."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LAION batch processing test")
    
    # Create storage backend
    storage_backend = FileStorageBackend(Path("batch_laion_test_results"))
    
    # Create batch orchestrator
    orchestrator = BatchOrchestrator(
        storage_backend=storage_backend,
        max_retries=2,
        checkpoint_interval=1  # Save checkpoint after each job for testing
    )
    
    # Test video path (you can modify this to point to an actual video file)
    test_video = Path("demovideos/babyjokes/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4")
    
    if not test_video.exists():
        logger.warning(f"Test video not found: {test_video}")
        logger.info("Creating a mock batch test instead...")
        
        # List available pipelines
        logger.info("Available pipelines:")
        for name in orchestrator.pipeline_classes.keys():
            logger.info(f"  - {name}")
        
        return
    
    logger.info(f"Using test video: {test_video}")
    
    # Test configurations for different pipeline combinations
    test_configs = [
        {
            "name": "LAION Face Only",
            "pipelines": ["laion_face_analysis"],
            "config": {
                "laion_face_analysis": {
                    "model_size": "small",
                    "confidence_threshold": 0.7,
                    "top_k_emotions": 5,
                    "device": "auto"
                }
            }
        },
        {
            "name": "LAION Voice Only", 
            "pipelines": ["laion_voice_analysis"],
            "config": {
                "laion_voice_analysis": {
                    "model_size": "small",
                    "segmentation_mode": "fixed_interval",
                    "min_segment_duration": 1.0,
                    "max_segment_duration": 10.0,
                    "include_transcription": True,
                    "top_k_emotions": 5
                }
            }
        },
        {
            "name": "Both LAION Pipelines",
            "pipelines": ["laion_face_analysis", "laion_voice_analysis"],
            "config": {
                "laion_face_analysis": {
                    "model_size": "small",
                    "confidence_threshold": 0.7,
                    "top_k_emotions": 5
                },
                "laion_voice_analysis": {
                    "model_size": "small", 
                    "segmentation_mode": "fixed_interval",
                    "include_transcription": True,
                    "top_k_emotions": 5
                }
            }
        }
    ]
    
    # Run each test configuration
    for i, test_config in enumerate(test_configs, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test {i}: {test_config['name']}")
        logger.info(f"{'='*60}")
        
        # Clear previous jobs
        orchestrator.clear_jobs()
        
        # Add job for this test
        job_id = orchestrator.add_job(
            video_path=test_video,
            output_dir=Path("batch_laion_test_results") / f"test_{i}_{test_config['name'].lower().replace(' ', '_')}",
            config=test_config["config"],
            selected_pipelines=test_config["pipelines"]
        )
        
        logger.info(f"Added job {job_id} with pipelines: {test_config['pipelines']}")
        
        # Run the batch
        try:
            start_time = datetime.now()
            report = orchestrator.run_batch(max_workers=1, save_checkpoints=True)
            end_time = datetime.now()
            
            logger.info(f"Batch completed in {(end_time - start_time).total_seconds():.2f} seconds")
            logger.info(f"Success rate: {report.success_rate:.1f}%")
            logger.info(f"Completed jobs: {report.completed_jobs}/{report.total_jobs}")
            
            if report.failed_jobs > 0:
                logger.warning(f"Failed jobs: {report.failed_jobs}")
                for error in report.errors:
                    logger.error(f"  - {error}")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            continue
    
    logger.info("\n" + "="*60)
    logger.info("LAION batch processing test completed")
    logger.info("="*60)

if __name__ == "__main__":
    main()
