"""Example: Testing the LAION Voice Pipeline.

This example demonstrates how to use the LAIONVoicePipeline
to process a video and extract voice emotion annotations.

Usage:
    python examples/test_laion_voice_pipeline.py --video_path /path/to/video.mp4 --output_dir /path/to/output"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# Import the LAIONVoicePipeline
from src.pipelines.audio_processing import LAIONVoicePipeline


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"laion_voice_test_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def test_laion_voice_pipeline(video_path: str, output_dir: str, config: dict = None):
    """Test the LAION Voice Pipeline with a video."""
    logger = logging.getLogger(__name__)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    logger.info("=" * 50)
    logger.info("Testing LAION Voice Pipeline")
    logger.info("=" * 50)

    # Initialize pipeline with default or custom config
    pipeline = LAIONVoicePipeline(config=config)
    pipeline.initialize()

    # Get pipeline info
    pipeline_info = pipeline.get_pipeline_info()
    logger.info(f"Pipeline Info: {json.dumps(pipeline_info, indent=2)}")

    # Process video
    logger.info(f"Processing video: {video_path}")
    start_time = datetime.now()

    results = pipeline.process(
        video_path=video_path,
        output_dir=output_dir,
        include_transcription=True,  # Enable transcription for more meaningful results
    )

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    # Display results summary
    logger.info("Results Summary:")
    logger.info(f"  - Total processing time: {processing_time:.2f} seconds")
    logger.info(f"  - Total segments: {len(results)}")

    if results:
        # Show some example results
        logger.info("  - Segment examples:")
        for i, segment in enumerate(results[:3]):  # Show first 3 segments
            logger.info(
                f"    Segment {i + 1}: {segment.get('start_time', 0):.2f}s - {segment.get('end_time', 0):.2f}s"
            )

            if segment.get("emotions"):
                top_emotions = ", ".join(
                    [
                        f"{emotion}({data['score']:.2f})"
                        for emotion, data in list(segment["emotions"].items())[
                            :3
                        ]  # Top 3 emotions
                    ]
                )
                logger.info(f"      Top emotions: {top_emotions}")

            if "transcription" in segment:
                logger.info(f"      Transcription: {segment['transcription']}")

    logger.info("=" * 50)
    logger.info("LAION Voice Pipeline Test Complete")
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the LAION Voice Pipeline")
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./laion_voice_test_results",
        help="Output directory",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Model size",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    # Custom config
    config = {
        "model_size": args.model_size,
        "include_transcription": True,
        "top_k_emotions": 5,
    }

    # Run the test
    test_laion_voice_pipeline(args.video_path, args.output_dir, config)
