"""Example usage of the diarization pipeline.

This script shows how to use the speaker diarization functionality from
the VideoAnnotator project.
"""

import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_diarization(video_path: str, output_dir: str = None):
    """Example function showing how to use the diarization pipeline.

    Args:
        video_path: Path to the video file to process
        output_dir: Optional output directory for audio files
    """

    try:
        from src.pipelines.audio_processing import (
            DiarizationPipeline,
            DiarizationPipelineConfig,
        )

        # Configure the pipeline
        config = DiarizationPipelineConfig(
            # Token will be read from HUGGINGFACE_TOKEN environment variable
            diarization_model="pyannote/speaker-diarization-3.1",
            use_gpu=True,  # Use GPU if available
        )

        # Create and initialize the pipeline
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()

        # Process the video
        logger.info(f"Processing video: {video_path}")
        results = pipeline.process(video_path, output_dir=output_dir)

        if results:
            diarization = results[0]

            # Print results
            print(f"\nDIARIZATION RESULTS for {Path(video_path).name}")
            print("=" * 60)
            print(f"Number of speakers detected: {len(diarization.speakers)}")
            print(f"Number of speaker segments: {len(diarization.segments)}")
            print(f"Total speech time: {diarization.total_speech_time:.2f} seconds")

            # Show speaker breakdown
            print("\nSpeaker Breakdown:")
            speaker_times = {}
            for segment in diarization.segments:
                speaker_id = segment["speaker_id"]
                duration = segment["end_time"] - segment["start_time"]
                if speaker_id not in speaker_times:
                    speaker_times[speaker_id] = 0
                speaker_times[speaker_id] += duration

            for speaker_id, total_time in speaker_times.items():
                percentage = (total_time / diarization.total_speech_time) * 100
                print(f"  {speaker_id}: {total_time:.2f}s ({percentage:.1f}%)")

            # Show timeline
            print("\nSpeaker Timeline (first 10 segments):")
            for i, segment in enumerate(diarization.segments[:10]):
                start = segment["start_time"]
                end = segment["end_time"]
                speaker = segment["speaker_id"]
                duration = end - start
                print(f"  {start:6.2f}s - {end:6.2f}s ({duration:5.2f}s): {speaker}")

            if len(diarization.segments) > 10:
                print(f"  ... and {len(diarization.segments) - 10} more segments")

            return diarization
        else:
            logger.error("No diarization results returned")
            return None

    except Exception as e:
        logger.error(f"Error in diarization example: {e}")
        return None


def main():
    """Main function."""

    # Check for HuggingFace token
    if not os.getenv("HUGGINGFACE_TOKEN"):
        print("HUGGINGFACE_TOKEN environment variable not set")
        print("Please set your HuggingFace token:")
        print("export HUGGINGFACE_TOKEN=your_token_here")
        print("\nGet a token from: https://huggingface.co/settings/tokens")
        return

    # Find a test video
    video_paths = []

    # Look in common directories
    for pattern in ["babyjokes videos/*.mp4", "data/demovideos/*.mp4", "data/*.mp4"]:
        video_paths.extend(list(Path(".").glob(pattern)))

    if not video_paths:
        print("No video files found")
        print("Please ensure video files are available in:")
        print("  - babyjokes videos/")
        print("  - data/demovideos/")
        print("  - data/")
        return

    # Use the first video found
    video_path = str(video_paths[0])
    print(f"Using video: {video_path}")

    # Run the example
    result = example_diarization(video_path)

    if result:
        print("\nDiarization example completed successfully!")
    else:
        print("\nDiarization example failed")


if __name__ == "__main__":
    main()
