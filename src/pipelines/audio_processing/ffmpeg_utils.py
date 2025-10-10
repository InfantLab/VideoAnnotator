"""FFmpeg-based audio extraction utilities for the audio processing pipeline."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available in the system PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio_from_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    format: str = "wav",
    overwrite: bool = True,
) -> Path | None:
    """
    Extract audio from video file using FFmpeg.

    Args:
        video_path: Path to input video file
        output_path: Path for output audio file (optional)
        sample_rate: Target sample rate in Hz
        channels: Number of audio channels (1=mono, 2=stereo)
        format: Output audio format (wav, mp3, etc.)
        overwrite: Whether to overwrite existing output file

    Returns:
        Path to extracted audio file or None if extraction fails
    """
    if not check_ffmpeg_available():
        logger.error("FFmpeg not found. Please install FFmpeg.")
        return None

    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return None

    # Check if video has audio streams
    if not has_audio_stream(video_path):
        logger.warning(f"No audio streams found in video: {video_path}")
        return None

    # Generate output path if not provided
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_audio.{format}"
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",  # No video stream
        "-acodec",
        "pcm_s16le",  # 16-bit PCM for WAV
        "-ar",
        str(sample_rate),  # Sample rate
        "-ac",
        str(channels),  # Audio channels
    ]

    if overwrite:
        cmd.append("-y")  # Overwrite output file

    cmd.append(str(output_path))

    try:
        logger.info(f"Extracting audio from {video_path} to {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            logger.info(f"Audio successfully extracted to {output_path}")
            return output_path
        else:
            logger.error(f"FFmpeg extraction failed: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Audio extraction error: {e}")
        return None


def convert_audio_format(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int | None = None,
    channels: int | None = None,
    overwrite: bool = True,
) -> Path | None:
    """
    Convert audio file to different format using FFmpeg.

    Args:
        input_path: Path to input audio file
        output_path: Path for output audio file
        sample_rate: Target sample rate (optional)
        channels: Number of channels (optional)
        overwrite: Whether to overwrite existing output file

    Returns:
        Path to converted audio file or None if conversion fails
    """
    if not check_ffmpeg_available():
        logger.error("FFmpeg not found. Please install FFmpeg.")
        return None

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        logger.error(f"Input audio file not found: {input_path}")
        return None

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg command
    cmd = ["ffmpeg", "-i", str(input_path)]

    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])

    if channels:
        cmd.extend(["-ac", str(channels)])

    if overwrite:
        cmd.append("-y")

    cmd.append(str(output_path))

    try:
        logger.info(f"Converting audio from {input_path} to {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            logger.info(f"Audio successfully converted to {output_path}")
            return output_path
        else:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return None


def get_audio_info(audio_path: str | Path) -> dict | None:
    """
    Get audio file information using FFprobe (part of FFmpeg).

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with audio information or None if failed
    """
    if not shutil.which("ffprobe"):
        logger.error("FFprobe not found. Please install FFmpeg.")
        return None

    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return None

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(audio_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            import json

            info = json.loads(result.stdout)

            # Extract relevant audio information
            audio_info = {}
            if "format" in info:
                format_info = info["format"]
                audio_info.update(
                    {
                        "duration": float(format_info.get("duration", 0)),
                        "size": int(format_info.get("size", 0)),
                        "bit_rate": int(format_info.get("bit_rate", 0)),
                    }
                )

            # Find audio stream information
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_info.update(
                        {
                            "codec": stream.get("codec_name"),
                            "sample_rate": int(stream.get("sample_rate", 0)),
                            "channels": int(stream.get("channels", 0)),
                            "channel_layout": stream.get("channel_layout"),
                        }
                    )
                    break

            return audio_info
        else:
            logger.error(f"FFprobe failed: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Error getting audio info: {e}")
        return None


def has_audio_stream(video_path: str | Path) -> bool:
    """
    Check if video file contains audio streams.

    Args:
        video_path: Path to video file

    Returns:
        True if video has audio streams, False otherwise
    """
    if not check_ffmpeg_available():
        logger.warning("FFmpeg not found. Cannot check for audio streams.")
        return False

    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return False

    # Use ffprobe to check for audio streams
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            # If there's any output, it means audio streams were found
            return bool(result.stdout.strip())
        else:
            logger.warning(f"ffprobe failed to analyze {video_path}: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error checking audio streams: {e}")
        return False
