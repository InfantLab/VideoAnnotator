"""
Standards-only audio processing pipeline.

This pipeline works directly with WebVTT and RTTM format outputs,
eliminating all custom schema dependencies.
"""

import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import webvtt
import librosa

from ..base_pipeline import BasePipeline
from ...exporters.native_formats import (
    create_webvtt_caption,
    create_rttm_turn,
    export_webvtt,
    export_rttm,
    validate_webvtt,
    validate_rttm,
)

# Optional imports
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from pyannote.audio import Pipeline as DiarizationPipeline

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


class AudioProcessingPipeline(BasePipeline):
    """
    Standards-only audio processing pipeline using WebVTT and RTTM formats.

    Returns native format objects/dictionaries instead of custom schemas.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            # Speech recognition settings
            "speech_model": "base",  # Whisper model size
            "language": "auto",
            "segment_length": 30.0,  # seconds
            "min_segment_duration": 1.0,
            # Diarization settings
            "enable_diarization": True,
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "min_speakers": 1,
            "max_speakers": 10,
            # Audio processing
            "sample_rate": 16000,
            "normalize_audio": True,
            # Output format
            "webvtt_format": "default",  # or "detailed"
            "rttm_format": "standard",
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

        self.logger = logging.getLogger(__name__)
        self.speech_model = None
        self.diarization_pipeline = None

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process video for speech recognition and speaker diarization.

        Returns:
            Dict containing 'speech' (WebVTT captions) and 'diarization' (RTTM turns)
        """

        # Extract audio from video
        audio_path = self._extract_audio(video_path, start_time, end_time)
        video_metadata = self._get_video_metadata(video_path)

        try:
            results = {"speech": [], "diarization": [], "metadata": video_metadata}

            # Speech recognition
            if WHISPER_AVAILABLE:
                captions = self._process_speech_recognition(audio_path, video_metadata)
                results["speech"] = captions
            else:
                self.logger.warning("Whisper not available, skipping speech recognition")

            # Speaker diarization
            if self.config["enable_diarization"] and PYANNOTE_AVAILABLE:
                turns = self._process_diarization(audio_path, video_metadata)
                results["diarization"] = turns
            else:
                self.logger.warning("Diarization disabled or pyannote not available")

            # Save results if output directory specified
            if output_dir:
                self._save_results(results, output_dir, video_metadata)

            return results

        finally:
            # Cleanup temporary audio file
            if audio_path != video_path:
                Path(audio_path).unlink(missing_ok=True)

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata."""
        try:
            # Use librosa to get audio duration
            duration = librosa.get_duration(path=video_path)

            return {
                "video_id": Path(video_path).stem,
                "filepath": video_path,
                "duration": duration,
                "sample_rate": self.config["sample_rate"],
            }
        except Exception as e:
            self.logger.error(f"Failed to get video metadata: {e}")
            return {
                "video_id": Path(video_path).stem,
                "filepath": video_path,
                "duration": 0.0,
                "sample_rate": self.config["sample_rate"],
            }

    def _extract_audio(
        self, video_path: str, start_time: float = 0.0, end_time: Optional[float] = None
    ) -> str:
        """Extract audio from video file."""

        # Create temporary file for audio
        temp_dir = Path(tempfile.gettempdir())
        audio_path = temp_dir / f"{Path(video_path).stem}_temp_audio.wav"

        # Build ffmpeg command
        cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", str(start_time)]  # Overwrite output

        if end_time is not None:
            cmd.extend(["-to", str(end_time)])

        cmd.extend(
            [
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(self.config["sample_rate"]),
                "-ac",
                "1",  # Mono
                str(audio_path),
            ]
        )

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"Audio extracted to: {audio_path}")
            return str(audio_path)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to extract audio: {e}")
            # Fallback to original video file
            return video_path

    def _process_speech_recognition(
        self, audio_path: str, video_metadata: Dict[str, Any]
    ) -> List[webvtt.Caption]:
        """Process speech recognition using Whisper."""

        # Initialize Whisper model
        if self.speech_model is None:
            self.speech_model = whisper.load_model(self.config["speech_model"])
            self.logger.info(f"Loaded Whisper model: {self.config['speech_model']}")

        # Transcribe audio
        language = None if self.config["language"] == "auto" else self.config["language"]

        result = self.speech_model.transcribe(
            audio_path, language=language, word_timestamps=True, verbose=False
        )

        # Convert to WebVTT captions
        captions = []

        for segment in result.get("segments", []):
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            text = segment.get("text", "").strip()

            if text and (end_time - start_time) >= self.config["min_segment_duration"]:
                caption = create_webvtt_caption(
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    confidence=segment.get("confidence"),
                    speaker_id=None,  # No speaker info from Whisper alone
                )
                captions.append(caption)

        self.logger.info(f"Speech recognition complete: {len(captions)} segments")
        return captions

    def _process_diarization(
        self, audio_path: str, video_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process speaker diarization using pyannote.audio."""

        # Initialize diarization pipeline
        if self.diarization_pipeline is None:
            self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                self.config["diarization_model"]
            )
            self.logger.info(f"Loaded diarization model: {self.config['diarization_model']}")

        # Apply diarization
        diarization = self.diarization_pipeline(
            audio_path,
            min_speakers=self.config["min_speakers"],
            max_speakers=self.config["max_speakers"],
        )

        # Convert to RTTM turns
        turns = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turn_dict = create_rttm_turn(
                file_id=video_metadata["video_id"],
                start_time=turn.start,
                duration=turn.duration,
                speaker_id=speaker,
                confidence=1.0,  # pyannote doesn't provide confidence by default
            )
            turns.append(turn_dict)

        self.logger.info(f"Diarization complete: {len(turns)} speaker turns")
        return turns

    def _save_results(
        self, results: Dict[str, Any], output_dir: str, video_metadata: Dict[str, Any]
    ):
        """Save results in standard formats."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        video_id = video_metadata["video_id"]

        # Save WebVTT captions
        if results["speech"]:
            webvtt_path = output_path / f"{video_id}_speech.vtt"

            # Create WebVTT file
            vtt = webvtt.WebVTT()
            for caption in results["speech"]:
                vtt.captions.append(caption)

            # Export using native formats module
            export_webvtt(results["speech"], str(webvtt_path))

            # Validate
            validation_result = validate_webvtt(str(webvtt_path))
            if validation_result.is_valid:
                self.logger.info(f"WebVTT validation successful: {webvtt_path}")
            else:
                self.logger.warning(
                    f"WebVTT validation warnings: {', '.join(validation_result.warnings)}"
                )

        # Save RTTM diarization
        if results["diarization"]:
            rttm_path = output_path / f"{video_id}_diarization.rttm"

            export_rttm(results["diarization"], str(rttm_path))

            # Validate
            validation_result = validate_rttm(str(rttm_path))
            if validation_result.is_valid:
                self.logger.info(f"RTTM validation successful: {rttm_path}")
            else:
                self.logger.warning(
                    f"RTTM validation warnings: {', '.join(validation_result.warnings)}"
                )

    def cleanup(self):
        """Cleanup resources."""
        if self.speech_model is not None:
            del self.speech_model
            self.speech_model = None

        if self.diarization_pipeline is not None:
            del self.diarization_pipeline
            self.diarization_pipeline = None
