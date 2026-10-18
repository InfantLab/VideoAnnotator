"""Speech Recognition Pipeline for VideoAnnotator.

Transcribes speech with Whisper on the faster-whisper runtime. It's designed
to be separable from other audio processing functionality.
"""

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from videoannotator.config_env import load_env_file
from videoannotator.version import __version__

from .native_files import write_webvtt
from .whisper_base_pipeline import WhisperBasePipeline, faster_whisper_available

load_env_file()

logger = logging.getLogger(__name__)

WHISPER_AVAILABLE = faster_whisper_available()
if not WHISPER_AVAILABLE:
    logger.warning("faster-whisper not available. Speech recognition will be disabled.")


def _suppress_tokens(value: Any) -> list[int] | None:
    """Our setting is a comma-separated string; faster-whisper wants ints."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, list | tuple):
        return [int(v) for v in value]
    return [int(v) for v in str(value).split(",") if v.strip()]


class SpeechPipeline(WhisperBasePipeline):
    """Speech recognition pipeline using Whisper (faster-whisper runtime).

    This implementation extends WhisperBasePipeline to leverage the
    shared Whisper functionality while focusing specifically on
    transcription.
    """

    @property
    def output_format(self) -> str:
        """Return the output format for this pipeline."""
        return "webvtt"

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the speech pipeline with Whisper configuration defaults."""
        # Speech-specific config defaults
        speech_config = {
            "whisper_model": "base",
            "language": None,
            "task": "transcribe",
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "patience": 1.0,
            "length_penalty": 1.0,
            "word_timestamps": True,
            "prepend_punctuations": '"\'"¿([{-',
            "append_punctuations": '"\'..,!?:")]},',
            "suppress_tokens": None,
            "min_segment_duration": 1.0,
        }

        # Update with user-provided config
        if config:
            speech_config.update(config)

        # Convert legacy "model" key to "whisper_model" if present
        if config and "model" in config and "whisper_model" not in config:
            speech_config["whisper_model"] = config["model"]

        # Initialize base pipeline with merged config
        super().__init__(speech_config)
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Refuse a translation the model can't do, then load the model."""
        model = str(self.config.get("whisper_model", ""))
        if self.config.get("task") == "translate" and "turbo" in model:
            raise ValueError(
                f"{model} can't translate: it returns the original language. "
                "Set whisper_model to large-v3 or medium for translation."
            )
        super().initialize()

    def process(
        self,
        video_path: str | Path,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 0.0,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Process video and return speech recognition results.

        Args:
            video_path: Path to video file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            pps: Predictions per second
            output_dir: Optional output directory

        Returns:
            List containing SpeechRecognition result
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Input file not found: {video_path}")

        if not self.is_initialized:
            self.initialize()

        audio, _sample_rate = self.extract_audio_from_video(video_path)
        result = self.transcribe_audio(audio)
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            write_webvtt([result], out / f"{video_path.stem}_speech_recognition.vtt")
        return [result]

    def _transcribe_options(self) -> dict[str, Any]:
        """Our settings as faster-whisper's `transcribe()` arguments."""
        options: dict[str, Any] = {
            "language": self.config.get("language"),
            "task": self.config.get("task", "transcribe"),
            "beam_size": self.config.get("beam_size", 5),
            "best_of": self.config.get("best_of", 5),
            "temperature": self.config.get("temperature", 0.0),
            "patience": self.config.get("patience", 1.0),
            "length_penalty": self.config.get("length_penalty", 1.0),
            "word_timestamps": self.config.get("word_timestamps", True),
            "prepend_punctuations": self.config.get(
                "prepend_punctuations", '"\'"¿([{-'
            ),
            "append_punctuations": self.config.get(
                "append_punctuations", '"\'..,!?:")],'
            ),
            # openai-whisper had no voice-activity filter; keep transcripts comparable.
            "vad_filter": False,
        }
        suppress = _suppress_tokens(self.config.get("suppress_tokens"))
        if suppress is not None:
            options["suppress_tokens"] = suppress
        return options

    def transcribe_audio(self, audio: str | Path | np.ndarray) -> dict[str, Any]:
        """Perform speech recognition on audio.

        Args:
            audio: Path to audio file or audio waveform

        Returns:
            SpeechRecognition object

        Raises:
            RuntimeError: if the model isn't loaded or transcription fails, so
                the job reports this pipeline as failed rather than completed
                with an empty transcript.
        """
        if not self.is_initialized:
            self.initialize()

        if not self.whisper_model:
            raise RuntimeError("Whisper model not initialized")

        if isinstance(audio, str | Path):
            audio_path = Path(audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            video_id = audio_path.stem
            waveform, _sr = librosa.load(
                audio_path, sr=self.config["sample_rate"], mono=True
            )
        elif isinstance(audio, np.ndarray):
            video_id = "audio_segment"
            waveform = audio
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")

        try:
            self.logger.info("Performing speech recognition")
            segments, info = self.whisper_model.transcribe(
                waveform.astype(np.float32), **self._transcribe_options()
            )
            # A generator: decoding happens here, so errors are raised here.
            return self._process_transcription_result(list(segments), info, video_id)
        except Exception as e:
            raise RuntimeError(f"Speech recognition failed: {e}") from e

    def _process_transcription_result(
        self, segments: list[Any], info: Any, video_id: str
    ) -> dict[str, Any]:
        """Turn faster-whisper's segments into the result dict openai-whisper's
        output used to fill, so the WebVTT and every reader are unchanged.

        Args:
            segments: faster-whisper `Segment`s
            info: faster-whisper `TranscriptionInfo`
            video_id: Identifier for the video/audio

        Returns:
            Processed transcription result
        """
        word_timestamps = [
            {
                "word": word.word.strip(),
                "start": float(word.start),
                "end": float(word.end),
                "confidence": float(word.probability),
            }
            for segment in segments
            for word in segment.words or []
        ]
        segment_dicts = [
            {
                "id": segment.id,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "tokens": list(segment.tokens),
                "temperature": float(segment.temperature or 0.0),
                "avg_logprob": float(segment.avg_logprob),
                "compression_ratio": float(segment.compression_ratio),
                "no_speech_prob": float(segment.no_speech_prob),
            }
            for segment in segments
        ]
        end = segment_dicts[-1]["end"] if segment_dicts else 0.0

        speech_result = {
            "video_id": video_id,
            "timestamp": 0.0,  # Start time
            "transcript": "".join(s.text for s in segments).strip(),
            "language": getattr(info, "language", None) or "unknown",
            "confidence": 0.0,  # Whisper doesn't provide overall confidence
            "words": word_timestamps,
            "metadata": {
                "model": self.config["whisper_model"],
                "task": self.config.get("task", "transcribe"),
                "num_segments": len(segment_dicts),
                "num_words": len(word_timestamps),
                "start_time": 0.0,
                "end_time": end,
                "total_duration": end,
                "segments": segment_dicts,  # Store segment details in metadata
            },
        }

        self.logger.info(
            f"Speech recognition completed. "
            f"Transcribed {len(segment_dicts)} segments with {len(word_timestamps)} words"
        )
        return speech_result

    def get_schema(self) -> dict[str, Any]:
        """Get the output schema for this pipeline."""
        return {
            "type": "speech_recognition",
            "description": "Speech recognition results with word and segment timestamps",
            "properties": {
                "transcript": {
                    "type": "string",
                    "description": "Full transcribed text",
                },
                "language": {
                    "type": "string",
                    "description": "Detected or specified language",
                },
                "confidence": {
                    "type": "number",
                    "description": "Overall confidence score",
                },
                "words": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "confidence": {"type": "number"},
                        },
                    },
                    "description": "Word-level timestamps with confidence",
                },
            },
        }

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get information about the speech recognition pipeline."""
        return {
            "name": "SpeechPipeline",
            "version": __version__,
            "capabilities": {
                "speech_recognition": WHISPER_AVAILABLE,
                "word_timestamps": True,
                "segment_timestamps": True,
                "language_detection": True,
                "translation": True,
            },
            "models": {
                "whisper_model": self.config["whisper_model"]
                if WHISPER_AVAILABLE
                else None,
                "runtime": "faster-whisper",
            },
            "config": {
                "model_name": self.config["whisper_model"],
                "language": self.config.get("language"),
                "task": self.config.get("task", "transcribe"),
                "word_timestamps": self.config.get("word_timestamps", True),
                "device": self.config.get("device", "auto"),
                "use_fp16": self.config.get("use_fp16", True),
                "compute_type": self.config.get("compute_type", "default"),
            },
            "requirements": {
                "whisper_available": WHISPER_AVAILABLE,
                "cuda_available": torch.cuda.is_available(),
            },
        }
