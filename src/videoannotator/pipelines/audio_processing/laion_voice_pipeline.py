"""LAION Empathic Insight Voice Pipeline.

This pipeline implements voice emotion analysis using LAION's Empathic
Insight Voice models. It utilizes a Whisper-based audio encoder and MLP
classifiers for emotion prediction.
"""

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Import WebVTT exporter
from videoannotator.version import __version__

from .whisper_base_pipeline import WhisperBasePipeline

# Define the emotion taxonomy based on LAION models (same 43 categories as face)
EMOTION_LABELS = {
    # Positive High-Energy
    "elation": "model_Elation_best.pth",
    "amusement": "model_Amusement_best.pth",
    "pleasure_ecstasy": "model_Pleasure_Ecstasy_best.pth",
    "astonishment_surprise": "model_Astonishment_Surprise_best.pth",
    "hope_enthusiasm_optimism": "model_Hope_Enthusiasm_Optimism_best.pth",
    "triumph": "model_Triumph_best.pth",
    "awe": "model_Awe_best.pth",
    "teasing": "model_Teasing_best.pth",
    "interest": "model_Interest_best.pth",
    # Positive Low-Energy
    "relief": "model_Relief_best.pth",
    "contentment": "model_Contentment_best.pth",
    "contemplation": "model_Contemplation_best.pth",
    "pride": "model_Pride_best.pth",
    "thankfulness_gratitude": "model_Thankfulness_Gratitude_best.pth",
    "affection": "model_Affection_best.pth",
    # Negative High-Energy
    "anger": "model_Anger_best.pth",
    "fear": "model_Fear_best.pth",
    "distress": "model_Distress_best.pth",
    "impatience_irritability": "model_Impatience_and_Irritability_best.pth",
    "disgust": "model_Disgust_best.pth",
    "malevolence_malice": "model_Malevolence_Malice_best.pth",
    # Negative Low-Energy
    "helplessness": "model_Helplessness_best.pth",
    "sadness": "model_Sadness_best.pth",
    "emotional_numbness": "model_Emotional_Numbness_best.pth",
    "jealousy_envy": "model_Jealousy_&_Envy_best.pth",
    "embarrassment": "model_Embarrassment_best.pth",
    "contempt": "model_Contempt_best.pth",
    "shame": "model_Shame_best.pth",
    "disappointment": "model_Disappointment_best.pth",
    "doubt": "model_Doubt_best.pth",
    "bitterness": "model_Bitterness_best.pth",
    # Cognitive States
    "concentration": "model_Concentration_best.pth",
    "confusion": "model_Confusion_best.pth",
    # Physical States
    "fatigue_exhaustion": "model_Fatigue_Exhaustion_best.pth",
    "pain": "model_Pain_best.pth",
    "sourness": "model_Sourness_best.pth",
    "intoxication_altered_states": "model_Intoxication_Altered_States_of_Consciousness_best.pth",
    # Longing & Lust
    "sexual_lust": "model_Sexual_Lust_best.pth",
    "longing": "model_Longing_best.pth",
    "infatuation": "model_Infatuation_best.pth",
    # Extra Dimensions
    "dominance": "model_Submissive_vs._Dominant_best.pth",  # Mapped correctly
    "arousal": "model_Arousal_best.pth",
    "emotional_vulnerability": "model_Vulnerable_vs._Emotionally_Detached_best.pth",  # Mapped correctly
}

# Constants for Whisper model
WHISPER_SEQ_LEN = 1500
WHISPER_EMBED_DIM = 768
PROJECTION_DIM = 64
MLP_HIDDEN_DIMS = [64, 32, 16]
MLP_DROPOUTS = [0.0, 0.1, 0.1, 0.1]


# Define the MLP model for emotion classification
class FullEmbeddingMLP(nn.Module):
    """MLP model for emotion classification from full Whisper embeddings.

    This model takes a flattened Whisper embedding, projects it to a
    lower dimension, and then passes it through a series of MLP layers
    to predict emotion scores.
    """

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        projection_dim: int,
        mlp_hidden_dims: list[int],
        mlp_dropout_rates: list[float],
    ):
        """Initialize the MLP model.

        Args:
            seq_len: Sequence length of the Whisper embedding
            embed_dim: Dimension of the Whisper embedding
            projection_dim: Dimension to project the flattened embedding to
            mlp_hidden_dims: List of hidden dimensions for MLP layers
            mlp_dropout_rates: List of dropout rates for each layer (including input)
        """
        super().__init__()
        if len(mlp_dropout_rates) != len(mlp_hidden_dims) + 1:
            raise ValueError(
                f"Dropout rates length error. Expected {len(mlp_hidden_dims) + 1}, got {len(mlp_dropout_rates)}"
            )

        self.flatten = nn.Flatten()
        self.proj = nn.Linear(seq_len * embed_dim, projection_dim)

        layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
        current_dim = projection_dim

        for i, h_dim in enumerate(mlp_hidden_dims):
            layers.extend(
                [
                    nn.Linear(current_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout_rates[i + 1]),
                ]
            )
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP model."""
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        return self.mlp(self.proj(self.flatten(x)))


class LAIONVoicePipeline(WhisperBasePipeline):
    """LAION Empathic Insight Voice Pipeline for audio emotion analysis."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the pipeline with configuration settings."""
        # LAION-specific configuration
        laion_config = {
            # Model configuration
            "model_size": "small",  # "small" or "large"
            "whisper_model": "mkrausio/EmoWhisper-AnS-Small-v0.1",
            "cache_dir": "./models/laion_voice",  # Override base cache_dir
            # LAION-specific audio processing
            "min_segment_duration": 1.0,
            "max_segment_duration": 30.0,
            # Segmentation strategy
            "segmentation_mode": "fixed_interval",  # "fixed_interval", "diarization", "scene_based", "vad"
            "segment_overlap": 0.0,  # Overlap between segments in seconds
            # Integration options
            "enable_diarization": False,  # Simultaneous speaker diarization
            "enable_scene_alignment": False,  # Align with scene boundaries
            # Output configuration
            "include_raw_scores": False,
            "include_transcription": False,  # Optional transcription with emotions
            "top_k_emotions": 5,
        }

        # Merge with user config
        if config:
            laion_config.update(config)

        # Call parent constructor (WhisperBasePipeline handles base Whisper config)
        super().__init__(laion_config)

        self.logger = logging.getLogger(__name__)
        self.classifiers = {}  # LAION emotion classifiers

        # Set CUDA capability for GPU optimization decisions
        self.cuda_capability = self._get_cuda_capability()

    def _get_cuda_capability(self) -> float:
        """Get CUDA capability version for optimization decisions."""
        try:
            if torch.cuda.is_available():
                # Get capability of device 0 by default
                major, minor = torch.cuda.get_device_capability(0)
                return float(f"{major}.{minor}")
            return 0.0
        except Exception:
            return 0.0

    def initialize(self) -> None:
        """Initialize the Whisper model and emotion classifiers."""
        if self.is_initialized:
            return

        self.logger.info(
            f"Initializing LAIONVoicePipeline with model_size: {self.config['model_size']}"
        )

        # Initialize base Whisper pipeline (handles device, model loading, etc.)
        super().initialize()

        # Load LAION emotion classifiers
        self._load_classifiers()

        self.logger.info("LAIONVoicePipeline initialized successfully")

    def get_schema(self) -> dict[str, Any]:
        """Return the output schema for voice emotion analysis."""
        # Basic WebVTT-like schema with emotion attributes
        schema = {
            "format": "webvtt",
            "segment_schema": {
                "start_time": "float",
                "end_time": "float",
                "speaker_id": "str",
                "emotions": "dict[str, {score: float, rank: int}]",
                "transcription": "str",
                "model_info": "dict",
            },
        }
        return schema

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 0.2,  # Default to 5-second segments (0.2 per sec)
        output_dir: str | None = None,
        diarization_results: dict[str, Any] | None = None,
        scene_detection_results: list[dict[str, Any]] | None = None,
        include_transcription: bool = False,
    ) -> list[dict[str, Any]]:
        """Process video to extract audio, segment it, and predict emotions.

        Args:
            video_path: Path to the video file
            start_time: Start time for processing in seconds
            end_time: End time for processing in seconds (None = end of video)
            pps: Predictions per second (controls segment length)
            output_dir: Directory to save output files
            diarization_results: Optional diarization results for speaker-based segmentation
            scene_detection_results: Optional scene detection results for scene-based segmentation
            include_transcription: Whether to include speech transcription in the results

        Returns:
            List of segment dictionaries with emotion predictions
        """
        # Ensure initialization
        if not self.is_initialized:
            self.initialize()

        video_path_obj = Path(video_path)
        self.logger.info(f"Processing video: {video_path_obj.name}")

        try:
            # Extract audio from video using parent method
            audio, sample_rate = self.extract_audio_from_video(video_path_obj)

            # Segment audio based on configuration
            segments = self._segment_audio(
                audio=audio,
                sample_rate=sample_rate,
                pps=pps,
                start_time=start_time,
                end_time=end_time,
                diarization_results=diarization_results,
                scene_detection_results=scene_detection_results,
            )

            # Process each segment
            results = []

            for i, segment in enumerate(segments):
                segment_audio = segment["audio"]
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                speaker_id = segment.get("speaker_id")

                self.logger.debug(
                    f"Processing segment {i + 1}/{len(segments)}: {start_time:.2f}s - {end_time:.2f}s"
                )

                # Get Whisper embedding
                embedding = self._get_whisper_embedding(segment_audio)

                # Skip if embedding failed
                if embedding is None:
                    self.logger.warning(
                        f"Skipping segment {i + 1} due to embedding failure"
                    )
                    continue

                # Predict emotions
                emotion_results = self._predict_emotions(embedding)

                # Get transcription if enabled
                transcription = None
                if include_transcription or self.config.get(
                    "include_transcription", False
                ):
                    try:
                        input_features = self.whisper_processor(
                            segment_audio,
                            sampling_rate=sample_rate,
                            return_tensors="pt",
                        ).input_features.to(self.device)

                        # Generate transcription
                        with torch.no_grad():
                            predicted_ids = self.whisper_model.generate(input_features)
                            transcription = self.whisper_processor.batch_decode(
                                predicted_ids, skip_special_tokens=True
                            )[0].strip()
                    except Exception as e:
                        self.logger.warning(f"Transcription failed: {e}")

                # Build result entry
                result = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "emotions": emotion_results.get("emotions", {}),
                    "model_info": {
                        "model_size": self.config["model_size"],
                        "segmentation_mode": self.config["segmentation_mode"],
                    },
                }

                # Add optional fields
                if speaker_id:
                    result["speaker_id"] = speaker_id

                if transcription:
                    result["transcription"] = transcription

                if (
                    self.config.get("include_raw_scores", False)
                    and "raw_scores" in emotion_results
                ):
                    result["raw_scores"] = emotion_results["raw_scores"]

                results.append(result)

            # Export results if requested
            if output_dir and results:
                output_path = Path(output_dir) / f"{video_path_obj.stem}_voice_emotions"

                # Export as JSON
                json_path = f"{output_path}.json"
                try:
                    with open(json_path, "w") as f:
                        import json

                        json.dump(
                            {
                                "segments": results,
                                "metadata": {
                                    "source": video_path_obj.name,
                                    "pipeline": self.name,
                                    "model_size": self.config["model_size"],
                                    "segmentation_mode": self.config[
                                        "segmentation_mode"
                                    ],
                                    "total_segments": len(results),
                                },
                            },
                            f,
                            indent=2,
                        )
                    self.logger.info(f"Exported JSON results to {json_path}")
                except Exception as e:
                    self.logger.error(f"Failed to export JSON: {e}")

                # Export as WebVTT if transcription is included
                if self.config.get("include_transcription", False):
                    vtt_path = f"{output_path}.vtt"
                    try:
                        with open(vtt_path, "w") as f:
                            f.write("WEBVTT\n")
                            f.write("NOTE Generated by LAION Voice Pipeline\n\n")

                            for result in results:
                                start = self._format_timestamp(result["start_time"])
                                end = self._format_timestamp(result["end_time"])

                                f.write(f"{start} --> {end}\n")

                                if "speaker_id" in result:
                                    f.write(f"<v {result['speaker_id']}>")

                                if "transcription" in result:
                                    f.write(f"{result['transcription']}\n")

                                # Add top emotions
                                if result["emotions"]:
                                    emotion_str = ", ".join(
                                        [
                                            f"{emotion}({data['score']:.2f})"
                                            for emotion, data in result[
                                                "emotions"
                                            ].items()
                                        ]
                                    )
                                    f.write(f"EMOTIONS: {emotion_str}\n")

                                f.write("\n")

                        self.logger.info(f"Exported WebVTT results to {vtt_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to export WebVTT: {e}")

            self.logger.info(f"LAIONVoicePipeline complete: {len(results)} segments")
            return results

        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            raise RuntimeError(f"Failed to process video: {e}")

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as WebVTT timestamp (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def _load_classifiers(self) -> None:
        """Load pre-trained emotion classifier models for each label."""
        model_size = self.config.get("model_size", "small")
        cache_dir = Path(self.config.get("cache_dir", "./models/laion_voice"))

        # Create directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Repository mapping
        repo_id = (
            "laion/Empathic-Insight-Voice-Small"
            if model_size == "small"
            else "laion/Empathic-Insight-Voice-Large"
        )

        self.logger.info(f"Loading emotion classifiers from {repo_id}")

        # Check CUDA capability for optimization decisions
        if self.device.type == "cuda":
            try:
                cuda_capability = torch.cuda.get_device_capability(
                    self.device.index if hasattr(self.device, "index") else 0
                )
                cuda_capability_major = (
                    cuda_capability[0]
                    if isinstance(cuda_capability, tuple)
                    else cuda_capability
                )
                if cuda_capability_major >= 7:
                    self.logger.info(
                        f"CUDA capability {cuda_capability_major}.{cuda_capability[1] if isinstance(cuda_capability, tuple) else 'x'} detected - torch.compile optimization available"
                    )
                else:
                    self.logger.info(
                        f"CUDA capability {cuda_capability_major}.{cuda_capability[1] if isinstance(cuda_capability, tuple) else 'x'} detected - using standard models (torch.compile requires >=7.0)"
                    )
            except Exception as e:
                self.logger.debug(f"Could not determine CUDA capability: {e}")

        loaded_count = 0
        for label, filename in EMOTION_LABELS.items():
            # Try to download the file if it doesn't exist locally
            local_path = cache_dir / filename

            if not local_path.exists():
                try:
                    self.logger.info(f"Downloading classifier for emotion: {label}")
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(
                            cache_dir.parent
                        ),  # Parent directory for HF cache structure
                        local_dir=str(cache_dir),  # Local directory to store file
                        local_dir_use_symlinks=False,
                    )
                    self.logger.info(f"Downloaded {filename} to {downloaded_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to download classifier {label}: {e}")
                    continue

            # Load classifier if available
            if local_path.exists():
                try:
                    # Initialize MLP model
                    model_instance = FullEmbeddingMLP(
                        seq_len=WHISPER_SEQ_LEN,
                        embed_dim=WHISPER_EMBED_DIM,
                        projection_dim=PROJECTION_DIM,
                        mlp_hidden_dims=MLP_HIDDEN_DIMS,
                        mlp_dropout_rates=MLP_DROPOUTS,
                    )

                    # Load state dict
                    state_dict = torch.load(local_path, map_location="cpu")

                    # Check if state dict needs stripping (in case it was compiled)
                    needs_stripping = any(
                        k.startswith("_orig_mod.") for k in state_dict.keys()
                    )
                    if needs_stripping:
                        stripped_state_dict = {
                            k[len("_orig_mod.") :]
                            if k.startswith("_orig_mod.")
                            else k: v
                            for k, v in state_dict.items()
                        }
                        state_dict = stripped_state_dict

                    model_instance.load_state_dict(state_dict)
                    model_instance.eval()

                    # Move to device and match Whisper model dtype if needed
                    # Handle meta tensor properly for newer PyTorch versions
                    try:
                        model_instance = model_instance.to(self.device)
                    except RuntimeError as e:
                        if "meta tensor" in str(e):
                            model_instance = model_instance.to_empty(device=self.device)
                        else:
                            raise

                    # If parent Whisper model uses FP16, convert classifier to FP16 too
                    if hasattr(self, "whisper_model") and hasattr(
                        self.whisper_model, "dtype"
                    ):
                        if self.whisper_model.dtype == torch.float16:
                            model_instance = model_instance.half()
                            self.logger.debug(f"Converted {label} classifier to FP16")

                    # Try to use torch.compile for GPU acceleration (only on newer GPUs)
                    if (
                        self.device.type == "cuda"
                        and hasattr(torch, "compile")
                        and torch.__version__ >= "2.0.0"
                    ):
                        # Check CUDA capability for triton compatibility
                        cuda_capability = torch.cuda.get_device_capability(
                            self.device.index if hasattr(self.device, "index") else 0
                        )
                        cuda_capability_major = (
                            cuda_capability[0]
                            if isinstance(cuda_capability, tuple)
                            else cuda_capability
                        )

                        # Only use torch.compile on GPUs with CUDA capability >= 7.0 (for triton support)
                        if cuda_capability_major >= 7:
                            try:
                                model_instance = torch.compile(
                                    model_instance, mode="reduce-overhead"
                                )
                                self.logger.debug(
                                    f"Using compiled model for {label} (CUDA capability {cuda_capability_major}.{cuda_capability[1] if isinstance(cuda_capability, tuple) else 'x'})"
                                )
                            except Exception as e_compile:
                                self.logger.warning(
                                    f"torch.compile failed for {label}: {e_compile}"
                                )
                        else:
                            self.logger.debug(
                                f"Skipping torch.compile for {label} (CUDA capability {cuda_capability_major}.{cuda_capability[1] if isinstance(cuda_capability, tuple) else 'x'} < 7.0, triton not supported)"
                            )

                    # Store in classifiers dictionary
                    self.classifiers[label] = model_instance
                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load classifier {label}: {e}")
                    continue

        self.logger.info(
            f"Loaded {loaded_count}/{len(EMOTION_LABELS)} emotion classifiers"
        )

    def _segment_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        pps: float,
        start_time: float = 0.0,
        end_time: float | None = None,
        diarization_results: dict[str, Any] | None = None,
        scene_detection_results: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Segment audio based on the selected segmentation strategy.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            pps: Predictions per second (used for fixed interval segmentation)
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of audio)
            diarization_results: Optional diarization results for speaker-based segmentation
            scene_detection_results: Optional scene detection results for scene-based segmentation

        Returns:
            List of segment dictionaries with audio data and metadata
        """
        segments = []
        audio_duration = len(audio) / sample_rate

        if end_time is None or end_time > audio_duration:
            end_time = audio_duration

        segmentation_mode = self.config["segmentation_mode"]
        self.logger.info(f"Segmenting audio using '{segmentation_mode}' mode")

        min_segment_duration = self.config["min_segment_duration"]
        max_segment_duration = self.config["max_segment_duration"]
        segment_overlap = self.config["segment_overlap"]

        # Fixed interval segmentation
        if segmentation_mode == "fixed_interval":
            # Calculate segment duration based on pps
            if pps <= 0:
                # If pps is 0 or negative, use one segment for the entire audio
                segment_duration = end_time - start_time
            else:
                # Normal case: segment duration is 1/pps
                segment_duration = 1.0 / pps

            # Ensure segment duration is within bounds
            segment_duration = max(
                min_segment_duration, min(segment_duration, max_segment_duration)
            )

            self.logger.info(
                f"Fixed interval segmentation: {segment_duration:.2f}s segments"
            )

            # Create segments
            current_start = start_time
            while current_start < end_time:
                current_end = min(current_start + segment_duration, end_time)

                # Skip segments that are too short
                if current_end - current_start < min_segment_duration:
                    break

                # Calculate audio indices
                start_idx = int(current_start * sample_rate)
                end_idx = int(current_end * sample_rate)

                if start_idx >= len(audio) or start_idx >= end_idx:
                    break

                # Create segment
                segment = {
                    "start_time": current_start,
                    "end_time": current_end,
                    "audio": audio[start_idx:end_idx],
                    "speaker_id": None,  # No speaker ID for fixed interval
                }

                segments.append(segment)

                # Move to next segment with potential overlap
                current_start += segment_duration - segment_overlap

        # Diarization-based segmentation
        elif segmentation_mode == "diarization" and diarization_results:
            self.logger.info("Using diarization-based segmentation")

            # Extract speaker segments from diarization results
            if "segments" in diarization_results:
                speaker_segments = diarization_results["segments"]

                for segment in speaker_segments:
                    seg_start = segment.get("start_time", 0)
                    seg_end = segment.get("end_time", 0)
                    speaker_id = segment.get("speaker", "unknown")

                    # Skip segments outside our time range
                    if seg_end <= start_time or seg_start >= end_time:
                        continue

                    # Adjust segment to our time range
                    seg_start = max(seg_start, start_time)
                    seg_end = min(seg_end, end_time)

                    # Skip segments that are too short
                    if seg_end - seg_start < min_segment_duration:
                        continue

                    # Calculate audio indices
                    start_idx = int(seg_start * sample_rate)
                    end_idx = int(seg_end * sample_rate)

                    if start_idx >= len(audio) or start_idx >= end_idx:
                        continue

                    # Create segment
                    segment = {
                        "start_time": seg_start,
                        "end_time": seg_end,
                        "audio": audio[start_idx:end_idx],
                        "speaker_id": speaker_id,
                    }

                    segments.append(segment)
            else:
                self.logger.warning("No segments found in diarization results")
                # Fall back to fixed interval segmentation
                return self._segment_audio(
                    audio, sample_rate, pps, start_time, end_time
                )

        # Scene-based segmentation
        elif segmentation_mode == "scene_based" and scene_detection_results:
            self.logger.info("Using scene-based segmentation")

            # Extract scene transitions
            scene_transitions = []
            for scene in scene_detection_results:
                if "start_time" in scene:
                    scene_transitions.append(scene["start_time"])

            # Add end time as final transition
            scene_transitions.append(end_time)

            # Sort transitions
            scene_transitions.sort()

            # Create segments based on scene transitions
            prev_transition = start_time
            for transition in scene_transitions:
                if transition <= prev_transition or transition <= start_time:
                    continue

                seg_start = prev_transition
                seg_end = transition

                # Skip segments that are too short
                if seg_end - seg_start < min_segment_duration:
                    prev_transition = transition
                    continue

                # Ensure segments are not too long
                if seg_end - seg_start > max_segment_duration:
                    # Split into multiple segments
                    subseg_start = seg_start
                    while subseg_start < seg_end:
                        subseg_end = min(subseg_start + max_segment_duration, seg_end)

                        # Calculate audio indices
                        start_idx = int(subseg_start * sample_rate)
                        end_idx = int(subseg_end * sample_rate)

                        if start_idx >= len(audio) or start_idx >= end_idx:
                            break

                        # Create segment
                        segment = {
                            "start_time": subseg_start,
                            "end_time": subseg_end,
                            "audio": audio[start_idx:end_idx],
                            "speaker_id": None,  # No speaker ID for scene-based
                        }

                        segments.append(segment)
                        subseg_start = subseg_end
                else:
                    # Calculate audio indices
                    start_idx = int(seg_start * sample_rate)
                    end_idx = int(seg_end * sample_rate)

                    if start_idx >= len(audio) or start_idx >= end_idx:
                        prev_transition = transition
                        continue

                    # Create segment
                    segment = {
                        "start_time": seg_start,
                        "end_time": seg_end,
                        "audio": audio[start_idx:end_idx],
                        "speaker_id": None,  # No speaker ID for scene-based
                    }

                    segments.append(segment)

                prev_transition = transition

        # Voice Activity Detection (VAD) - simplified approach
        elif segmentation_mode == "vad":
            self.logger.info("Using Voice Activity Detection (VAD) segmentation")

            # Simple energy-based VAD
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)  # 10ms hop

            # Calculate energy
            energy = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length
            )[0]

            # Simple threshold (could be improved with proper VAD)
            threshold = np.mean(energy) * 0.5

            # Find speech segments
            speech_frames = energy > threshold

            # Convert to time segments
            in_speech = False
            speech_start = 0
            speech_segments = []

            for i, is_speech in enumerate(speech_frames):
                frame_time = i * hop_length / sample_rate

                if is_speech and not in_speech:
                    # Speech start
                    in_speech = True
                    speech_start = frame_time

                elif not is_speech and in_speech:
                    # Speech end
                    in_speech = False
                    speech_end = frame_time

                    # Only add if segment is long enough
                    if speech_end - speech_start >= min_segment_duration:
                        speech_segments.append((speech_start, speech_end))

            # Add final segment if still in speech
            if in_speech:
                speech_end = len(speech_frames) * hop_length / sample_rate
                if speech_end - speech_start >= min_segment_duration:
                    speech_segments.append((speech_start, speech_end))

            # Create segments from VAD results
            for seg_start, seg_end in speech_segments:
                # Skip segments outside our time range
                if seg_end <= start_time or seg_start >= end_time:
                    continue

                # Adjust segment to our time range
                seg_start = max(seg_start, start_time)
                seg_end = min(seg_end, end_time)

                # Ensure segments are not too long
                if seg_end - seg_start > max_segment_duration:
                    # Split into multiple segments
                    subseg_start = seg_start
                    while subseg_start < seg_end:
                        subseg_end = min(subseg_start + max_segment_duration, seg_end)

                        # Calculate audio indices
                        start_idx = int(subseg_start * sample_rate)
                        end_idx = int(subseg_end * sample_rate)

                        if start_idx >= len(audio) or start_idx >= end_idx:
                            break

                        # Create segment
                        segment = {
                            "start_time": subseg_start,
                            "end_time": subseg_end,
                            "audio": audio[start_idx:end_idx],
                            "speaker_id": None,  # No speaker ID for VAD
                        }

                        segments.append(segment)
                        subseg_start = subseg_end
                else:
                    # Calculate audio indices
                    start_idx = int(seg_start * sample_rate)
                    end_idx = int(seg_end * sample_rate)

                    if start_idx >= len(audio) or start_idx >= end_idx:
                        continue

                    # Create segment
                    segment = {
                        "start_time": seg_start,
                        "end_time": seg_end,
                        "audio": audio[start_idx:end_idx],
                        "speaker_id": None,  # No speaker ID for VAD
                    }

                    segments.append(segment)

        else:
            # Default to fixed interval if no other method works
            self.logger.warning(
                f"Unsupported segmentation mode '{segmentation_mode}' or missing required data. Falling back to fixed interval."
            )
            return self._segment_audio(audio, sample_rate, pps, start_time, end_time)

        self.logger.info(f"Created {len(segments)} audio segments")
        return segments

    @torch.no_grad()
    def _get_whisper_embedding(self, audio_segment: np.ndarray) -> torch.Tensor | None:
        """Get Whisper embeddings for an audio segment using parent pipeline.

        Args:
            audio_segment: Audio waveform as numpy array

        Returns:
            Whisper embedding tensor padded/trimmed to WHISPER_SEQ_LEN
        """
        try:
            # Use parent's get_whisper_embedding method with padding
            embedding = self.get_whisper_embedding(
                audio_segment, pad_or_trim=True, target_seq_len=WHISPER_SEQ_LEN
            )
            return embedding

        except Exception as e:
            self.logger.error(f"Error getting Whisper embedding: {e}")
            return None

    def _predict_emotions(self, embedding: torch.Tensor) -> dict[str, Any]:
        """Predict emotions from audio embedding using MLP classifiers.

        Args:
            embedding: Whisper embedding tensor

        Returns:
            Dictionary of emotion predictions with scores and ranks
        """
        # Skip prediction if embedding is None
        if embedding is None:
            return {"emotions": {}}

        # Get raw scores from classifiers
        raw_scores: dict[str, float] = {}

        for label, clf in self.classifiers.items():
            try:
                with torch.no_grad():
                    # Create a working copy of the embedding for this classifier
                    working_embedding = embedding

                    # Ensure embedding and classifier are on same device and dtype
                    classifier_param = clf.parameters().__next__()

                    if working_embedding.device != classifier_param.device:
                        working_embedding = working_embedding.to(
                            classifier_param.device
                        )

                    # Ensure embedding dtype matches classifier dtype
                    if working_embedding.dtype != classifier_param.dtype:
                        working_embedding = working_embedding.to(
                            dtype=classifier_param.dtype
                        )

                    pred = clf(working_embedding).cpu().item()
                raw_scores[label] = float(pred)
            except Exception as e:
                # Check if this is a triton/torch.compile related error
                error_msg = str(e).lower()
                if (
                    "triton" in error_msg
                    or "cuda capability" in error_msg
                    or "compile" in error_msg
                ):
                    self.logger.warning(
                        f"Triton/compile error for {label} (older GPU detected), skipping this emotion"
                    )
                    self.logger.debug(f"Full error: {e}")
                else:
                    self.logger.warning(f"Failed to predict {label}: {e}")
                    self.logger.debug(
                        f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}, device: {embedding.device}"
                    )
                    if hasattr(clf, "parameters"):
                        try:
                            clf_param = clf.parameters().__next__()
                            self.logger.debug(
                                f"Classifier dtype: {clf_param.dtype}, device: {clf_param.device}"
                            )
                        except:
                            pass

        # Apply softmax across all emotions to get proper probability distribution
        if raw_scores:
            scores_array = np.array(list(raw_scores.values()))
            max_score = np.max(scores_array)
            exp_scores = np.exp(scores_array - max_score)
            softmax_scores = exp_scores / np.sum(exp_scores)

            # Create scored emotions dictionary with rank
            emotions = {}
            for i, (label, score) in enumerate(
                zip(raw_scores.keys(), softmax_scores, strict=False)
            ):
                emotions[label] = {"score": float(score), "rank": i + 1}

            # Sort by score and limit to top_k
            top_k = self.config.get("top_k_emotions", 5)
            sorted_emotions = dict(
                sorted(emotions.items(), key=lambda x: x[1]["score"], reverse=True)[
                    :top_k
                ]
            )

            # Re-rank after sorting and limiting
            for i, (label, data) in enumerate(sorted_emotions.items()):
                data["rank"] = i + 1

            return {
                "emotions": sorted_emotions,
                "raw_scores": raw_scores
                if self.config.get("include_raw_scores", False)
                else None,
            }

        return {"emotions": {}}

    def cleanup(self) -> None:
        """Clean up resources used by the pipeline."""
        try:
            # Clean up LAION-specific classifiers
            if hasattr(self, "device") and self.device and self.device.type == "cuda":
                for label in list(self.classifiers.keys()):
                    if self.classifiers[label] is not None:
                        self.classifiers[label].cpu()

            # Clear classifiers
            self.classifiers = {}

            # Call parent cleanup for Whisper model cleanup
            super().cleanup()

            self.logger.info("LAIONVoicePipeline resources cleaned up")

        except Exception as e:
            self.logger.error(f"Error cleaning up LAION resources: {e}")
            # Still call parent cleanup to ensure base resources are freed
            try:
                super().cleanup()
            except:
                pass

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get information about the LAION Voice Pipeline."""
        return {
            "name": "LAIONVoicePipeline",
            "version": __version__,
            "capabilities": {
                "audio_processing": True,
                "emotion_analysis": True,
                "whisper_available": True,
            },
            "models": {
                "whisper_base": self.config.get(
                    "whisper_model", "mkrausio/EmoWhisper-AnS-Small-v0.1"
                ),
                "classifier_type": "FullEmbeddingMLP",
                "variant": self.config.get(
                    "model_size", "small"
                ),  # Use model_size instead of model_variant
                "emotion_labels": list(EMOTION_LABELS.keys()),
            },
            "config": {
                "segmentation_strategy": self.config.get(
                    "segmentation_mode", "fixed_interval"
                ),
                "segment_length": self.config.get("segment_length", 5.0),
                "min_segment_length": self.config.get("min_segment_duration", 1.0),
                "top_k_emotions": self.config.get("top_k_emotions", 5),
                "use_gpu": self.config.get("use_gpu", torch.cuda.is_available()),
            },
            "requirements": {
                "torch_available": True,
                "librosa_available": True,
                "cuda_available": torch.cuda.is_available(),
            },
        }
