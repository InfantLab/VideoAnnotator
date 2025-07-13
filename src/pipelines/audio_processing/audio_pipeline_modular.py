"""
Modular audio processing pipeline coordinator.

This pipeline coordinates multiple audio processing pipelines and returns separate data streams.
"""

import logging
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import librosa

from ..base_pipeline import BasePipeline
from .speech_pipeline import SpeechPipeline  
from .diarization_pipeline import DiarizationPipeline
# Future pipelines can be imported here:
# from .emotion_pipeline import EmotionPipeline
# from .f0_pipeline import F0Pipeline
# from .timbre_pipeline import TimbrePipeline

from ...exporters.native_formats import (
    export_webvtt_captions,
    export_rttm_diarization,
)


class AudioPipelineModular(BasePipeline):
    """
    Modular audio processing pipeline coordinator.

    Manages multiple audio pipelines and returns separate timestamped data streams.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            # Audio processing settings
            "sample_rate": 16000,
            "normalize_audio": True,
            
            # Pipeline configurations
            "pipelines": {
                "speech_recognition": {
                    "enabled": True,
                    "model": "base",
                    "language": None,  # Auto-detect
                    "min_segment_duration": 1.0,
                    "word_timestamps": True,
                },
                "speaker_diarization": {
                    "enabled": True,
                    "model": "pyannote/speaker-diarization-3.1",
                    "min_speakers": 1,
                    "max_speakers": 10,
                    "use_auth_token": True,
                },
                # Future pipeline configs:
                # "emotion_recognition": {"enabled": False},
                # "f0_extraction": {"enabled": False},
                # "timbre_analysis": {"enabled": False},
            }
        }
        if config:
            # Deep merge the nested config
            for key, value in config.items():
                if key == "pipelines" and isinstance(value, dict):
                    for pipeline_name, pipeline_config in value.items():
                        if pipeline_name in default_config["pipelines"]:
                            default_config["pipelines"][pipeline_name].update(pipeline_config)
                        else:
                            default_config["pipelines"][pipeline_name] = pipeline_config
                else:
                    default_config[key] = value
        
        super().__init__(default_config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline instances
        self.audio_pipelines = {}
        self._setup_pipelines()

    def _setup_pipelines(self) -> None:
        """Initialize individual audio pipelines based on configuration."""
        pipeline_configs = self.config.get("pipelines", {})
        
        # Speech recognition pipeline
        if pipeline_configs.get("speech_recognition", {}).get("enabled", False):
            speech_config = pipeline_configs["speech_recognition"]
            self.audio_pipelines["speech_recognition"] = SpeechPipeline(speech_config)
            self.logger.info("SpeechPipeline configured")
        
        # Speaker diarization pipeline
        if pipeline_configs.get("speaker_diarization", {}).get("enabled", False):
            diarization_config = pipeline_configs["speaker_diarization"]
            self.audio_pipelines["speaker_diarization"] = DiarizationPipeline(diarization_config)
            self.logger.info("DiarizationPipeline configured")
        
        # Future pipelines can be added here:
        # if pipeline_configs.get("emotion_recognition", {}).get("enabled", False):
        #     emotion_config = pipeline_configs["emotion_recognition"]
        #     self.audio_pipelines["emotion_recognition"] = EmotionPipeline(emotion_config)

    def initialize(self) -> None:
        """Initialize all enabled audio pipelines."""
        if self.is_initialized:
            return

        self.logger.info("Initializing AudioPipeline coordinator...")
        
        # Initialize all configured pipelines
        failed_pipelines = []
        for pipeline_name, pipeline in self.audio_pipelines.items():
            try:
                self.logger.info(f"Initializing {pipeline_name}")
                pipeline.initialize()
                self.logger.info(f"{pipeline_name} initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize {pipeline_name}: {e}")
                # Mark for removal instead of deleting during iteration
                failed_pipelines.append(pipeline_name)
        
        # Remove failed pipelines after iteration
        for pipeline_name in failed_pipelines:
            del self.audio_pipelines[pipeline_name]
                
        self.is_initialized = True
        self.logger.info(f"AudioPipeline coordinator initialized with {len(self.audio_pipelines)} active pipelines")

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process video through all enabled audio pipelines.

        Returns:
            List of results from all pipelines, each with separate timestamped data
        """
        if not self.is_initialized:
            self.initialize()

        # Extract audio from video
        audio_path = self._extract_audio(video_path, start_time, end_time)
        metadata = self._get_video_metadata(video_path)

        try:
            results = []
            
            # Process through each enabled pipeline
            for pipeline_name, pipeline in self.audio_pipelines.items():
                try:
                    self.logger.info(f"Processing with {pipeline_name}")
                    # Use BasePipeline interface - pass audio_path as video_path for audio pipelines
                    pipeline_results = pipeline.process(
                        video_path=audio_path,
                        start_time=start_time,
                        end_time=end_time,
                        output_dir=output_dir
                    )
                    
                    # Add pipeline results as separate stream
                    pipeline_data = {
                        "pipeline": pipeline_name,
                        "format": pipeline.output_format,
                        "data": pipeline_results,
                        "metadata": {
                            "pipeline_name": pipeline_name,
                            "output_format": pipeline.output_format,
                            "processed_segments": len(pipeline_results)
                        }
                    }
                    results.append(pipeline_data)
                    
                    self.logger.info(f"{pipeline_name} completed: {len(pipeline_results)} segments")
                    
                except Exception as e:
                    self.logger.error(f"Error in {pipeline_name}: {e}")
                    # Continue with other pipelines even if one fails
                    continue

            # Save results if output directory specified
            if output_dir:
                self._save_results(results, output_dir, metadata)

            # Return all pipeline results as separate streams
            return results

        finally:
            # Cleanup temporary audio file
            if audio_path != video_path:
                Path(audio_path).unlink(missing_ok=True)

    def _extract_audio(
        self, video_path: str, start_time: float = 0.0, end_time: Optional[float] = None
    ) -> str:
        """Extract audio from video file."""
        # Create temporary file for audio
        temp_dir = Path(tempfile.gettempdir())
        audio_path = temp_dir / f"{Path(video_path).stem}_temp_audio.wav"

        # Build ffmpeg command
        cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", str(start_time)]

        if end_time is not None:
            cmd.extend(["-to", str(end_time)])

        cmd.extend([
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", str(self.config["sample_rate"]),
            "-ac", "1",  # Mono
            str(audio_path),
        ])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"Audio extracted to: {audio_path}")
            return str(audio_path)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to extract audio: {e}")
            return video_path

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata."""
        try:
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

    def _save_results(
        self, results: List[Dict[str, Any]], output_dir: str, metadata: Dict[str, Any]
    ) -> None:
        """Save results from all pipelines in their native formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        video_id = metadata["video_id"]

        for pipeline_result in results:
            pipeline_name = pipeline_result["pipeline"]
            output_format = pipeline_result["format"]
            data = pipeline_result["data"]

            try:
                if output_format == "webvtt" and data:
                    # Save speech recognition as WebVTT
                    webvtt_path = output_path / f"{video_id}_{pipeline_name}.vtt"
                    export_webvtt_captions(data, str(webvtt_path))
                    self.logger.info(f"Saved {pipeline_name} to {webvtt_path}")

                elif output_format == "rttm" and data:
                    # Save diarization as RTTM
                    rttm_path = output_path / f"{video_id}_{pipeline_name}.rttm"
                    export_rttm_diarization(data, str(rttm_path))
                    self.logger.info(f"Saved {pipeline_name} to {rttm_path}")

                # Future: Add other format exports here
                # elif output_format == "csv" and data:  # For emotion, F0, etc.
                #     csv_path = output_path / f"{video_id}_{pipeline_name}.csv"
                #     export_csv_data(data, str(csv_path))

            except Exception as e:
                self.logger.error(f"Failed to save {pipeline_name} results: {e}")

    def get_schema(self) -> Dict[str, Any]:
        """Return the schema for modular audio pipeline outputs."""
        return {
            "type": "array",
            "description": "Results from multiple audio processing pipelines",
            "items": {
                "type": "object",
                "properties": {
                    "pipeline": {"type": "string", "description": "Name of the audio pipeline"},
                    "format": {"type": "string", "description": "Output format (webvtt, rttm, etc.)"},
                    "data": {
                        "type": "array",
                        "description": "Timestamped data from the pipeline",
                        "items": {"type": "object"}
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Pipeline processing metadata",
                        "properties": {
                            "pipeline_name": {"type": "string"},
                            "output_format": {"type": "string"},
                            "processed_segments": {"type": "integer"}
                        }
                    }
                },
                "required": ["pipeline", "format", "data", "metadata"]
            }
        }

    def cleanup(self) -> None:
        """Cleanup all audio pipeline resources."""
        self.logger.info("Cleaning up AudioPipeline coordinator...")
        
        for pipeline_name, pipeline in self.audio_pipelines.items():
            try:
                pipeline.cleanup()
                self.logger.info(f"{pipeline_name} cleanup complete")
            except Exception as e:
                self.logger.error(f"Error cleaning up {pipeline_name}: {e}")
        
        self.audio_pipelines.clear()
        self.is_initialized = False
        self.logger.info("AudioPipeline coordinator cleanup complete")
