"""
Audio Processing Pipeline for VideoAnnotator

This pipeline handles comprehensive audio analysis including:
- Speech transcription and diarization
- Audio event detection and classification
- Music and silence detection
- Audio quality assessment
- Emotion analysis from speech

Supports multiple backends:
- Whisper for speech transcription
- pyannote.audio for speaker diarization
- torch-audiomentations for audio event detection
- librosa for acoustic feature extraction
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
import librosa
import torch
from dataclasses import dataclass

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper not available. Speech transcription will be disabled.")

try:
    from pyannote.audio import Pipeline as PyannoteePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote.audio not available. Speaker diarization will be disabled.")

from ..base_pipeline import BasePipeline
from .ffmpeg_utils import extract_audio_from_video, check_ffmpeg_available
from ...schemas.audio_schema import (
    AudioSegment, SpeechRecognition, SpeakerDiarization, 
    AudioClassification, AudioFeatures
)


@dataclass
class AudioPipelineConfig:
    """Configuration for audio processing pipeline."""
    
    # General settings
    sample_rate: int = 16000
    chunk_duration: float = 30.0  # seconds
    overlap_duration: float = 1.0  # seconds
    
    # Authentication
    huggingface_token: Optional[str] = None  # For PyAnnote models
    
    # Speech transcription
    whisper_model: str = "base"  # tiny, base, small, medium, large
    transcription_language: Optional[str] = None
    
    # Speaker diarization
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int = 1
    max_speakers: int = 10
    
    # Audio event detection
    event_detection_threshold: float = 0.5
    event_min_duration: float = 0.1
    
    # Music detection
    music_detection_threshold: float = 0.7
    music_min_duration: float = 2.0
    
    # Audio quality
    quality_window_size: float = 1.0
    snr_threshold: float = 10.0
    
    # Emotion analysis
    emotion_window_size: float = 3.0
    emotion_overlap: float = 0.5


class AudioPipeline(BasePipeline):
    """
    Audio processing pipeline with support for multiple audio analysis tasks.
    """
    
    def __init__(self, config: Optional[AudioPipelineConfig] = None):
        super().__init__()
        self.config = config or AudioPipelineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set up authentication token
        if not self.config.huggingface_token:
            # Try HF_AUTH_TOKEN first (standard), then HUGGINGFACE_TOKEN (legacy)
            self.config.huggingface_token = os.getenv("HF_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        # Initialize models
        self._whisper_model = None
        self._diarization_pipeline = None
        self.is_initialized = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize audio processing models."""
        try:
            # Check FFmpeg availability for video processing
            if not check_ffmpeg_available():
                self.logger.warning("FFmpeg not found. Video processing will be disabled.")
            
            # Initialize Whisper for speech transcription
            if WHISPER_AVAILABLE:
                self.logger.info(f"Loading Whisper model: {self.config.whisper_model}")
                self._whisper_model = whisper.load_model(self.config.whisper_model)
            
            # Initialize pyannote for speaker diarization
            if PYANNOTE_AVAILABLE:
                self.logger.info(f"Loading diarization model: {self.config.diarization_model}")
                if self.config.huggingface_token:
                    self._diarization_pipeline = PyannoteePipeline.from_pretrained(
                        self.config.diarization_model,
                        use_auth_token=self.config.huggingface_token
                    )
                    # Send to GPU if available
                    if torch.cuda.is_available():
                        self._diarization_pipeline.to(torch.device("cuda"))
                        self.logger.info("Diarization pipeline moved to GPU")
                else:
                    self.logger.warning("No HuggingFace token provided. Speaker diarization may fail.")
            
            self.is_initialized = True
            self.logger.info("Audio pipeline initialization completed")
                    
        except Exception as e:
            self.logger.error(f"Error initializing audio models: {e}")
            raise
    
    def initialize(self) -> None:
        """Initialize the pipeline (load models, etc.)."""
        if not self.is_initialized:
            self._initialize_models()
            self.is_initialized = True
            self.logger.info("Audio pipeline initialized successfully")
    
    def process(
        self, 
        video_path: str, 
        start_time: float = 0.0, 
        end_time: Optional[float] = None,
        pps: float = 0.0,  # predictions per second, 0 = once per segment
        output_dir: Optional[str] = None
    ) -> List:
        """
        Process video segment and extract audio features.
        
        Args:
            video_path: Path to video file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds (None = full video)
            pps: Predictions per second (0 = once per segment)
            output_dir: Optional output directory for saving results
            
        Returns:
            List of audio annotation objects
        """
        try:
            # First extract audio from video if needed
            video_path = Path(video_path)
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                audio_path = output_path / f"{video_path.stem}.wav"
            else:
                audio_path = video_path.with_suffix('.wav')
            
            # Extract audio using FFmpeg
            if not audio_path.exists():
                extracted_audio = extract_audio_from_video(
                    video_path=video_path,
                    output_path=audio_path,
                    sample_rate=self.config.sample_rate,
                    channels=1  # Mono for speech processing
                )
                if extracted_audio:
                    audio_path = Path(extracted_audio)
                else:
                    raise RuntimeError(f"Failed to extract audio from {video_path}")
            
            # Process the audio file
            results = self.process_audio(audio_path)
            
            # Convert results to annotation objects
            annotations = []
            
            # Add speaker diarization annotations
            if results.get('speaker_diarization'):
                annotations.append(results['speaker_diarization'])
            
            # Add speech transcription annotations
            if results.get('speech_transcription'):
                annotations.append(results['speech_transcription'])
            
            # Add audio events
            annotations.extend(results.get('audio_events', []))
            
            return annotations
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return []

    def process_diarization_only(self, audio_path: Union[str, Path]) -> Optional[SpeakerDiarization]:
        """
        Process only speaker diarization for an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            SpeakerDiarization object or None if failed
        """
        if not self.is_initialized:
            self.initialize()
            
        return self._diarize_speakers(Path(audio_path))

    def process_audio(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process audio file and extract comprehensive audio features.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all audio analysis results
        """
        if not self.is_initialized:
            self._initialize_models()
            self.is_initialized = True
            
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.logger.info(f"Processing audio file: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=self.config.sample_rate)
        duration = len(audio) / sr
        
        results = {
            'file_path': str(audio_path),
            'duration': duration,
            'sample_rate': sr,
            'segments': [],
            'speech_transcription': None,
            'speaker_diarization': None,
            'audio_events': [],
            'music_detection': None,
            'audio_quality': None,
            'speech_emotions': []
        }
        
        # Process in chunks for long audio files
        chunk_samples = int(self.config.chunk_duration * sr)
        overlap_samples = int(self.config.overlap_duration * sr)
        
        for start_sample in range(0, len(audio), chunk_samples - overlap_samples):
            end_sample = min(start_sample + chunk_samples, len(audio))
            chunk = audio[start_sample:end_sample]
            chunk_start_time = start_sample / sr
            chunk_end_time = end_sample / sr
            
            # Create audio segment
            segment = AudioSegment(
                start_time=chunk_start_time,
                end_time=chunk_end_time,
                audio_data=chunk,
                sample_rate=sr
            )
            results['segments'].append(segment)
        
        # Perform full-file analysis
        if self._whisper_model:
            results['speech_transcription'] = self._transcribe_speech(audio_path)
        
        if self._diarization_pipeline:
            results['speaker_diarization'] = self._diarize_speakers(audio_path)
        
        results['audio_events'] = self._detect_audio_events(audio, sr)
        results['music_detection'] = self._detect_music(audio, sr)
        results['audio_quality'] = self._assess_audio_quality(audio, sr)
        results['speech_emotions'] = self._analyze_speech_emotions(audio, sr)
        
        return results
    
    def _transcribe_speech(self, audio_path: Path) -> Optional[SpeechRecognition]:
        """Transcribe speech using Whisper."""
        if not self._whisper_model:
            return None
        
        try:
            result = self._whisper_model.transcribe(
                str(audio_path),
                language=self.config.transcription_language
            )
            
            # Extract word-level timestamps if available
            word_timestamps = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word in segment['words']:
                            word_timestamps.append({
                                'word': word['word'],
                                'start': word['start'],
                                'end': word['end'],
                                'confidence': word.get('probability', 0.0)
                            })
            
            return SpeechRecognition(
                text=result['text'],
                language=result.get('language', 'unknown'),
                confidence=0.0,  # Whisper doesn't provide overall confidence
                word_timestamps=word_timestamps,
                segments=[{
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text']
                } for seg in result.get('segments', [])]
            )
            
        except Exception as e:
            self.logger.error(f"Speech transcription failed: {e}")
            return None
    
    def _diarize_speakers(self, audio_path: Path) -> Optional[SpeakerDiarization]:
        """Perform speaker diarization using pyannote."""
        if not self._diarization_pipeline:
            return None
        
        try:
            diarization = self._diarization_pipeline(str(audio_path))
            
            speakers = {}
            segments = []
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_id = f"speaker_{speaker}"
                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        'speaker_id': speaker_id,
                        'total_duration': 0.0,
                        'segments': []
                    }
                
                segment = {
                    'start_time': turn.start,
                    'end_time': turn.end,
                    'speaker_id': speaker_id,
                    'confidence': 1.0  # pyannote doesn't provide confidence scores
                }
                
                speakers[speaker_id]['segments'].append(segment)
                speakers[speaker_id]['total_duration'] += turn.end - turn.start
                segments.append(segment)
            
            return SpeakerDiarization(
                speakers=list(speakers.values()),
                segments=segments,
                num_speakers=len(speakers)
            )
            
        except Exception as e:
            self.logger.error(f"Speaker diarization failed: {e}")
            return None
    
    def _detect_audio_events(self, audio: np.ndarray, sr: int) -> List[AudioClassification]:
        """Detect audio events (applause, laughter, etc.)."""
        events = []
        
        # Placeholder implementation - would use trained models
        # This is a simple energy-based detection
        window_size = int(0.1 * sr)  # 100ms windows
        hop_size = window_size // 2
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            energy = np.mean(window ** 2)
            
            if energy > self.config.event_detection_threshold:
                event = AudioClassification(
                    event_type="generic_event",
                    start_time=i / sr,
                    end_time=(i + window_size) / sr,
                    confidence=min(energy, 1.0),
                    properties={'energy': float(energy)}
                )
                events.append(event)
        
        # Merge nearby events
        merged_events = self._merge_nearby_events(events)
        return merged_events
    
    def _detect_music(self, audio: np.ndarray, sr: int) -> Optional[AudioClassification]:
        """Detect music segments in audio."""
        # Placeholder implementation using spectral features
        # Real implementation would use trained music detection models
        
        # Calculate spectral centroid as a simple music indicator
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Music typically has more consistent spectral centroid
        consistency = 1.0 - np.std(spectral_centroids) / np.mean(spectral_centroids)
        
        is_music = consistency > self.config.music_detection_threshold
        
        if is_music:
            return AudioClassification(
                is_music=True,
                confidence=consistency,
                segments=[{
                    'start_time': 0.0,
                    'end_time': len(audio) / sr,
                    'music_type': 'unknown',
                    'confidence': consistency
                }],
                properties={
                    'spectral_consistency': float(consistency),
                    'mean_spectral_centroid': float(np.mean(spectral_centroids))
                }
            )
        
        return AudioClassification(
            is_music=False,
            confidence=1.0 - consistency,
            segments=[],
            properties={}
        )
    
    def _assess_audio_quality(self, audio: np.ndarray, sr: int) -> AudioClassification:
        """Assess audio quality metrics."""
        # Calculate various quality metrics
        
        # Signal-to-noise ratio (simplified)
        signal_power = np.mean(audio ** 2)
        noise_floor = np.percentile(audio ** 2, 10)  # Assume bottom 10% is noise
        snr = 10 * np.log10(signal_power / max(noise_floor, 1e-10))
        
        # Dynamic range
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / max(np.mean(np.abs(audio)), 1e-10))
        
        # Clipping detection
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
        clipping_ratio = clipped_samples / len(audio)
        
        # Overall quality score (0-1)
        quality_score = min(1.0, max(0.0, (snr / 40.0 + (1.0 - clipping_ratio)) / 2.0))
        
        return AudioClassification(
            signal_to_noise_ratio=snr,
            dynamic_range=dynamic_range,
            clipping_ratio=clipping_ratio,
            quality_score=quality_score,
            issues=[] if quality_score > 0.7 else ['low_quality'],
            metrics={
                'signal_power': float(signal_power),
                'noise_floor': float(noise_floor),
                'max_amplitude': float(np.max(np.abs(audio)))
            }
        )
    
    def _analyze_speech_emotions(self, audio: np.ndarray, sr: int) -> List[AudioClassification]:
        """Analyze emotions in speech."""
        # Placeholder implementation
        # Real implementation would use trained emotion recognition models
        
        emotions = []
        window_size = int(self.config.emotion_window_size * sr)
        hop_size = int(window_size * (1.0 - self.config.emotion_overlap))
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            
            # Extract basic acoustic features
            mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=window, sr=sr)[0]
            
            # Simple rule-based emotion detection (placeholder)
            mean_mfcc = np.mean(mfccs)
            mean_centroid = np.mean(spectral_centroid)
            
            # Determine emotion based on acoustic features
            if mean_centroid > 2000:
                emotion = "excitement"
                confidence = 0.6
            elif mean_centroid < 1000:
                emotion = "sadness"
                confidence = 0.5
            else:
                emotion = "neutral"
                confidence = 0.7
            
            emotion_obj = AudioClassification(
                emotion=emotion,
                confidence=confidence,
                start_time=i / sr,
                end_time=(i + window_size) / sr,
                acoustic_features={
                    'mean_mfcc': float(mean_mfcc),
                    'mean_spectral_centroid': float(mean_centroid)
                }
            )
            emotions.append(emotion_obj)
        
        return emotions
    
    def _merge_nearby_events(self, events: List[AudioClassification]) -> List[AudioClassification]:
        """Merge nearby audio events."""
        if not events:
            return []
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda x: x.start_time)
        merged = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            last_event = merged[-1]
            
            # Merge if events are close and of same type
            if (event.start_time - last_event.end_time < 0.5 and 
                event.event_type == last_event.event_type):
                last_event.end_time = event.end_time
                last_event.confidence = max(last_event.confidence, event.confidence)
            else:
                merged.append(event)
        
        # Filter out events that are too short
        return [e for e in merged if e.end_time - e.start_time >= self.config.event_min_duration]
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline and its capabilities."""
        return {
            'name': 'AudioPipeline',
            'version': '1.0.0',
            'capabilities': {
                'speech_transcription': WHISPER_AVAILABLE,
                'speaker_diarization': PYANNOTE_AVAILABLE,
                'audio_event_detection': True,
                'music_detection': True,
                'audio_quality_assessment': True,
                'speech_emotion_analysis': True
            },
            'models': {
                'whisper_model': self.config.whisper_model if WHISPER_AVAILABLE else None,
                'diarization_model': self.config.diarization_model if PYANNOTE_AVAILABLE else None
            },
            'config': self.config.__dict__
        }
    
    def process_video(self, video_path: Union[str, Path], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process video file by extracting audio first, then running audio analysis.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted audio and results
            
        Returns:
            Dictionary containing all audio analysis results
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Set up output directory
        if output_dir is None:
            output_dir = video_path.parent / "audio_processing"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract audio from video
        self.logger.info(f"Extracting audio from video: {video_path}")
        audio_path = extract_audio_from_video(
            video_path, 
            output_dir / f"{video_path.stem}_audio.wav",
            sample_rate=self.config.sample_rate,
            channels=1  # Mono for speech processing
        )
        
        if audio_path is None:
            raise RuntimeError(f"Failed to extract audio from video: {video_path}")
        
        # Process the extracted audio
        results = self.process_audio(audio_path)
        results['original_video_path'] = str(video_path)
        results['extracted_audio_path'] = str(audio_path)
        
        return results
    
    def diarize_only(self, audio_path: Union[str, Path]) -> Optional[SpeakerDiarization]:
        """
        Perform only speaker diarization on an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            SpeakerDiarization object or None if failed
        """
        if not self.is_initialized:
            self._initialize_models()
            self.is_initialized = True
        
        return self._diarize_speakers(Path(audio_path))
    
    def transcribe_only(self, audio_path: Union[str, Path]) -> Optional[SpeechRecognition]:
        """
        Perform only speech transcription on an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            SpeechRecognition object or None if failed
        """
        if not self.is_initialized:
            self._initialize_models()
            self.is_initialized = True
        
        return self._transcribe_speech(Path(audio_path))

    def extract_f0_features(self, audio_path: Union[str, Path]) -> Optional[AudioFeatures]:
        """
        Extract fundamental frequency (F0) features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioFeatures object with F0 data or None if failed
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            self.logger.error(f"Audio file not found: {audio_path}")
            return None
        
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.config.sample_rate)
            duration = len(audio) / sr
            
            # Extract fundamental frequency using librosa
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                frame_length=2048,
                hop_length=512
            )
            
            # Calculate additional prosodic features
            # Remove NaN values for statistics
            f0_clean = f0[~np.isnan(f0)]
            
            prosodic_features = {}
            if len(f0_clean) > 0:
                prosodic_features = {
                    'f0_mean': float(np.mean(f0_clean)),
                    'f0_std': float(np.std(f0_clean)),
                    'f0_min': float(np.min(f0_clean)),
                    'f0_max': float(np.max(f0_clean)),
                    'f0_range': float(np.max(f0_clean) - np.min(f0_clean)),
                    'voiced_ratio': float(np.sum(voiced_flag) / len(voiced_flag))
                }
            
            # Convert to regular Python lists for JSON serialization
            f0_list = [float(x) if not np.isnan(x) else None for x in f0]
            voiced_flag_list = [bool(x) for x in voiced_flag]
            voiced_probs_list = [float(x) for x in voiced_probs]
            
            return AudioFeatures(
                duration=duration,
                sample_rate=sr,
                fundamental_frequency=f0_list,
                prosodic_features=prosodic_features,
                metadata={
                    'voiced_flag': voiced_flag_list,
                    'voiced_probabilities': voiced_probs_list,
                    'frame_length': 2048,
                    'hop_length': 512
                }
            )
            
        except Exception as e:
            self.logger.error(f"F0 extraction failed: {e}")
            return None
