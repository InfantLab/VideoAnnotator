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
from ...schemas.audio_schema import (
    AudioSegment, SpeechRecognition, SpeakerDiarization, 
    AudioClassification
)


@dataclass
class AudioPipelineConfig:
    """Configuration for audio processing pipeline."""
    
    # General settings
    sample_rate: int = 16000
    chunk_duration: float = 30.0  # seconds
    overlap_duration: float = 1.0  # seconds
    
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
        
        # Initialize models
        self._whisper_model = None
        self._diarization_pipeline = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize audio processing models."""
        try:
            # Initialize Whisper for speech transcription
            if WHISPER_AVAILABLE:
                self.logger.info(f"Loading Whisper model: {self.config.whisper_model}")
                self._whisper_model = whisper.load_model(self.config.whisper_model)
            
            # Initialize pyannote for speaker diarization
            if PYANNOTE_AVAILABLE:
                self.logger.info(f"Loading diarization model: {self.config.diarization_model}")
                self._diarization_pipeline = PyannoteePipeline.from_pretrained(
                    self.config.diarization_model
                )
                
        except Exception as e:
            self.logger.error(f"Error initializing audio models: {e}")
            raise
    
    def process_audio(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process audio file and extract comprehensive audio features.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all audio analysis results
        """
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
