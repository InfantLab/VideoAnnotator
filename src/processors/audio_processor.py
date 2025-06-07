"""
Functions for processing audio data.
"""

import os
import time
import json
import numpy as np
import torch
import dotenv
import moviepy.editor as mp
from pyannote.audio import Pipeline

def extract_audio(video_path, output_dir, output_ext="wav"):
    """
    Extract audio from a video file using moviepy.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the audio file
        output_ext (str): Audio file extension (mp3 or wav)
        
    Returns:
        str: Path to the extracted audio file
    """
    basename = os.path.basename(video_path)
    filename, _ = os.path.splitext(basename)
    audio_path = os.path.join(output_dir, f"{filename}.{output_ext}")
    
    if not os.path.exists(audio_path):
        try:
            clip = mp.VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, logger=None)
            clip.close()
        except Exception as e:
            print(f"Error extracting audio from {basename}: {e}")
            return None
    
    return audio_path

def convert_video_to_audio_moviepy(videos_in, video_file, out_path, output_ext="mp3"):
    """
    Converts video to audio using MoviePy library.
    
    Args:
        videos_in (str): Input video directory
        video_file (str): Video file name
        out_path (str): Output directory
        output_ext (str): Output file extension
        
    Returns:
        str: Path to the output audio file
    """
    try:
        filename = os.path.splitext(video_file)[0]
        video = os.path.join(videos_in, video_file)
        clip = mp.VideoFileClip(video)
        audio_file = os.path.join(out_path, f"{filename}.{output_ext}")
        clip.audio.write_audiofile(audio_file)
        clip.close()
        return audio_file
    except Exception as e:
        print(f"Error converting {video_file} to {output_ext}: {e}")
        return None

def convert_mp3_to_wav_moviepy(audio_file, output_ext="wav"):
    """
    Converts MP3 to WAV using MoviePy library.
    
    Args:
        audio_file (str): Audio file path
        output_ext (str): Output file extension
        
    Returns:
        str: Path to the output WAV file
    """
    filename, ext = os.path.splitext(audio_file)
    clip = mp.AudioFileClip(audio_file)
    wav_file = f"{filename}.{output_ext}"
    clip.write_audiofile(wav_file)
    return wav_file

def transcribe_audio(audio_file, output_dir, model_name="base"):
    """
    Transcribe audio using OpenAI's Whisper model.
    
    Args:
        audio_file (str): Path to the audio file
        output_dir (str): Directory to save the transcript
        model_name (str): Whisper model name (tiny, base, small, medium, large)
        
    Returns:
        tuple: (Path to the transcript file, transcript data)
    """
    import whisper
    
    basename = os.path.basename(audio_file)
    filename, _ = os.path.splitext(basename)
    json_path = os.path.join(output_dir, f"{filename}.transcript.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            result = json.load(f)
        return json_path, result
    
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_file, verbose=True)
        
        with open(json_path, "w") as f:
            json.dump(result, f)
        
        return json_path, result
    except Exception as e:
        print(f"Error transcribing {basename}: {e}")
        return None, None

def diarize_audio(audio_file, huggingface_token):
    """
    Diarize an audio file using the pyannote speaker diarization model.
    
    Args:
        audio_file (str): Path to a WAV file
        huggingface_token (str): HuggingFace API token
        
    Returns:
        Diarization: Diarization results in JSON format
    """
    try:
        print(f"Diarizing audio file: {audio_file}")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=huggingface_token)

        # send pipeline to GPU (when available)
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # apply pretrained pipeline
        diarization = pipeline(audio_file)
        
        return diarization
        
    except Exception as e:
        print(f"diarize_audio Error: {str(e)}")
        return None

def extract_fundamental_frequency(audio_file, output_dir):
    """
    Extract fundamental frequency (F0) from audio file.
    
    Args:
        audio_file (str): Path to the audio file
        output_dir (str): Directory to save the F0 data
        
    Returns:
        str: Path to the F0 data file
    """
    import librosa
    
    basename = os.path.basename(audio_file)
    filename, _ = os.path.splitext(basename)
    f0_path = os.path.join(output_dir, f"{filename}.f0.npz")
    
    if os.path.exists(f0_path):
        return f0_path
    
    try:
        # Load audio
        y, sr = librosa.load(audio_file)
        
        # Extract fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7')
        )
        
        # Save results
        np.savez(f0_path, f0=f0, voiced_flag=voiced_flag, voiced_probs=voiced_probs)
        
        return f0_path
    except Exception as e:
        print(f"Error extracting F0 from {basename}: {e}")
        return None

def detect_laughter(audio_file, output_dir):
    """
    Detect laughter segments in audio file.
    
    Args:
        audio_file (str): Path to the WAV audio file
        output_dir (str): Directory to save the laughter segments data
        
    Returns:
        str: Path to the laughter segments data file
    """
    basename = os.path.basename(audio_file)
    filename, ext = os.path.splitext(basename)
    laughter_path = os.path.join(output_dir, f"{filename}.laughter.json")
    
    if os.path.exists(laughter_path):
        return laughter_path
    
    if ext.lower() != ".wav":
        print(f"Laughter detection requires WAV format, but {basename} is {ext}")
        return None
    
    try:
        import laughter_segmenter
        
        # Detect laughter segments
        laughter_segments = laughter_segmenter.extract_laughter_segments(audio_file)
        
        # Save results
        with open(laughter_path, 'w') as f:
            json.dump(laughter_segments, f)
        
        return laughter_path
    except ImportError:
        print("Laughter segmenter not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "git+https://github.com/jrgillick/laughter-detection.git"])
        return detect_laughter(audio_file, output_dir)
    except Exception as e:
        print(f"Error detecting laughter in {basename}: {e}")
        return None

def process_audio(video_path, output_dir, enable_whisper=True, enable_diarization=True, 
                 enable_f0=True, enable_laughter=True, force_process=False):
    """
    Process audio from a video file with all available methods.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the processed data
        enable_whisper (bool): Enable speech transcription
        enable_diarization (bool): Enable speaker diarization
        enable_f0 (bool): Enable fundamental frequency extraction
        enable_laughter (bool): Enable laughter detection
        force_process (bool): Force reprocessing of existing files
        
    Returns:
        dict: Dictionary of results with paths to output files
    """
    results = {}
    
    # Extract audio
    audio_path = extract_audio(video_path, output_dir, output_ext="wav")
    results["audio"] = audio_path
    
    if audio_path is None:
        return results
    
    # Speech transcription
    if enable_whisper:
        transcript_path, _ = transcribe_audio(audio_path, output_dir)
        results["transcript"] = transcript_path
    
    # Speaker diarization
    if enable_diarization:
        dotenv.load_dotenv()
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("Warning: HUGGINGFACE_TOKEN not found in environment, diarization may fail.")
        diarization_result = diarize_audio(audio_path, hf_token)
        results["diarization"] = diarization_result
    
    # Fundamental frequency
    if enable_f0:
        f0_path = extract_fundamental_frequency(audio_path, output_dir)
        results["f0"] = f0_path
    
    # Laughter detection
    if enable_laughter and audio_path.lower().endswith(".wav"):
        laughter_path = detect_laughter(audio_path, output_dir)
        results["laughter"] = laughter_path
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio from video files")
    parser.add_argument("video_path", type=str, help="Path to video file or directory of videos")
    parser.add_argument("--output", "-o", type=str, default="./data", help="Output directory")
    parser.add_argument("--no-whisper", action="store_true", help="Disable Whisper transcription")
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--no-f0", action="store_true", help="Disable F0 extraction")
    parser.add_argument("--no-laughter", action="store_true", help="Disable laughter detection")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing of existing files")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.video_path):
        for file in os.listdir(args.video_path):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_file = os.path.join(args.video_path, file)
                print(f"Processing {file}...")
                process_audio(
                    video_file, 
                    args.output, 
                    enable_whisper=not args.no_whisper,
                    enable_diarization=not args.no_diarization,
                    enable_f0=not args.no_f0,
                    enable_laughter=not args.no_laughter,
                    force_process=args.force
                )
    else:
        process_audio(
            args.video_path, 
            args.output, 
            enable_whisper=not args.no_whisper,
            enable_diarization=not args.no_diarization,
            enable_f0=not args.no_f0,
            enable_laughter=not args.no_laughter,
            force_process=args.force
        )
