import os
import librosa
import numpy as np
import moviepy.editor as mp

def convert_video_to_audio_moviepy(videos_in, video_file, out_path, output_ext="mp3"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
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
    """Converts mp3 to wav using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(audio_file)
    clip = mp.AudioFileClip(audio_file)
    clip.write_audiofile(f"{filename}.{output_ext}")

def find_f0(audio_file):
    """
    Extract the fundamental frequency (F0) from an audio file.
    Args:
        audio_file (str): The path to the audio file.
    Returns:
        np.array: The fundamental frequency values.
    """
    # Load the audio file
    y, sr = librosa.load(audio_file)
    # Extract the fundamental frequency
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return f0, voiced_flag, voiced_probs
