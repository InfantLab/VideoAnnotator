import os
import librosa


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
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    return f0, voiced_flag, voiced_probs
