"""Audio utility helpers for extracting basic signal features."""


def find_f0(audio_file):
    """Extract the fundamental frequency (F0) from an audio file.

    Args:
        audio_file (str): The path to the audio file.
    Returns:
        np.array: The fundamental frequency values.
    """
    # librosa is an `audio`/`audio-laion` extra, not a core dependency — import
    # lazily so `videoannotator.utils` (imported by the core CLI/API) doesn't
    # require it just to load this module. See 004-extras-based-install.
    import librosa

    # Load the audio file
    y, _sr = librosa.load(audio_file)
    # Extract the fundamental frequency
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    return f0, voiced_flag, voiced_probs
