import os
import torch
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def diarize_audio(audio_file):
    """Extract speaker diarization from an audio file.
    Args:
        audio_file (str): The path to the audio file.
    Returns:
        Annotation: PyAnnote diarization object with speaker segments.
    """
    # Get HuggingFace token from environment variable
    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if not hf_token:
        print("Warning: HUGGINGFACE_ACCESS_TOKEN not found in environment, diarization may fail.")
    
    # Initialize diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Run diarization
    diarization = pipeline(audio_file)
    
    return diarization

def load_rttm(rttm_file):
    """Load an RTTM file into a PyAnnote annotation object.
    Args:
        rttm_file (str): The path to the RTTM file.
    Returns:
        dict: Dictionary with file URI as key and Annotation as value.
    """
    from pyannote.database.util import load_rttm
    return load_rttm(rttm_file)
