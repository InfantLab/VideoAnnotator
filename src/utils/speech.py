import os
import json
import whisper


def whisper_transcribe(audio_file, save_path, saveJSON=True):
    """
    Transcribe audio using OpenAI's Whisper model.

    Args:
        audio_file (str): Path to the audio file to transcribe
        save_path (str): Directory to save the JSON output
        saveJSON (bool): Whether to save the result to a JSON file

    Returns:
        tuple: (path to JSON file, transcription result) or just the result if saveJSON=False
    """
    # Load model once (in practice, this should be done outside this function)
    model = whisper.load_model("base")

    result = model.transcribe(audio_file, verbose=True)

    if saveJSON:
        basename = os.path.basename(audio_file)
        filename, ext = os.path.splitext(basename)
        jsonfile = os.path.join(save_path, filename + ".json")
        with open(jsonfile, "w") as f:
            json.dump(result, f)
        return jsonfile, result
    else:
        return result
