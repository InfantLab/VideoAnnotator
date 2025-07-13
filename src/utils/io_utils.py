"""
Utility functions for file I/O operations.
"""

import os
import json
import pandas as pd
from os.path import normpath
from pathlib import PureWindowsPath


def posixpath(path):
    """
    Convert Windows path to POSIX path.

    Args:
        path (str): Windows path

    Returns:
        str: POSIX path
    """
    return PureWindowsPath(normpath(PureWindowsPath(path).as_posix())).as_posix()


def localpath(path):
    """
    Convert path to normalized local path.

    Args:
        path (str): Path to normalize

    Returns:
        str: Normalized path
    """
    return os.path.normpath(path)


def get_stem_name(file_path):
    """
    Get the stem name from a file path (filename without extension).

    Args:
        file_path (str): File path

    Returns:
        str: Stem name
    """
    basename = os.path.basename(file_path)
    return os.path.splitext(basename)[0]


def getProcessedVideos(data_dir, filename=None):
    """
    Look in data_dir for the processed videos file (CSV or Excel),
    if it exists, load it, otherwise create it.

    Args:
        data_dir (str): Data directory
        filename (str, optional): File name. If None, uses the value from PATH_CONFIG

    Returns:
        DataFrame: DataFrame of processed videos
    """
    # Try to import PATH_CONFIG without creating circular imports
    try:
        from src.config import PATH_CONFIG

        config_filename = PATH_CONFIG.get("processed_videos_file", "processed_videos.csv")
    except ImportError:
        config_filename = "processed_videos.csv"

    # Use provided filename or default from config
    filename = filename or config_filename

    # Ensure filename has an extension
    if not os.path.splitext(filename)[1]:
        # Default to CSV if no extension specified
        filename = f"{filename}.csv"

    filepath = os.path.join(data_dir, filename)

    # For backward compatibility, check if Excel file exists but CSV was requested
    excel_path = None
    if filepath.endswith(".csv"):
        excel_path = os.path.join(data_dir, f"{os.path.splitext(filename)[0]}.xlsx")

    # First try the specified path
    if os.path.exists(filepath):
        if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            processedvideos = pd.read_excel(filepath, index_col=None)
            print(f"Found existing {filename} with {processedvideos.shape[0]} rows.")
        else:
            processedvideos = pd.read_csv(filepath, index_col=None)
            print(f"Found existing {filename} with {processedvideos.shape[0]} rows.")
    # For backward compatibility, try Excel if CSV doesn't exist
    elif excel_path and os.path.exists(excel_path):
        processedvideos = pd.read_excel(excel_path, index_col=None)
        print(
            f"Found existing Excel file {os.path.basename(excel_path)} with {processedvideos.shape[0]} rows."
        )
        # Save as CSV for future use
        processedvideos.to_csv(filepath, index=False)
        print(f"Converted to CSV format at {filepath}")
    else:
        # Create new dataframe for info about processed videos
        print(f"Creating new {filename}")
        cols = [
            "VideoID",
            "ChildID",
            "JokeType",
            "JokeNum",
            "JokeRep",
            "JokeTake",
            "HowFunny",
            "LaughYesNo",
            "Frames",
            "FPS",
            "Width",
            "Height",
            "Duration",
            "Keypoints.when",
            "Keypoints.file",
            "Audio.when",
            "Audio.file",
            "Faces.when",
            "Faces.file",
            "Speech.when",
            "Speech.file",
            "Diary.file",
            "Diary.when",
            "LastError",
            "annotatedVideo",
            "annotated.when",
        ]
        processedvideos = pd.DataFrame(columns=cols)
        # Save as CSV by default
        if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            processedvideos.to_excel(filepath, index=False)
        else:
            processedvideos.to_csv(filepath, index=False)

    return processedvideos


def saveProcessedVideos(df, data_out, filename=None):
    """
    Save the processed videos dataframe.

    Args:
        df (DataFrame): DataFrame with processed videos information
        data_out (str): Path to data output directory
        filename (str, optional): File name. If None, uses the value from PATH_CONFIG
    """
    # Try to import PATH_CONFIG without creating circular imports
    try:
        from src.config import PATH_CONFIG

        config_filename = PATH_CONFIG.get("processed_videos_file", "processed_videos.csv")
    except ImportError:
        config_filename = "processed_videos.csv"

    # Use provided filename or default from config
    filename = filename or config_filename

    # Ensure filename has an extension
    if not os.path.splitext(filename)[1]:
        # Default to CSV if no extension specified
        filename = f"{filename}.csv"

    processed_videos_path = os.path.join(data_out, filename)

    # Save in appropriate format based on extension
    if processed_videos_path.endswith(".xlsx") or processed_videos_path.endswith(".xls"):
        df.to_excel(processed_videos_path, index=False)
    else:
        df.to_csv(processed_videos_path, index=False)

    print(f"Saved processed videos information to {processed_videos_path}")


def getVideoProperty(processedvideos, VideoID, Property):
    """
    Get the value of a property for a video in the processedvideos dataframe.

    Args:
        processedvideos (DataFrame): DataFrame of processed videos
        VideoID (str): Video ID
        Property (str): Property name

    Returns:
        any: Property value
    """
    videodata = processedvideos[processedvideos["VideoID"] == VideoID]
    if videodata.shape[0] > 0:
        return videodata[Property].values[0]
    else:
        return None


def readKeyPointsFromCSV(processedvideos, VIDEO_FILE, normed=False):
    """
    Get the keypoints from the CSV file.

    Args:
        processedvideos (DataFrame): DataFrame of processed videos
        VIDEO_FILE (str): Video file path
        normed (bool): Whether to get normalized keypoints

    Returns:
        DataFrame: DataFrame of keypoints
    """
    videoname = os.path.basename(VIDEO_FILE)
    # is video in the processedvideos dataframe?
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] <= 0:
        raise FileNotFoundError(f"No keypoints file found for {videoname}")
    if normed:
        kptsfile = videodata["Keypoints.normed"].values[0]
    else:
        kptsfile = videodata["Keypoints.file"].values[0]
    return pd.read_csv(kptsfile)


def getKeyPoints(processedvideos, videoname):
    """
    Get keypoints DataFrame for a video.

    Args:
        processedvideos (DataFrame): DataFrame of processed videos
        videoname (str): Video name

    Returns:
        DataFrame: DataFrame of keypoints
    """
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] <= 0:
        raise FileNotFoundError(f"No keypoints file found for {videoname}")
    print(f"We have a keypoints file for {videoname}")
    kptsfile = videodata["Keypoints.file"].values[0]
    return pd.read_csv(kptsfile)


def getFaceData(processedvideos, videoname):
    """
    Get face data DataFrame for a video.

    Args:
        processedvideos (DataFrame): DataFrame of processed videos
        videoname (str): Video name

    Returns:
        DataFrame: DataFrame of face data
    """
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] <= 0:
        raise FileNotFoundError(f"No face data file found for {videoname}")
    print(f"We have a face data file for {videoname}")
    facesfile = videodata["Faces.file"].values[0]
    return pd.read_csv(facesfile)


def getSpeechData(processedvideos, videoname):
    """
    Get speech data for a video.

    Args:
        processedvideos (DataFrame): DataFrame of processed videos
        videoname (str): Video name

    Returns:
        dict: Speech data for the video
    """
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] <= 0:
        raise FileNotFoundError(f"No speech data file found for {videoname}")
    print(f"We have a speech data file for {videoname}")
    speechfile = videodata["Speech.file"].values[0]
    # Load the speech file
    with open(speechfile) as f:
        speechdata = json.load(f)
    return speechdata


def getfacecols():
    """
    Get the column names for facial landmarks X and Y coordinates.

    Returns:
        tuple: Lists of X and Y column names for facial landmarks
    """
    # Common facial landmark column prefixes
    landmark_prefixes = [
        "left_eye",
        "right_eye",
        "nose",
        "mouth_left",
        "mouth_right",
        "left_eyebrow",
        "right_eyebrow",
        "mouth_center",
    ]

    # Generate X and Y column names
    facecolsx = [f"{prefix}_x" for prefix in landmark_prefixes]
    facecolsy = [f"{prefix}_y" for prefix in landmark_prefixes]

    # Add basic coordinates
    facecolsx.append("x")
    facecolsy.append("y")

    return facecolsx, facecolsy
