"""Legacy batch processing entry point for standalone workflows."""

import os
import time

import pandas as pd
from src.config import MODEL_CONFIG, PATH_CONFIG, VIDEO_CONFIG
from src.processors.video_processor import get_video_metadata, videotokeypoints
from src.processors.video_understanding import extract_video_understanding
from src.utils.io_utils import get_stem_name, getProcessedVideos, saveProcessedVideos
from src.utils.keypoint_utils import normalize_keypoints
from ultralytics import YOLO


def process_video(model, video_path, data_out, metadata_row, force_process=False):
    """Process a single video file to extract keypoints and metadata.

        Parameters:
        -----------
        model : YOLO
            YOLOv8 model for keypoint detection
        video_path : str
            Path to the video file
        data_out : str
            Directory to save processed data
        metadata_row : dict or None
            Metadata for the video if available
        force_process : bool
            Whether to force reprocessing if already done

    Returns:
        --------
        dict
        Updated metadata row
    """
    videoname = os.path.basename(video_path)
    stemname = get_stem_name(video_path)
    print(f"Processing video: {videoname}")

    # Verify the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist")
        return None

    try:
        # Get processed videos dataframe
        processedvideos = getProcessedVideos(data_out)

        # Get or create metadata
        row = processedvideos.loc[processedvideos["VideoID"] == videoname]
        if row.empty and metadata_row is not None:
            # Get video metadata
            video_meta = get_video_metadata(video_path)
            if video_meta is None:
                print(f"Error opening video: {video_path}")
                return None

            # Create new row
            row = {
                "VideoID": videoname,
                "ChildID": metadata_row.get("ChildID", ""),
                "JokeType": metadata_row.get("JokeType", ""),
                "JokeNum": metadata_row.get("JokeNum", ""),
                "JokeRep": metadata_row.get("JokeRep", ""),
                "JokeTake": metadata_row.get("JokeTake", ""),
                "HowFunny": metadata_row.get("HowFunny", ""),
                "LaughYesNo": metadata_row.get("LaughYesNo", ""),
                **video_meta,
            }
            newrow = pd.DataFrame([row])

            # Ensure processedvideos has the same dtypes if it exists
            if not processedvideos.empty:
                for col in newrow.columns:
                    if col in processedvideos.columns:
                        newrow[col] = newrow[col].astype(processedvideos[col].dtype)

            processedvideos = pd.concat([processedvideos, newrow], ignore_index=True)
            row = processedvideos.loc[processedvideos["VideoID"] == videoname]

        # Skip if already processed and not forcing reprocess
        if (
            not force_process
            and not pd.isnull(row["Keypoints.file"].values[0])
            and os.path.exists(row["Keypoints.file"].values[0])
        ):
            print(
                f"Already processed {videoname} results in {row['Keypoints.file'].values[0]}"
            )
            return row.to_dict("records")[0]

        # Extract keypoints
        print(f"Extracting keypoints from {videoname}...")
        keypointsdf = videotokeypoints(model, video_path, track=VIDEO_CONFIG["track"])

        if keypointsdf.empty:
            print(f"Warning: No keypoints detected in {videoname}")
            # Still save the empty dataframe so we don't try to process this video again

        # Save keypoints
        keypointspath = os.path.join(data_out, f"{stemname}.csv")
        keypointsdf.to_csv(keypointspath, index=False)

        # Update row
        idx = processedvideos.loc[processedvideos["VideoID"] == videoname].index[0]
        processedvideos.at[idx, "Keypoints.file"] = keypointspath
        processedvideos.at[idx, "Keypoints.when"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime()
        )

        # Save updated processedvideos
        saveProcessedVideos(processedvideos, data_out)

        return processedvideos.loc[processedvideos["VideoID"] == videoname].to_dict(
            "records"
        )[0]

    except Exception as e:
        print(f"Error processing video {videoname}: {e!s}")
        import traceback

        traceback.print_exc()
        return None


def normalize_and_save_keypoints(video_row, data_out):
    """Normalize keypoints from raw keypoints and save to file.

        Parameters:
        -----------
        video_row : dict
            Row from processedvideos containing video metadata
        data_out : str
            Directory to save processed data

    Returns:
        --------
        str
        Path to the normalized keypoints file
    """
    # Read the raw keypoints
    keypoints_df = pd.read_csv(video_row["Keypoints.file"])

    # Normalize keypoints (only normalization, no other processing)
    normalized_df = normalize_keypoints(
        keypoints_df, video_row["Height"], video_row["Width"]
    )

    # Save normalized keypoints
    stemname = os.path.splitext(video_row["Keypoints.file"])[0]
    normedkeypointspath = f"{stemname}_normed.csv"
    normalized_df.to_csv(normedkeypointspath, index=False)

    # Update processed videos
    processedvideos = getProcessedVideos(data_out)

    # Ensure the column is of string dtype before assigning a string path
    if "Keypoints.normed" in processedvideos.columns and pd.api.types.is_float_dtype(
        processedvideos["Keypoints.normed"]
    ):
        processedvideos["Keypoints.normed"] = processedvideos[
            "Keypoints.normed"
        ].astype("object")

    idx = processedvideos.loc[processedvideos["VideoID"] == video_row["VideoID"]].index[
        0
    ]
    processedvideos.at[idx, "Keypoints.normed"] = normedkeypointspath
    saveProcessedVideos(processedvideos, data_out)

    return normedkeypointspath


def process_all_videos(
    videos_in, data_out, metadata_file=None, force_metadata=False, force_process=False
):
    """Process all videos in a directory.

    Parameters:
    -----------
    videos_in : str
        Directory containing input videos
    data_out : str
        Directory to save processed data
    metadata_file : str
        Filename of metadata Excel file
    force_metadata : bool
        Whether to force metadata extraction
    force_process : bool
        Whether to force video processing
    """
    # Use default metadata file if none provided
    if metadata_file is None:
        metadata_file = PATH_CONFIG["default_metadata_file"]

    # Get metadata from Excel file
    metadata = pd.read_excel(os.path.join(videos_in, metadata_file))

    # Load YOLO model
    model = YOLO(MODEL_CONFIG["pose_model"])

    # Process each video
    for _index, mrow in metadata.iterrows():
        videoname = mrow["VideoID"]
        video_path = os.path.join(videos_in, videoname)

        # Process video
        row = process_video(model, video_path, data_out, mrow, force_process)
        if row:
            # Normalize keypoints
            normalize_and_save_keypoints(row, data_out)


def understand_videos(videos_in, data_out):
    """Extract understanding from videos using the Ask-Anything approach.

    Parameters:
    -----------
    videos_in : str
        Directory containing input videos
    data_out : str
        Directory to save processed data
    """
    processedvideos = getProcessedVideos(data_out)

    for _index, row in processedvideos.iterrows():
        videoname = row["VideoID"]
        video_path = os.path.join(videos_in, videoname)

        # Extract video understanding
        understanding_file = extract_video_understanding(video_path, data_out)

        # Update processed videos
        processedvideos.at[_index, "Understanding.file"] = understanding_file
        processedvideos.at[_index, "Understanding.when"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime()
        )

    saveProcessedVideos(processedvideos, data_out)
