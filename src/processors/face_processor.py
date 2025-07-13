import os
import cv2
import math
import pandas as pd
import numpy as np
from deepface import DeepFace


def truncate_small_values(value, precision=6):
    """
    Truncate very small floating point values to a specified precision.

    Args:
        value (float): The value to truncate
        precision (int): Number of decimal places to keep

    Returns:
        float: Truncated value
    """
    # Return non-numeric values unchanged
    if not isinstance(value, (int, float)) or math.isnan(value):
        return value

    # Round to specified precision
    return round(value, precision)


def create_faces_dataframe():
    """
    Create an empty DataFrame for storing face data.

    Returns:
        DataFrame: Empty DataFrame with columns for face data
    """
    cols = [
        "frame",
        "face_idx",
        "face_x",
        "face_y",
        "face_w",
        "face_h",
        "emotion_angry",
        "emotion_disgust",
        "emotion_fear",
        "emotion_happy",
        "emotion_sad",
        "emotion_surprise",
        "emotion_neutral",
        "age",
        "gender",
        "dominant_emotion",
        "right_eye_x",
        "right_eye_y",
        "left_eye_x",
        "left_eye_y",
        "nose_x",
        "nose_y",
        "mouth_right_x",
        "mouth_right_y",
        "mouth_left_x",
        "mouth_left_y",
    ]
    return pd.DataFrame(columns=cols)


def add_faces_to_dataframe(faces_df, frame_idx, faces):
    """
    Add detected faces to the faces DataFrame.

    Args:
        faces_df (DataFrame): DataFrame for storing face data
        frame_idx (int): Current frame index
        faces (list): List of faces detected by DeepFace

    Returns:
        DataFrame: Updated DataFrame with new faces
    """
    new_df = faces_df.copy()

    for face_idx, face_dict in enumerate(faces):
        record = {
            "frame": frame_idx,
            "face_idx": face_idx,
            "face_x": face_dict.get("region", {}).get("x", 0),
            "face_y": face_dict.get("region", {}).get("y", 0),
            "face_w": face_dict.get("region", {}).get("w", 0),
            "face_h": face_dict.get("region", {}).get("h", 0),
        }

        # Add emotions
        if "emotion" in face_dict:
            for emotion, score in face_dict["emotion"].items():
                record[f"emotion_{emotion.lower()}"] = score
            record["dominant_emotion"] = face_dict.get("dominant_emotion", "")

        # Add age and gender
        record["age"] = face_dict.get("age", 0)
        record["gender"] = face_dict.get("gender", "")

        # Add facial landmarks
        if "landmarks" in face_dict:
            landmarks = face_dict["landmarks"]
            if "right_eye" in landmarks:
                record["right_eye_x"] = landmarks["right_eye"][0]
                record["right_eye_y"] = landmarks["right_eye"][1]
            if "left_eye" in landmarks:
                record["left_eye_x"] = landmarks["left_eye"][0]
                record["left_eye_y"] = landmarks["left_eye"][1]
            if "nose" in landmarks:
                record["nose_x"] = landmarks["nose"][0]
                record["nose_y"] = landmarks["nose"][1]
            if "mouth_right" in landmarks:
                record["mouth_right_x"] = landmarks["mouth_right"][0]
                record["mouth_right_y"] = landmarks["mouth_right"][1]
            if "mouth_left" in landmarks:
                record["mouth_left_x"] = landmarks["mouth_left"][0]
                record["mouth_left_y"] = landmarks["mouth_left"][1]

        # Add row to DataFrame
        new_df = pd.concat([new_df, pd.DataFrame([record])], ignore_index=True)

    return new_df


def extract_faces_from_image(
    frame, backend="opencv", features=["emotion", "age", "gender"], precision=6, debug=False
):
    """
    Extract facial data from a single image/frame using DeepFace.

    Args:
        frame (numpy.ndarray): The image/frame to process
        backend (str): Face detection backend to use
        features (list): List of analyses to perform (emotion, age, gender, race)
        precision (int): Number of decimal places for emotion probabilities
        debug (bool): Whether to print debug information

    Returns:
        list: List of dictionaries containing facial data
    """
    face_data_list = []

    try:
        # Analyze faces in the frame
        faces = DeepFace.analyze(
            img_path=frame, actions=features, enforce_detection=False, detector_backend=backend
        )

        # Process each face
        if isinstance(faces, dict):  # Single face
            faces = [faces]

        for face_idx, face in enumerate(faces):

            face_data = {"face_id": face_idx}

            # Add emotion data
            if "emotion" in features:
                emotions = face.get("emotion", {})
                if debug and emotions:
                    print(f"  Emotions: {', '.join([f'{k}:{v:.2f}' for k,v in emotions.items()])}")
                for emotion, score in emotions.items():
                    face_data[f"emotion_{emotion}"] = truncate_small_values(score, precision)
                face_data["dominant_emotion"] = (
                    max(emotions, key=emotions.get) if emotions else None
                )

            # Add age data
            if "age" in features:
                age = face.get("age", math.nan)
                face_data["age"] = age
                if debug:
                    print(f"  Age: {age}")

            # Add gender data
            if "gender" in features:
                gender = face.get("gender", {})
                if isinstance(gender, dict):  # Handle gender as dict with probabilities
                    for gender_key, score in gender.items():
                        face_data[f"gender_{gender_key}"] = truncate_small_values(score, precision)
                    face_data["gender"] = max(gender, key=gender.get) if gender else None
                else:  # Handle gender as string
                    face_data["gender"] = gender
                if debug:
                    print(f"  Gender: {face_data['gender']}")

            # Add facial region
            region = face.get("region", {})
            face_data["x"] = region.get("x", 0)
            face_data["y"] = region.get("y", 0)
            face_data["w"] = region.get("w", 0)
            face_data["h"] = region.get("h", 0)

            # Add facial landmarks if available
            landmarks = face.get("landmarks", {})
            for landmark, coords in landmarks.items():
                if isinstance(coords, list) and len(coords) >= 2:
                    face_data[f"{landmark}_x"] = coords[0]
                    face_data[f"{landmark}_y"] = coords[1]
                elif isinstance(coords, dict) and "x" in coords and "y" in coords:
                    face_data[f"{landmark}_x"] = coords["x"]
                    face_data[f"{landmark}_y"] = coords["y"]

            face_data_list.append(face_data)

    except Exception as e:
        if debug:
            print(f"Error processing image: {str(e)}")

    return face_data_list


def extract_faces_from_video(
    video_path,
    output_file=None,
    backend="opencv",
    skip_frames=0,
    features=("emotion", "age", "gender"),
    precision=5,
):
    """
    Extract faces from video frames using DeepFace.

    Args:
        video_path: Path to video file
        output_file: Path to output CSV file
        backend: Face detection backend to use
        skip_frames: Process every nth frame
        features: Tuple of features to extract
        precision: Decimal place precision for face coordinates

    Returns:
        Dictionary with DataFrame and output file path
    """
    print(f"Video: {video_path}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {"faces_path": None, "faces_df": None}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Dimensions: {width}x{height}, FPS: {fps}, Frames: {frame_count}")

    # Create output file if not specified
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = f"{base_name}_faces.csv"

    # Initialize an empty list for all face data
    all_faces_data = []

    # Process each frame
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if needed
        if skip_frames > 0 and frame_number % skip_frames != 0:
            frame_number += 1
            continue

        # Calculate timestamp from frame number and fps
        timestamp = frame_number / fps if fps > 0 else 0

        # Extract faces from the frame
        try:
            faces_in_frame = extract_faces_from_image(
                frame, backend=backend, features=features, precision=precision
            )

            # Add frame number and timestamp to each face entry
            for face in faces_in_frame:
                face_data = {
                    "frame": frame_number,  # Ensure Frame is included and named properly
                    "timestamp": round(timestamp, 3),  # Round timestamp to 3 decimal places
                    **face,  # Unpack the rest of the face data
                }
                all_faces_data.append(face_data)

        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")

        frame_number += 1

    cap.release()

    # Create the DataFrame with explicit column ordering
    if all_faces_data:
        # Ensure Frame and timestamp are the first columns
        columns = ["frame", "timestamp"] + [
            col for col in all_faces_data[0].keys() if col not in ["frame", "timestamp"]
        ]
        df = pd.DataFrame(all_faces_data)
        df = df[columns]  # Reorder columns to ensure frame and timestamp are first

        # Save to CSV with headers
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} facial data points to {output_file}")

        return {"faces_path": output_file, "faces_df": df}
    else:
        print("No faces detected in the video")
        return {"faces_path": None, "faces_df": None}


def normalize_facial_keypoints(faces_df, height, width):
    """
    Normalize facial keypoint coordinates by dividing by video dimensions.

    Args:
        faces_df (DataFrame): DataFrame with facial keypoints
        height (int): Video height
        width (int): Video width

    Returns:
        DataFrame: DataFrame with normalized coordinates
    """
    # Create a copy to avoid modifying the original
    normed_df = faces_df.copy()

    # Get columns that contain coordinate information
    x_cols = [col for col in normed_df.columns if col.endswith("_x") or col == "x"]
    y_cols = [col for col in normed_df.columns if col.endswith("_y") or col == "y"]

    # Also normalize width and height columns if present
    if "w" in normed_df.columns:
        normed_df["w"] = normed_df["w"] / width

    if "h" in normed_df.columns:
        normed_df["h"] = normed_df["h"] / height

    # Normalize x coordinates by dividing by width
    for col in x_cols:
        if col in normed_df.columns:
            normed_df[col] = normed_df[col] / width

    # Normalize y coordinates by dividing by height
    for col in y_cols:
        if col in normed_df.columns:
            normed_df[col] = normed_df[col] / height

    return normed_df


def match_faces_to_poses(faces_df, poses_df):
    """
    Match facial data with pose keypoints based on frame number.

    Args:
        faces_df (DataFrame): DataFrame containing facial data
        poses_df (DataFrame): DataFrame containing pose keypoints

    Returns:
        DataFrame: Merged DataFrame with both face and pose data
    """
    if faces_df.empty or poses_df.empty:
        return pd.DataFrame()

    # Ensure both dataframes have 'frame' column
    if "frame" not in faces_df.columns or "frame" not in poses_df.columns:
        raise ValueError("Both dataframes must have 'frame' column")

    # Merge dataframes on frame number
    merged_df = pd.merge(faces_df, poses_df, on="frame", how="inner", suffixes=("_face", "_pose"))

    # Handle potential multiple faces per frame by selecting the largest face
    # or the one closest to the pose keypoints
    if len(merged_df) > len(poses_df):
        # Group by frame and select best match
        merged_df = (
            merged_df.groupby("frame").apply(select_best_face_for_pose).reset_index(drop=True)
        )

    return merged_df


def select_best_face_for_pose(group):
    """
    Select the best matching face for a pose based on size and position.
    """
    # If only one face, return it
    if len(group) == 1:
        return group.iloc[0]

    # Calculate face size
    group["face_size"] = group["w"] * group["h"]

    # Find pose keypoints for the nose if available
    nose_x, nose_y = None, None
    if "nose_x" in group.columns and "nose_y" in group.columns:
        nose_cols = [col for col in group.columns if "nose_x" in col or "nose_y" in col]
        if len(nose_cols) >= 2:
            nose_x_col = [col for col in nose_cols if "_x" in col][0]
            nose_y_col = [col for col in nose_cols if "_y" in col][0]
            nose_x = group[nose_x_col].values[0]
            nose_y = group[nose_y_col].values[0]

    # If nose position is available, find the closest face
    if nose_x is not None and nose_y is not None:
        group["face_center_x"] = group["x"] + group["w"] / 2
        group["face_center_y"] = group["y"] + group["h"] / 2
        group["distance_to_nose"] = np.sqrt(
            (group["face_center_x"] - nose_x) ** 2 + (group["face_center_y"] - nose_y) ** 2
        )
        # Return face with smallest distance to nose
        return group.loc[group["distance_to_nose"].idxmin()]

    # If no nose position, return the largest face
    return group.loc[group["face_size"].idxmax()]


def process_video_faces(
    video_path,
    output_dir,
    video_metadata=None,
    poses_df=None,
    skip_frames=0,
    backend="opencv",
    force_process=False,
    precision=6,
    debug=False,
):
    """
    End-to-end processing of facial data from a video.

    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save output files
        video_metadata (dict): Video metadata including Height and Width
        poses_df (DataFrame, optional): Pose data for matching
        skip_frames (int): Number of frames to skip
        backend (str): Face detection backend
        force_process (bool): Force reprocessing even if files exist
        precision (int): Number of decimal places for emotion probabilities
        debug (bool): Whether to print debug information

    Returns:
        dict: Paths to generated files
    """
    results = {"faces": None, "normed": None, "matched": None}

    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]

    # Step 1: Extract facial data
    faces_path = os.path.join(output_dir, f"{base_name}_faces.csv")
    faces_df = None

    if force_process or not os.path.exists(faces_path):
        print(f"Extracting facial data from {video_name}...")
        extract_result = extract_faces_from_video(
            video_path=video_path,
            output_file=faces_path,
            backend=backend,
            skip_frames=skip_frames,
            precision=precision,
        )
        results["faces"] = extract_result["faces_path"]
        faces_df = extract_result["faces_df"]
    else:
        print(f"Using existing facial data for {video_name}")
        results["faces"] = faces_path
        try:
            faces_df = pd.read_csv(faces_path)
        except Exception as e:
            print(f"Error reading existing faces file: {e}")

    # Step 2: Normalize facial coordinates
    if faces_df is not None and len(faces_df) > 0 and video_metadata:
        normed_path = os.path.join(output_dir, f"{base_name}_faces_normed.csv")

        if force_process or not os.path.exists(normed_path):
            print(f"Normalizing facial coordinates for {video_name}...")
            height = video_metadata.get("Height", 0)
            width = video_metadata.get("Width", 0)

            normed_df = normalize_facial_keypoints(faces_df, height, width)
            normed_df.to_csv(normed_path, index=False)
            results["normed"] = normed_path
        else:
            print(f"Using existing normalized data for {video_name}")
            results["normed"] = normed_path

    # Step 3: Match with poses if available
    if poses_df is not None and faces_df is not None and not faces_df.empty:
        matched_path = os.path.join(output_dir, f"{base_name}_faces_matched.csv")

        if force_process or not os.path.exists(matched_path):
            print(f"Matching faces to poses for {video_name}...")
            matched_df = match_faces_to_poses(faces_df, poses_df)

            if not matched_df.empty:
                matched_df.to_csv(matched_path, index=False)
                results["matched"] = matched_path
                print(f"Saved matched data with {len(matched_df)} records")
            else:
                print("No matches found between faces and poses")
        else:
            print(f"Using existing matched data for {video_name}")
            results["matched"] = matched_path

    return results


def get_facial_stats(faces_df, debug=False):
    """
    Calculate statistics from facial data.

    Args:
        faces_df (DataFrame): DataFrame with facial data
        debug (bool): Whether to print debug information

    Returns:
        dict: Dictionary containing various statistics grouped by face_id
    """
    stats = {}

    if faces_df.empty:
        return {"total_faces": 0, "unique_frames": 0, "avg_faces_per_frame": 0}

    # Basic counts (overall)
    stats["total_faces"] = len(faces_df)
    stats["unique_frames"] = faces_df["frame"].nunique()
    stats["avg_faces_per_frame"] = (
        stats["total_faces"] / stats["unique_frames"] if stats["unique_frames"] > 0 else 0
    )

    if debug:
        print(f"Face statistics: {stats['total_faces']} faces in {stats['unique_frames']} frames")
        print(f"Average faces per frame: {stats['avg_faces_per_frame']:.2f}")

    # Group by face_id
    if "face_id" in faces_df.columns:
        face_groups = faces_df.groupby("face_id")
        stats["face_data"] = {}

        for face_id, group in face_groups:
            face_stats = {
                "count": len(group),
                "frames": group["frame"].nunique(),
                "first_frame": group["frame"].min(),
                "last_frame": group["frame"].max(),
            }

            # Emotion analysis if available
            emotion_cols = [col for col in group.columns if col.startswith("emotion_")]
            if emotion_cols:
                emotion_distribution = {}
                for col in emotion_cols:
                    emotion_name = col.replace("emotion_", "")
                    emotion_distribution[emotion_name] = group[col].mean()
                face_stats["emotion_distribution"] = emotion_distribution

                # Get dominant emotion per frame for this face
                emotion_only = group[emotion_cols]
                face_stats["dominant_emotions"] = (
                    emotion_only.idxmax(axis=1).str.replace("emotion_", "").value_counts().to_dict()
                )

            stats["face_data"][face_id] = face_stats

            if debug:
                print(f"\nFace ID {face_id} statistics:")
                print(f"  Detected in {face_stats['count']} frames")
                if "dominant_emotions" in face_stats:
                    print("  Dominant emotions:")
                    for emotion, count in face_stats["dominant_emotions"].items():
                        print(
                            f"    {emotion}: {count} frames ({count/face_stats['count']*100:.1f}%)"
                        )

    # Emotion analysis across all faces (kept for backward compatibility)
    emotion_cols = [col for col in faces_df.columns if col.startswith("emotion_")]
    if emotion_cols:
        emotion_distribution = {}
        for col in emotion_cols:
            emotion_name = col.replace("emotion_", "")
            emotion_distribution[emotion_name] = faces_df[col].mean()
        stats["emotion_distribution"] = emotion_distribution

        # Get dominant emotion per frame
        emotion_only = faces_df[emotion_cols]
        stats["dominant_emotions"] = (
            emotion_only.idxmax(axis=1).str.replace("emotion_", "").value_counts().to_dict()
        )

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video files to extract facial data")
    parser.add_argument("video_path", type=str, help="Path to video file or directory of videos")
    parser.add_argument("--output", "-o", type=str, default="./data", help="Output directory")
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="ssd",
        choices=["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8", "yunet"],
        help="DeepFace detector backend",
    )
    parser.add_argument(
        "--skip", "-s", type=int, default=0, help="Frames to skip between processing"
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Force reprocessing of existing files"
    )
    parser.add_argument(
        "--precision", "-p", type=int, default=8, help="Decimal precision for emotion values"
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Print debug information")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.video_path):
        for file in os.listdir(args.video_path):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_file = os.path.join(args.video_path, file)
                print(f"Processing {file}...")
                process_video_faces(
                    video_file,
                    args.output,
                    skip_frames=args.skip,
                    backend=args.backend,
                    force_process=args.force,
                    precision=args.precision,
                    debug=args.debug,
                )
    else:
        process_video_faces(
            args.video_path,
            args.output,
            skip_frames=args.skip,
            backend=args.backend,
            force_process=args.force,
            precision=args.precision,
            debug=args.debug,
        )
