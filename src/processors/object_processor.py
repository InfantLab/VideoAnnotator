import os
import cv2
import pandas as pd
from ultralytics import YOLO


def extract_objects_from_video(video_path, model=None, confidence=0.4, tracker="bytetrack"):
    """
    Extract objects from a video using YOLOv8.

    Args:
        video_path (str): Path to the video file
        model: YOLOv8 model for object detection (if None, loads default)
        confidence (float): Confidence threshold for object detection
        tracker (str): Object tracker to use ('bytetrack', 'botsort', etc.)

    Returns:
        DataFrame: DataFrame containing detected objects
    """
    print(f"Processing video for objects: {video_path}")

    # Load model if not provided
    if model is None:
        model = YOLO("yolov8n.pt")

    # Process video with YOLOv8
    results = model.track(
        source=video_path, stream=True, conf=confidence, save=False, tracker=tracker
    )

    # Prepare dataframe
    data = []

    # Process each frame's results
    for i, result in enumerate(results):
        boxes = result.boxes

        # Process each detected object
        if len(boxes) == 0:
            continue

        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Get object class and name
            class_id = int(box.cls.cpu().numpy()[0])
            class_name = model.names[class_id]

            # Get confidence score
            conf = float(box.conf.cpu().numpy()[0])

            # Get tracking ID if available
            track_id = -1
            if hasattr(box, "id") and box.id is not None:
                track_id = int(box.id.cpu().numpy()[0])

            # Add to data
            data.append(
                {
                    "frame": i,
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def normalize_object_coordinates(objects_df, video_height, video_width):
    """
    Normalize object bounding box coordinates by video dimensions.

    Args:
        objects_df (DataFrame): DataFrame with object bounding boxes
        video_height (int): Video height
        video_width (int): Video width

    Returns:
        DataFrame: DataFrame with normalized coordinates
    """
    if objects_df is None or len(objects_df) == 0:
        return objects_df

    # Create a copy to avoid modifying the original
    normed_df = objects_df.copy()

    # Normalize x coordinates
    normed_df["x1"] = normed_df["x1"] / video_width
    normed_df["x2"] = normed_df["x2"] / video_width

    # Normalize y coordinates
    normed_df["y1"] = normed_df["y1"] / video_height
    normed_df["y2"] = normed_df["y2"] / video_height

    return normed_df


def match_objects_to_persons(objects_df, poses_df, distance_threshold=0.2):
    """
    Match detected objects to persons based on spatial proximity.

    Args:
        objects_df (DataFrame): DataFrame with object detections
        poses_df (DataFrame): DataFrame with pose keypoints
        distance_threshold (float): Maximum normalized distance for a match

    Returns:
        DataFrame: Original objects_df with an additional 'person_id' column
    """
    if objects_df is None or len(objects_df) == 0 or poses_df is None or len(poses_df) == 0:
        if objects_df is not None:
            objects_df["person_id"] = -1
        return objects_df

    # Create a copy of objects DataFrame to add the person_id
    matched_df = objects_df.copy()
    matched_df["person_id"] = -1  # Default to -1 (no match)

    # Calculate object centers
    matched_df["center_x"] = (matched_df["x1"] + matched_df["x2"]) / 2
    matched_df["center_y"] = (matched_df["y1"] + matched_df["y2"]) / 2

    # Get unique frames
    frames = objects_df["frame"].unique()

    # Process each frame
    for frame in frames:
        # Get objects in this frame
        frame_objects = matched_df[matched_df["frame"] == frame]

        # Get poses in this frame
        frame_poses = poses_df[poses_df["frame"] == frame]

        if len(frame_poses) == 0:
            continue

        # For each object, find the closest person
        for obj_idx, obj in frame_objects.iterrows():
            obj_center_x = obj["center_x"]
            obj_center_y = obj["center_y"]

            best_dist = float("inf")
            best_id = -1

            # Check each pose
            for _, pose in frame_poses.iterrows():
                # Calculate average position of upper body keypoints (0:10)
                pose_x = 0
                pose_y = 0
                valid_points = 0

                for i in range(10):  # Upper body keypoints
                    if pose[f"kpt_{i}_x"] > 0 and pose[f"kpt_{i}_y"] > 0:
                        pose_x += pose[f"kpt_{i}_x"]
                        pose_y += pose[f"kpt_{i}_y"]
                        valid_points += 1

                if valid_points > 0:
                    pose_x /= valid_points
                    pose_y /= valid_points

                    # Calculate distance
                    dist = ((obj_center_x - pose_x) ** 2 + (obj_center_y - pose_y) ** 2) ** 0.5

                    # Update if this is the closest pose within threshold
                    if dist < best_dist and dist < distance_threshold:
                        best_dist = dist
                        best_id = pose["id"]

            # Assign the closest pose ID to this object
            matched_df.loc[obj_idx, "person_id"] = best_id

    # Drop helper columns
    return matched_df.drop(["center_x", "center_y"], axis=1)


def process_video_objects(
    video_path,
    output_dir,
    video_metadata=None,
    poses_df=None,
    confidence=0.4,
    model_path="yolov8n.pt",
    force_process=False,
):
    """
    Process a video to extract object data, normalize coordinates, and match with poses.

    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save results
        video_metadata (dict): Dictionary containing video metadata
        poses_df (DataFrame): DataFrame with pose keypoints
        confidence (float): Confidence threshold for object detection
        model_path (str): Path to YOLOv8 model
        force_process (bool): Force reprocessing of existing files

    Returns:
        dict: Dictionary with paths to result files
    """
    basename = os.path.basename(video_path)
    stemname, _ = os.path.splitext(basename)

    # File paths
    objects_path = os.path.join(output_dir, f"{stemname}.objects.csv")
    normed_path = os.path.join(output_dir, f"{stemname}.objects_normed.csv")
    matched_path = os.path.join(output_dir, f"{stemname}.objects_matched.csv")

    results = {"objects": None, "normed": None, "matched": None}

    # Check if files already exist and we're not forcing reprocess
    if not force_process and os.path.exists(objects_path):
        results["objects"] = objects_path
        if os.path.exists(normed_path):
            results["normed"] = normed_path
        if os.path.exists(matched_path):
            results["matched"] = matched_path

        # If all files exist, return early
        if all(results.values()):
            print(f"Already processed object data for {basename}")
            return results

    # Load model
    model = YOLO(model_path)

    # Extract objects
    objects_df = extract_objects_from_video(video_path, model, confidence)

    if objects_df is None or len(objects_df) == 0:
        print(f"No objects detected in {basename}")
        return results

    # Save objects data
    objects_df.to_csv(objects_path, index=False)
    results["objects"] = objects_path

    # Get video metadata if not provided
    height = None
    width = None
    if video_metadata:
        height = video_metadata.get("Height")
        width = video_metadata.get("Width")

    if height is None or width is None:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    # Normalize coordinates
    normed_df = normalize_object_coordinates(objects_df, height, width)
    normed_df.to_csv(normed_path, index=False)
    results["normed"] = normed_path

    # Match with poses if available
    if poses_df is not None:
        matched_df = match_objects_to_persons(normed_df, poses_df)
        matched_df.to_csv(matched_path, index=False)
        results["matched"] = matched_path

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video files to extract object data")
    parser.add_argument("video_path", type=str, help="Path to video file or directory of videos")
    parser.add_argument("--output", "-o", type=str, default="./data", help="Output directory")
    parser.add_argument("--confidence", "-c", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--model", "-m", type=str, default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument(
        "--force", "-f", action="store_true", help="Force reprocessing of existing files"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.video_path):
        for file in os.listdir(args.video_path):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_file = os.path.join(args.video_path, file)
                print(f"Processing {file}...")
                process_video_objects(
                    video_file,
                    args.output,
                    confidence=args.confidence,
                    model_path=args.model,
                    force_process=args.force,
                )
    else:
        process_video_objects(
            args.video_path,
            args.output,
            confidence=args.confidence,
            model_path=args.model,
            force_process=args.force,
        )
