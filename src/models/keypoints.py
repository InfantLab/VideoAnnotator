"""
Module for keypoint definitions and related constants.

This module centralizes all definitions related to keypoints, including
keypoint names, coordinate systems, and other constants.
"""

import pandas as pd

# Standard keypoint names in YOLO/COCO format
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_hip", "right_hip", 
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Coordinate components for each keypoint
COORDINATE_COMPONENTS = ["x", "y", "conf"]

def get_keypoint_columns():
    """
    Get lists of x and y coordinate column names based on keypoint definitions.
    
    Returns:
        tuple: (x_columns, y_columns) - Lists of column names for x and y coordinates
    """
    x_cols = [f"{name}.x" for name in KEYPOINT_NAMES]
    y_cols = [f"{name}.y" for name in KEYPOINT_NAMES]
    return x_cols, y_cols

def get_confidence_columns():
    """
    Get list of confidence column names based on keypoint definitions.
    
    Returns:
        list: List of column names for confidence values
    """
    return [f"{name}.conf" for name in KEYPOINT_NAMES]

def get_face_columns():
    """
    Get lists of face-related column names (first 5 keypoints).
    
    Returns:
        tuple: (x_face_cols, y_face_cols) - Lists of column names for face x and y coordinates
    """
    x_cols, y_cols = get_keypoint_columns()
    # First 5 keypoints are face-related (nose, eyes, ears)
    return x_cols[:5], y_cols[:5]

def create_keypoints_df():
    """
    Create an empty dataframe for keypoints with the proper column structure.
    
    Returns:
        DataFrame: Empty keypoints dataframe with standardized column names
    """
    # Base columns for identification and bounding box
    columns = ["frame", "person", "index", "bbox.x1", "bbox.y1", "bbox.x2", "bbox.y2", "bbox.conf"]
    
    # Add keypoint columns (x, y, conf for each keypoint)
    for name in KEYPOINT_NAMES:
        for component in COORDINATE_COMPONENTS:
            columns.append(f"{name}.{component}")
    
    return pd.DataFrame(columns=columns)
