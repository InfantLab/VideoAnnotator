"""
Central configuration file for the babyjokes project.
Contains all configurable parameters used across the application.
"""

import os

# Model configurations
MODEL_CONFIG = {"pose_model": "yolov8n-pose.pt", "confidence_threshold": 0.5}

# File paths and directories
PATH_CONFIG = {
    "default_metadata_file": "_LookitLaughter.test.xlsx",
    "processed_videos_file": "processed_videos.csv",
    "videos_in": os.path.join("..", "LookitLaughter.test"),
    "data_out": os.path.join("..", "data", "1_interim"),
    "demo_data": os.path.join("..", "data", "demo"),
}

# Video processing parameters
VIDEO_CONFIG = {"track": True, "interpolate_missing": True}

# Keypoint processing parameters
KEYPOINT_CONFIG = {"num_keypoints": 17, "confidence_threshold": 0.5}
