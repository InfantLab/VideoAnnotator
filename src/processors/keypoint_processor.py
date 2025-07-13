"""
Module for keypoint processing operations.

This module contains functions for processing keypoint data, including
normalization, filtering, and other transformations needed for analysis.
"""

from src.utils.keypoint_utils import (
    normalize_keypoints,
    filter_keypoints_by_confidence,
    interpolate_missing_keypoints,
)
from src.config import KEYPOINT_CONFIG


def process_keypoints_for_modeling(keypoints_df, height, width):
    """
    Process keypoint data for modeling, including normalization,
    confidence filtering, and interpolation.

    This function applies a standard processing pipeline to raw keypoint data:
    1. Filter out low-confidence keypoints
    2. Normalize coordinates to [0,1] range
    3. Interpolate missing values
    4. Optionally compute additional features

    Args:
        keypoints_df (DataFrame): Raw keypoints DataFrame
        height (int): Video height
        width (int): Video width

    Returns:
        DataFrame: Processed keypoints ready for modeling
    """
    # Step 1: Filter low confidence keypoints
    filtered_df = filter_keypoints_by_confidence(
        keypoints_df, confidence_threshold=KEYPOINT_CONFIG["confidence_threshold"]
    )

    # Step 2: Normalize keypoint coordinates
    normalized_df = normalize_keypoints(filtered_df, height, width)

    # Step 3: Interpolate missing keypoints if needed
    if KEYPOINT_CONFIG.get("interpolate_missing", True):
        processed_df = interpolate_missing_keypoints(normalized_df)
    else:
        processed_df = normalized_df

    # Step 4: Calculate additional features if needed
    # (e.g., velocities, accelerations, etc.)

    return processed_df
