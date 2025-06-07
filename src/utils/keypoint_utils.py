"""
Utility functions for keypoint processing.

This module contains functions for common operations on keypoint data,
such as normalization, filtering, and interpolation.
"""

import pandas as pd
import numpy as np
from src.config import KEYPOINT_CONFIG
from src.models.keypoints import get_keypoint_columns, KEYPOINT_NAMES

def normalize_keypoints(keypoints_df, height, width):
    """
    Normalize keypoint coordinates by dividing x by width and y by height.
    
    This standardizes all coordinates to the range [0, 1], making them
    invariant to the original video dimensions.
    
    Args:
        keypoints_df (DataFrame): DataFrame with keypoint coordinates
        height (int): Video frame height
        width (int): Video frame width
    
    Returns:
        DataFrame: DataFrame with normalized coordinates
    """
    normalized_df = keypoints_df.copy()
    
    # Get coordinate columns
    x_cols, y_cols = get_keypoint_columns()
    
    # Normalize bounding box coordinates
    bbox_cols = ['bbox.x1', 'bbox.y1', 'bbox.x2', 'bbox.y2']
    h_or_w = np.array([width, height, width, height])  
    for col in bbox_cols:
        if col in normalized_df.columns:
            # Normalize bounding box coordinates
            normalized_df[col] = normalized_df[col] / h_or_w[bbox_cols.index(col)] 
    
    # Normalize x coordinates by width
    for col in x_cols:
        if col in normalized_df.columns:
            normalized_df[col] = normalized_df[col] / width
    
    # Normalize y coordinates by height
    for col in y_cols:
        if col in normalized_df.columns:
            normalized_df[col] = normalized_df[col] / height
    
    return normalized_df

def filter_keypoints_by_confidence(keypoints_df, confidence_threshold=None):
    """
    Filter keypoints by confidence threshold.
    
    Sets coordinates to NaN for keypoints with confidence below threshold.
    
    Args:
        keypoints_df (DataFrame): DataFrame with keypoint coordinates and confidences
        confidence_threshold (float): Minimum confidence value, defaults to config value
    
    Returns:
        DataFrame: Filtered DataFrame with low-confidence points set to NaN
    """
    if confidence_threshold is None:
        confidence_threshold = KEYPOINT_CONFIG['confidence_threshold']
        
    filtered_df = keypoints_df.copy()
    
    # For each keypoint, check confidence and set coordinates to NaN if below threshold
    for name in KEYPOINT_NAMES:
        conf_col = f"{name}.conf"
        x_col = f"{name}.x"
        y_col = f"{name}.y"
        
        if conf_col in filtered_df.columns and x_col in filtered_df.columns and y_col in filtered_df.columns:
            # Create a mask for low-confidence points
            mask = filtered_df[conf_col] < confidence_threshold
            
            # Set coordinates to NaN for low-confidence points
            filtered_df.loc[mask, x_col] = np.nan
            filtered_df.loc[mask, y_col] = np.nan
    
    return filtered_df

def interpolate_missing_keypoints(keypoints_df):
    """
    Interpolate missing keypoint coordinates within each track.
    
    Uses linear interpolation to fill NaN values for each person/track
    across frames, helping to smooth tracking and fill missing detections.
    
    Args:
        keypoints_df (DataFrame): DataFrame with keypoint coordinates
    
    Returns:
        DataFrame: DataFrame with interpolated coordinates
    """
    interpolated_df = keypoints_df.copy()
    
    # Get coordinate columns
    x_cols, y_cols = get_keypoint_columns()
    all_coord_cols = x_cols + y_cols
    
    # Check if 'id' column exists to identify tracks
    if 'id' in interpolated_df.columns:
        # For each track ID
        for track_id in interpolated_df['id'].unique():
            mask = interpolated_df['id'] == track_id
            track_data = interpolated_df.loc[mask].sort_values('frame')
            
            # Interpolate all coordinate columns
            for col in all_coord_cols:
                if col in track_data.columns:
                    interpolated_df.loc[mask, col] = track_data[col].interpolate(method='linear')
    else:
        # If no track ID, interpolate by person index
        for idx in interpolated_df['index'].unique():
            mask = interpolated_df['index'] == idx
            track_data = interpolated_df.loc[mask].sort_values('frame')
            
            # Interpolate all coordinate columns
            for col in all_coord_cols:
                if col in track_data.columns:
                    interpolated_df.loc[mask, col] = track_data[col].interpolate(method='linear')
    
    return interpolated_df

# Legacy function aliases for backward compatibility
getkeypointcols = get_keypoint_columns
getfacecols = get_keypoint_columns
