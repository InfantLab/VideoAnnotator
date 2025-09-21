"""
Utility functions for transforming coordinates and bounding boxes.
"""

import numpy as np


def normaliseCoordinates(df, xcols, ycols, frameHeight, frameWidth):
    """
    Normalise the x and y pixel based coordinates to be between 0 and 1.

    Args:
        df (DataFrame): Dataframe with coordinate columns
        xcols (list): List of x coordinate column indices
        ycols (list): List of y coordinate column indices
        frameHeight (int): Height of the frame
        frameWidth (int): Width of the frame

    Returns:
        DataFrame: Dataframe with normalised coordinates
    """
    for col in xcols:
        df.iloc[:, [col]] = df.iloc[:, [col]] / frameWidth
    for col in ycols:
        df.iloc[:, [col]] = df.iloc[:, [col]] / frameHeight
    return df


def denormaliseCoordinates(df, xcols, ycols, frameHeight, frameWidth):
    """
    For normalised x and y coordinates (between 0 and 1) convert back to pixel based coordinates.

    Args:
        df (DataFrame): Dataframe with normalised coordinate columns
        xcols (list): List of x coordinate column indices
        ycols (list): List of y coordinate column indices
        frameHeight (int): Height of the frame
        frameWidth (int): Width of the frame

    Returns:
        DataFrame: Dataframe with pixel-based coordinates
    """
    for col in xcols:
        df[col] = df[col] * frameWidth
    for col in ycols:
        df[col] = df[col] * frameHeight
    return df


def xyxy2ltwh(bbox):
    """
    Convert [x1 y1 x2 y2] box format to [x y w h] format.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format

    Returns:
        list: Bounding box in [x, y, w, h] format
    """
    if isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
