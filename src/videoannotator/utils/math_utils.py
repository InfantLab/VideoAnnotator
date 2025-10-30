"""Mathematical utility functions for calculations on keypoints and audio.

data.
"""

import numpy as np


def avgxys(xyc, threshold=0.5):
    """Calculate average x and y for keypoints with confidence above threshold.

    Args:
        xyc: [nrows x 3] array of x,y,conf values.
        threshold: confidence threshold.

    Returns:
        avgx, avgy
    """
    # get the x,y values where conf > threshold
    x = xyc[:, 0]
    y = xyc[:, 1]
    conf = xyc[:, 2]
    x = x[conf > threshold]
    y = y[conf > threshold]
    # calculate the average
    avgx = np.mean(x)
    avgy = np.mean(y)
    return avgx, avgy


def stdevxys(xyc, threshold=0.5):
    """Calculate standard deviation of x and y for keypoints above threshold.

    Args:
        xyc: [nrows x 3] array of x,y,conf values.
        threshold: confidence threshold.

    Returns:
        stdx, stdy
    """
    # get the x,y values where conf > threshold
    x = xyc[:, 0]
    y = xyc[:, 1]
    conf = xyc[:, 2]
    x = x[conf > threshold]
    y = y[conf > threshold]
    # calculate the average
    stdevx = np.std(x)
    stdevy = np.std(y)
    return stdevx, stdevy


def rowcogs(keypoints1d, threshold=0.5):
    """Return average x and y for a 1-D array of keypoints.

    Args:
        keypoints1d: 1d array of keypoints.
        threshold: confidence threshold.

    Returns:
        avgx, avgy
    """
    # get the x,y values where conf > threshold
    xyc3 = keypoints1d.to_numpy().reshape(-1, 3)
    return avgxys(xyc3, threshold)


def rowstds(keypoints1d, threshold=0.5):
    """Return standard deviations for a 1-D array of keypoints.

    Args:
        keypoints1d: 1d array of keypoints.
        threshold: confidence threshold.

    Returns:
        stdx, stdy
    """
    xycs3 = keypoints1d.to_numpy().reshape(-1, 3)
    return stdevxys(xycs3, threshold)


def centreOfGravity(df, frames=(), people="all", bodypart="whole"):
    """Compute center of gravity columns for keypoints in a DataFrame.

        Compute average position of a bodypart across frames and people and add
        `cog.x` and `cog.y` columns to the DataFrame. Useful for plotting time
        series of movement.

    Args:
        df (DataFrame): Dataframe of keypoints.
        frames (list): List of frames to include.
        people (list or str): List of people to include, or 'all'.
        bodypart (str): Which bodypart to use, default is "whole" for all keypoints.

    Returns:
        DataFrame: Dataframe with added center of gravity columns.
    """
    if len(frames) == 0:
        frames = df.frame.unique()

    if people == "all":
        people = df.person.unique()

    if bodypart != "whole":
        raise NotImplementedError("Only whole body implemented for now")

    threshold = 0.5

    # create new columns for the centre of gravity
    df["cog.x"] = np.nan
    df["cog.y"] = np.nan

    for frame in frames:
        for person in people:
            # get the keypoints for this person in this frame
            kpts = df[(df["frame"] == frame) & (df["person"] == person)]

            if not kpts.empty:
                # get the average position of the bodypart
                if bodypart == "whole":
                    xyc = kpts.iloc[:, 8:59].to_numpy()  # just keypoints
                    xyc = xyc.reshape(-1, 3)  # reshape to n x 3 array (x,y,conf)
                    avgx, avgy = avgxys(xyc, threshold)

                df.loc[(df["frame"] == frame) & (df["person"] == person), "cog.x"] = (
                    avgx
                )
                df.loc[(df["frame"] == frame) & (df["person"] == person), "cog.y"] = (
                    avgy
                )

    return df
