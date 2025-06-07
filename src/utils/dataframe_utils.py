"""
Utility functions for dataframe creation and manipulation.
"""

import pandas as pd
import numpy as np
import torch
import ultralytics.utils as ultrautils

def createKeypointsDF():
    """
    Create empty dataframe to store keypoints, one per person per frame.
    
    Returns:
        DataFrame: Empty dataframe with keypoint columns
    """
    bodyparts = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    coords = ["x", "y", "c"]
    bodypartsxy = [f"{bp}.{c}" for bp in bodyparts for c in coords]
    boundingbox = ["bbox.x1", "bbox.y1", "bbox.x2", "bbox.y2", "bbox.c"]
    cols = ["frame", "person", "index"] + boundingbox + bodypartsxy
    return pd.DataFrame(columns=cols)


def addKeypointsToDF(df, framenumber, bbox, bconf, keypointsdata):
    """
    Take output from yolov8 and add to dataframe, person by person.
    
    Args:
        df (DataFrame): Keypoints dataframe
        framenumber (int): Frame number
        bbox (array): Bounding box data
        bconf (array): Bounding box confidence
        keypointsdata (array): Keypoints data
        
    Returns:
        DataFrame: Updated keypoints dataframe
    """
    for idx in range(len(bbox)):
        person = "child" if idx == 0 else "adult"
        row = [int(framenumber), person, idx]
        # YOLO returns bounding boxes as [centre.x,centre.y,w,h]
        # but for consistency with other models we want [x1,y1,x2,y2]
        bb = ultrautils.ops.xywh2xyxy(bbox[idx])
        row += bb.tolist()
        row += torch.flatten(bconf[idx]).tolist()
        row += torch.flatten(keypointsdata[idx]).tolist()
        # add row to dataframe
        df.loc[len(df)] = row
    return df


def createfacesdf():
    """
    Creates a dataframe with the facial data from the videos.
    
    Returns:
        DataFrame: Empty faces dataframe
    """
    cols = [
        "frame",
        "person",
        "index",
        "bbox.x1",
        "bbox.y1",
        "bbox.x2",
        "bbox.y2",
        "emotion",
        "age",
        "gender",
    ]
    return pd.DataFrame(columns=cols)


def addfacestodf(facesdf, frameidx, facedata):
    """
    Add the faces identified by face detection model to the dataframe, 
    along with emotion, age and gender.
    
    Args:
        facesdf (DataFrame): Faces dataframe
        frameidx (int): Frame index
        facedata (list): Face detection data
        
    Returns:
        DataFrame: Updated faces dataframe
    """
    for idx, face in enumerate(facedata):
        newrow = {
            "frame": frameidx,
            "person": "unknown",
            "index": idx,
            "bbox.x1": face["region"]["x"],
            "bbox.y1": face["region"]["y"],
            "bbox.x2": face["region"]["x"] + face["region"]["w"],
            "bbox.y2": face["region"]["y"] + face["region"]["h"],
            "emotion": face["dominant_emotion"],
            "age": face["age"],
            "gender": face["dominant_gender"],
        }
        facesdf.loc[len(facesdf)] = newrow
    return facesdf


def relabelPersonIndex(
    df,
    person=None,
    index=None,
    newPerson=None,
    newIndex=None,
    startFrame=None,
    endFrame=None,
):
    """
    Replace person and/or index values with new values for a range of frames.
    
    Args:
        df (DataFrame): DataFrame with person and index columns
        person (str): Original person value
        index (int): Original index value
        newPerson (str): New person value
        newIndex (int): New index value
        startFrame (int): Start frame for the change
        endFrame (int): End frame for the change
        
    Returns:
        DataFrame: Updated dataframe
    """
    if startFrame is None:
        startFrame = 0
    if endFrame is None:
        endFrame = df["frame"].max()
    if person is None and index is None:
        return df
        
    if person is not None and index is None:
        # just person
        df.loc[
            (df["frame"] >= startFrame)
            & (df["frame"] <= endFrame)
            & (df["person"] == person),
            "person",
        ] = newPerson
        
    if person is None and index is not None:
        # just index
        df.loc[
            (df["frame"] >= startFrame)
            & (df["frame"] <= endFrame)
            & (df["index"] == index),
            "index",
        ] = newIndex
        
    if person is not None and index is not None:
        # both
        df.loc[
            (df["frame"] >= startFrame)
            & (df["frame"] <= endFrame)
            & (df["person"] == person)
            & (df["index"] == index),
            "person",
        ] = newPerson
        df.loc[
            (df["frame"] >= startFrame)
            & (df["frame"] <= endFrame)
            & (df["person"] == newPerson)
            & (df["index"] == index),
            "index",
        ] = newIndex
        
    return df


def getkeypointcols():
    """
    Get lists of x and y coordinate column indices.
    
    Returns:
        tuple: (xcols, ycols) - Lists of x and y coordinate column indices
    """
    # set of coord columns, 0,1 are x,y, 2 is confidence, so loop through
    xcols = []
    ycols = []
    for c in range(17 * 3):
        if c % 3 == 0:
            xcols.append(c)
        elif c % 3 == 1:
            ycols.append(c)

    xcols = [x + 8 for x in xcols]  # shift by 8 to get to correct column
    ycols = [y + 8 for y in ycols]  # shift by 8 to get to correct column

    xkeys = [3, 5, *xcols]
    ykeys = [4, 6, *ycols]
    return xkeys, ykeys


def getfacecols():
    """
    Get lists of face x and y coordinate column indices.
    
    Returns:
        tuple: (xcols, ycols) - Lists of face x and y coordinate column indices
    """
    return [3, 5], [4, 6]  # xcols, ycols


def appendDictToDf(df, dict_to_append):
    """
    Append a dictionary to a dataframe.
    
    Args:
        df (DataFrame): DataFrame to append to
        dict_to_append (dict): Dictionary to append
        
    Returns:
        DataFrame: Updated dataframe
    """
    df = pd.concat([df, pd.DataFrame.from_records(dict_to_append)], ignore_index=True)
    return df


def padMovementData(keyPoints, maxFrames=None):
    """
    Pad the keyPoints array so that for each person:
    1. There is a row entry for every frame in the video up to maxFrames
    2a. If the video is less than the maxFrames, we pad out to maxFrames
    2b. If video is longer than maxFrames, we truncate to maxFrames
    
    Args:
        keyPoints (DataFrame): DataFrame containing key points data
        maxFrames (int): Maximum number of frames to pad
        
    Returns:
        DataFrame: Padded DataFrame with consistent frame numbers for each person
    """
    if maxFrames is None:
        maxFrames = keyPoints["frame"].max()
    
    # a list of frame numbers
    frameNumbers = pd.Index(np.arange(0, maxFrames + 1), name="frame")
    
    paddedKeyPoints = keyPoints.iloc[:0].copy()
    
    # There are two people indexed 0 and 1
    # We need to pad both arrays
    for idx in range(2):
        thisperson = keyPoints[keyPoints["index"] == idx]
        missing_frames = frameNumbers.difference(thisperson["frame"])
        
        # pad and fill missing frames
        add_df = pd.DataFrame(index=missing_frames, columns=thisperson.columns).fillna(np.nan)
        add_df["frame"] = missing_frames
        add_df["index"] = idx
        add_df["person"] = idx
        thisperson = pd.concat([thisperson, add_df])
        
        # truncate to maxFrames
        if thisperson.shape[0] > maxFrames:
            thisperson = thisperson[thisperson["frame"] <= maxFrames]
            
        # add the paddedKeyPoints to the dataframe
        paddedKeyPoints = appendDictToDf(paddedKeyPoints, thisperson)
        
    return paddedKeyPoints.sort_values(by=["frame", "index"])


def interpolateMovementData(keyPoints):
    """
    Interpolates movement data to fill missing frames for each person.
    
    Args:
        keyPoints (DataFrame): DataFrame containing key points data
        
    Returns:
        DataFrame: Interpolated DataFrame with filled missing frames
    """
    # a list of frame numbers
    maxFrames = keyPoints["frame"].max()
    frameNumbers = pd.Index(np.arange(0, maxFrames + 1), name="frame")
    
    interpolatedKeyPoints = keyPoints.iloc[:0].copy()
    
    # There are two people indexed 0 and 1
    # We need to interpolate both arrays
    for idx in range(2):
        thisperson = keyPoints[keyPoints["index"] == idx]
        thisperson = thisperson.set_index("frame")
        thisperson = thisperson.reindex(frameNumbers)
        thisperson = thisperson.interpolate(method='linear', axis=0, limit_direction='backward')
        thisperson["frame"] = thisperson.index
        thisperson["index"] = idx
        thisperson["person"] = idx
        
        # add the interpolated data to the dataframe
        interpolatedKeyPoints = appendDictToDf(interpolatedKeyPoints, thisperson)
        
    return interpolatedKeyPoints.sort_values(by=["frame", "index"])


def flattenMovementDataset(keyPoints):
    """
    Flattens the movement dataset by restructuring the key points data.
    
    Args:
        keyPoints (DataFrame): DataFrame containing key points data
        
    Returns:
        DataFrame: Flattened DataFrame with restructured columns
    """
    # There are two people indexed 0 and 1
    flattenedKps = keyPoints.pivot(index='frame', columns='index')
    flattenedKps.columns = ["_".join((str(j), i)) for i, j in flattenedKps.columns]
    flattenedKps = flattenedKps.reset_index()
    return flattenedKps
