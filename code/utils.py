# trunk-ignore-all(black)
import json
import os

import moviepy.editor as mp
import pandas as pd
import numpy as np
import torch
import ultralytics.utils as ultrautils

from os.path import normpath
from pathlib import PureWindowsPath

from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
import torch, torchaudio

# take secrets from environment variables
from dotenv import load_dotenv
load_dotenv()  
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

def posixpath(path):
    return PureWindowsPath(normpath(PureWindowsPath(path).as_posix())).as_posix()

def localpath(path):
    return os.path.normpath(path)

# helper functions for processing videos and dataframes

def getVideoProperty(processedvideos, VideoID, Property):
    # get the value of a property for a video in the processedvideos dataframe
    videodata = processedvideos[processedvideos["VideoID"] == VideoID]
    if videodata.shape[0] > 0:
        return videodata[Property].values[0]
    else:
        return None


def getProcessedVideos(data_dir, filename="processedvideos.xlsx"):
    # looks in data_dir for processedvideos.xlsx, if it exists, loads it, otherwise creates it.
    filepath = os.path.join(data_dir, filename)
    # check if we have already processed some videos
    if os.path.exists(filepath):
        processedvideos = pd.read_excel(filepath, index_col=None)
        print(f"Found existing {filename} with {processedvideos.shape[0]} rows.")
    else:
        # create new dataframe for info about processed videos
        print(f"Creating new {filename}")
        cols = [
            "VideoID",
            "ChildID",
            "JokeType",
            "JokeNum",
            "JokeRep",
            "JokeTake",
            "HowFunny",
            "LaughYesNo",
            "Frames",
            "FPS",
            "Width",
            "Height",
            "Duration",
            "Keypoints.when",
            "Keypoints.file",
            "Audio.when",
            "Audio.file",
            "Faces.when",
            "Faces.file",
            "Speech.when",
            "Speech.file",
            "Diary.file",
            "Diary.when",
            "LastError",
            "annotatedVideo",
            "annotated.when",
        ]
        processedvideos = pd.DataFrame(columns=cols)
        ## create numerical labels for joke type
        #labels, unique = pd.factorize(df['category_column'])
        #processedvideos["JokeType.label"] = labels
        processedvideos.to_excel(filepath, index=False)
    return processedvideos


def saveProcessedVideos(processedvideos, data_dir, filename="processedvideos.xlsx"):
    filepath = os.path.join(data_dir, filename)
    processedvideos.to_excel(filepath, index=False)


def createKeypointsDF():
    # create empty dataframe to store keypoints, one per person per frame
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
    # take output from yolov8 and add to dataframe, person by person.
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
        # print(row)
        df.loc[len(df)] = row
    return df


def readKeyPointsFromCSV(processedvideos, VIDEO_FILE, normed=False):
    # get the keypoints from the csv file
    videoname = os.path.basename(VIDEO_FILE)
    # is video in the processedvideos dataframe?
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] <= 0:
        raise FileNotFoundError(f"No keypoints file found for {videoname}")
    if normed:
        kptsfile = videodata["Keypoints.normed"].values[0]
    else:
        kptsfile = videodata["Keypoints.file"].values[0]
    return pd.read_csv(kptsfile)


def getFrameKpts(kptsdf, framenumber):
    framekpts = kptsdf[kptsdf["frame"] == framenumber]
    nrows = framekpts.shape[0]
    bboxlabels = [None] * nrows
    # for each row framekpts, create a label for the bounding box from person and index cols
    for idx in range(nrows):
        pers = framekpts["person"].values[idx]
        index = framekpts["index"].values[idx]
        bboxlabels[idx] = f"{pers}: {index}"

    bboxes = framekpts.iloc[:, 3:7].values
    xycs = framekpts.iloc[:, 8:].values

    return bboxlabels, bboxes, xycs


def videotokeypoints(model, videopath, track=False):
    # Run inference on the source
    if track:
        results = model.track(videopath, stream=True)
    else:
        results = model(videopath, stream=True)  # generator of Results objects
    df =createKeypointsDF()
    for frame, r in enumerate(results):
        # print(torch.flatten(r.keypoints.xy[0]).tolist())
        df = addKeypointsToDF(df, frame, r.boxes.xywh, r.boxes.conf, r.keypoints.data)
    return df


def convert_video_to_audio_moviepy(videos_in, video_file, out_path, output_ext="mp3"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    try:
        filename = os.path.splitext(video_file)[0]
        video = os.path.join(videos_in, video_file)
        clip = mp.VideoFileClip(video)
        audio_file = os.path.join(out_path, f"{filename}.{output_ext}")
        clip.audio.write_audiofile(audio_file)
        clip.close()
        return audio_file
    except Exception as e:
        print(f"Error converting {video_file} to {output_ext}: {e}")
        return None


def convert_mp3_to_wav_moviepy(audio_file, output_ext="wav"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(audio_file)
    clip = mp.AudioFileClip(audio_file)
    clip.write_audiofile(f"{filename}.{output_ext}")


def createfacesdf():
    # creates a dataframe with the facial data from the videos
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
    ]  # , "allemotions","allgenders"]
    return pd.DataFrame(columns=cols)


def addfacestodf(facesdf, frameidx, facedata):
    # add the faces identified by face detection model to the dataframe, along with emotion, age and gender.
    # note that unlike YOLO bounding boxes, these are returned top left corner and width/height not centre and width/height
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
    """replace person and/or index values with new values for a range of frames"""
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
    # set of coord columns, 0,1 are x,y, 2 is confidence, so loop through
    xcols = []
    ycols = []
    for c in range(17 * 3):
        if c % 3 == 0:
            xcols.append(c)
        elif c % 3 == 1:
            ycols.append(c)

    xcols = [x + 8 for x in xcols]  # shift by 8 to get to correct column
    ycols = [x + 8 for x in ycols]  # shift by 8 to get to correct column

    xkeys = [3, 5, *xcols]
    ykeys = [4, 6, *ycols]
    return xkeys, ykeys


def getfacecols():
    #    return [3,4,5,6]
    return [3, 5], [4, 6]  # xcols,ycols


def getKeyPoints(processedvideos, videoname):
    # look in processed videos to see if we have a keypoints file for this video
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] <= 0:
        raise FileNotFoundError(f"No keypoints file found for {videoname}")
    print(f"We have a keypoints file for {videoname}")
    kptsfile = videodata["Keypoints.file"].values[0]
    return pd.read_csv(kptsfile)


def getFaceData(processedvideos, videoname):
    # look in processed videos to see if we have a keypoints file for this video
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] <= 0:
        raise FileNotFoundError(f"No face data file found for {videoname}")
    print(f"We have a face data file for {videoname}")
    facesfile = videodata["Faces.file"].values[0]
    return pd.read_csv(facesfile)


def getSpeechData(processedvideos, videoname):
    """
    Retrieves speech data for a specific video from processed videos.

    Args:
        processedvideos (pandas.DataFrame): DataFrame containing processed video data.
        videoname (str): Name of the video to retrieve speech data for.

    Returns:
        dict: Speech data for the specified video.
    Raises:
        FileNotFoundError: If no speech data file is found for the provided video name.
    """
    # look in processed videos to see if we have a keypoints file for this video
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] <= 0:
        raise FileNotFoundError(f"No speech data file found for {videoname}")
    print(f"We have a speech data file for {videoname}")
    speechfile = videodata["Speech.file"].values[0]
    # Load the keypoints file
    with open(speechfile) as f:
        speechdata = json.load(f)
    return speechdata



############################################################################################################
### Dataframe manipulation functions
############################################################################################################
def appendDictToDf(df, dict_to_append):
    df = pd.concat([df, pd.DataFrame.from_records(dict_to_append)],ignore_index=True)
    return df

def padMovementData(keyPoints, maxFrames = None):
    """
    We pad the keyPoints array so that for each person [0,1]:
    1. There is a row entry for every frame in the video upto maxFrames
    2a. if the video is less than the maxFrames, we pad out to maxFrames
    2b. if vides is longer than maxFrames, we truncate to maxFrames
    Nan values are used to pad the array.

    Args:
        keyPoints (pandas.DataFrame): DataFrame containing key points data.
        maxFrames (int, optional): Maximum number of frames to pad. Defaults to None.

    Returns:
        pandas.DataFrame: Padded DataFrame with consistent frame numbers for each person.
    """
    if maxFrames is None:
        maxFrames = keyPoints.shape[1]
    
    # a list of frame numbers
    frameNumbers = pd.Index(np.arange(0,maxFrames + 1), name="frame")
    
    paddedKeyPoints = keyPoints.iloc[:0].copy()
    
    #There are two people indexed 0 and 1. 
    #We need to pad both arrays
    for idx in range(2):
        thisperson = keyPoints[keyPoints["index"]==idx]
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
        paddedKeyPoints = appendDictToDf(paddedKeyPoints,thisperson)
        
    return paddedKeyPoints.sort_values(by=["frame","index"])

def interpolateMovementData(keyPoints):
    """
    Interpolates movement data to fill missing frames for each person.
        We interpolate the keyPoints array so that for each person [0,1]:
        1. There is a row entry for each frame in the video 
        2. if the video is less than the maxFrames, we pad out to maxFrames
        Nan values are used to pad the array.

    Args:
        keyPoints (pandas.DataFrame): DataFrame containing key points data.

    Returns:
        pandas.DataFrame: Interpolated DataFrame with filled missing frames.
    """
    # a list of frame numbers
    maxFrames = keyPoints["frame"].max()
    frameNumbers = pd.Index(np.arange(0,maxFrames + 1), name="frame")
    
    interpolatedKeyPoints = keyPoints.iloc[:0].copy()
    
    #There are two people indexed 0 and 1. 
    #We need to interpolate both arrays
    for idx in range(2):
        thisperson = keyPoints[keyPoints["index"]==idx]
        thisperson = thisperson.set_index("frame")
        thisperson = thisperson.reindex(frameNumbers)
        thisperson = thisperson.interpolate(method='linear',axis=0,limit_direction='backward')
        thisperson["frame"] = thisperson.index
        thisperson["index"] = idx
        thisperson["person"] = idx
        # add the paddedKeyPoints to the dataframe
        interpolatedKeyPoints = appendDictToDf(interpolatedKeyPoints,thisperson)
        
    return interpolatedKeyPoints.sort_values(by=["frame","index"])

def flattenMovementDataset(keyPoints):
    """
    Flattens the movement dataset by restructuring the key points data.
    We go from 1 row per frame per person to 1 row per frame with columns for each key point.

    Args:
    keyPoints (pandas.DataFrame): DataFrame containing key points data.

    Returns:
    pandas.DataFrame: Flattened DataFrame with restructured columns.
    """
    #There are two people indexed 0 and 1. 
    flattenedKps = keyPoints.pivot(index='frame', columns='index')
    flattenedKps.columns = ["_".join((str(j),i)) for i,j in flattenedKps.columns]
    flattenedKps = flattenedKps.reset_index()
    return flattenedKps


def diarize_audio(audio_file):
    """Diarize an audio file using the pyannote speaker diarization model.
    from https://github.com/pyannote/pyannote-audio

    Args:
        audio_file (wav): expects a wav file

    Returns:
        Diarization: in json format {{start: float, end: float, speaker: str},...}
    """
    try:
        ## Convert audio to WAV if necessary
        #audio_file = convert_audio(audio_file, "wav")

        print(f"Diarizing audio file: {audio_file}")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_ACCESS_TOKEN)

        # send pipeline to GPU (when available)
        pipeline.to(torch.device("cuda"))

        # apply pretrained pipeline
        diarization = pipeline(audio_file)
        
        return diarization
        
    except Exception as e:
        print(f"diarize_audio Error: {str(e)}")
        return None