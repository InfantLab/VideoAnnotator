# trunk-ignore-all(black)
import json
import os

import moviepy.editor as mp
import pandas as pd
import torch
import ultralytics.utils as ultrautils


from os.path import normpath
from pathlib import PureWindowsPath

def posixpath(path):
    return PureWindowsPath(normpath(PureWindowsPath(path).as_posix())).as_posix()

def localpath(path):
    return os.path.normpath(path)

# helper functions for processing videos and dataframes

def getprocessedvideos(data_dir, filename="processedvideos.xlsx"):
    # looks in data_dir for processedvideos.xlsx, if it exists, loads it, otherwise creates it.
    filepath = os.path.join(data_dir, filename)
    # check if we have already processed some videos
    if os.path.exists(filepath):
        print(f"Found existing {filename}")
        processedvideos = pd.read_excel(filepath, index_col=None)
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
            "LastError",
            "annotatedVideo",
            "annotated.when",
        ]
        processedvideos = pd.DataFrame(columns=cols)
        processedvideos.to_excel(filepath, index=False)
    return processedvideos


def saveprocessedvideos(processedvideos, data_dir, filename="processedvideos.xlsx"):
    filepath = os.path.join(data_dir, filename)
    processedvideos.to_excel(filepath, index=False)


def createkeypointsdf():
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
    df = pd.DataFrame(columns=cols)
    return df


def addkeypointstodf(df, framenumber, bbox, bconf, keypointsdata):
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
    if normed:
        kptsfile = videodata["Keypoints.normed"].values[0]
    else:
        kptsfile = videodata["Keypoints.file"].values[0]
    keypoints = pd.read_csv(kptsfile)
    return keypoints


def getframekpts(kptsdf, framenumber):
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
    df = createkeypointsdf()
    frame = 0
    for r in results:
        # print(torch.flatten(r.keypoints.xy[0]).tolist())
        df = addkeypointstodf(df, frame, r.boxes.xywh, r.boxes.conf, r.keypoints.data)
        frame += 1
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
    # note that unlike YOLO bounding boxes are returned top left corner and width/height not centre and width/height
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

    xkeys = [3, 5]  # bounding box
    xkeys.extend(xcols)  # keypoints
    ykeys = [4, 6]  # bounding box
    ykeys.extend(ycols)  # keypoints
    return xkeys, ykeys


def getfacecols():
    #    return [3,4,5,6]
    return [3, 5], [4, 6]  # xcols,ycols


def getKeyPoints(processedvideos, videoname):
    # look in processed videos to see if we have a keypoints file for this video
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] > 0:
        print(f"We have a keypoints file for {videoname}")
        kptsfile = videodata["Keypoints.file"].values[0]
        # Load the keypoints file
        kpts = pd.read_csv(kptsfile)
    else:
        raise FileNotFoundError(f"No keypoints file found for {videoname}")
    return kpts


def getFaceData(processedvideos, videoname):
    # look in processed videos to see if we have a keypoints file for this video
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] > 0:
        print(f"We have a face data file for {videoname}")
        facesfile = videodata["Faces.file"].values[0]
        # Load the keypoints file
        facedata = pd.read_csv(facesfile)
    else:
        raise FileNotFoundError(f"No face data file found for {videoname}")
    return facedata


def getSpeechData(processedvideos, videoname):
    # look in processed videos to see if we have a keypoints file for this video
    videodata = processedvideos[processedvideos["VideoID"] == videoname]
    if videodata.shape[0] > 0:
        print(f"We have a speech data file for {videoname}")
        speechfile = videodata["Speech.file"].values[0]
        # Load the keypoints file
        with open(speechfile) as f:
            speechdata = json.load(f)
    else:
        raise FileNotFoundError(f"No speech data file found for {videoname}")
    return speechdata
