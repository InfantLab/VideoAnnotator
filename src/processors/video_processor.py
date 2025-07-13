"""
Functions for processing video data.
"""

import cv2
from src.utils.dataframe_utils import createKeypointsDF, addKeypointsToDF


def get_video_metadata(video_path):
    """
    Extract metadata from a video file.

    Args:
        video_path (str): Path to the video file

    Returns:
        dict: Video metadata including dimensions and length
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return {"Width": width, "Height": height, "FPS": fps, "FrameCount": frame_count}


def videotokeypoints(model, videopath, track=True):
    """
    Run keypoint detection on a video.

    Args:
        model: YOLO model to use
        videopath (str): Path to the video
        track (bool): Whether to track objects

    Returns:
        DataFrame: DataFrame of keypoints
    """
    # Run inference on the source
    if track:
        results = model.track(videopath, stream=True)
    else:
        results = model(videopath, stream=True)  # generator of Results objects

    df = createKeypointsDF()
    for frame, r in enumerate(results):
        df = addKeypointsToDF(df, frame, r.boxes.xywh, r.boxes.conf, r.keypoints.data)
    return df


def getFrameKpts(kptsdf, framenumber):
    """
    Get keypoints for a specific frame.

    Args:
        kptsdf (DataFrame): DataFrame of keypoints
        framenumber (int): Frame number

    Returns:
        tuple: (bboxlabels, bboxes, xycs)
    """
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
