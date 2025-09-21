"""
Functions for rendering annotated frames and other visualizations.
"""

import ultralytics.utils as ultrautils


def drawOneFrame(
    baseImage,
    bboxlabels=None,
    bboxes=None,
    keyPoints=None,
    speechLabel=None,
    objectData=None,
):
    """
    Redraw one frame with all the annotations we provide.
    Use ultralytics.utils.Annotator where we can.

    Args:
        baseImage (np.ndarray): Base image to annotate
        bboxlabels (list): List of labels for each bounding box
        bboxes (list): List of bounding boxes, each containing [x1,y1,x2,y2]
        keyPoints (np.ndarray): Keypoints array [nrows x 51]
        speechLabel (str): String of speech happening during this frame
        objectData (list): Object detection data [objecttype,objectinfo,x,y,w,h]

    Returns:
        np.ndarray: Annotated image
    """
    annotator = ultrautils.plotting.Annotator(baseImage)

    if bboxlabels is not None and bboxes is not None:
        for idx, box in enumerate(bboxes):
            annotator.box_label(box=box, label=bboxlabels[idx])

    if keyPoints is not None:
        for kpts in keyPoints:
            kpts = kpts.reshape(17, 3)
            annotator.kpts(kpts)

    if speechLabel is not None:
        h, w = baseImage.shape[:2]
        annotator.text([int(w / 3), int(h / 10)], speechLabel, anchor="top")

    return annotator.result()


def WhisperExtractCurrentCaption(speechjson, frame, fps):
    """
    Looks through 'segments' data in the json output from whisper
    and returns the caption that is current for the given frame.

    Args:
        speechjson (dict): Speech data from Whisper
        frame (int): Frame number
        fps (float): Frames per second

    Returns:
        str: Caption for the current frame
    """
    time = frame / fps
    for seg in speechjson["segments"]:
        if time >= seg["start"] and time <= seg["end"]:
            return seg["text"]
    return ""
