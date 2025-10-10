"""
Functions for video creation and manipulation.
"""

import os
import time

import cv2
from IPython.display import clear_output
from IPython.display import display as ipydisplay
from PIL import Image as PILImage

from src.visualization.render import WhisperExtractCurrentCaption, drawOneFrame


def createAnnotatedVideo(
    videopath, kptsdf=None, facesdf=None, speechjson=None, videos_out=None, debug=False
):
    """
    Take a processed video and go through frame by frame, adding the bounding boxes,
    keypoints, face/emotion and speech info. Then export the resulting video to a file.

    Args:
        videopath (str): Path to the video file
        kptsdf (DataFrame): Dataframe of the keypoints
        facesdf (DataFrame): Dataframe of the faces
        speechjson (dict): Speech data from Whisper
        videos_out (str): Path to the output directory
        debug (bool): Whether to display debug output

    Returns:
        str: Path to the output video
    """
    # check if video exists
    if not os.path.exists(videopath):
        print(f"Video file {videopath} not found.")
        return None

    video = cv2.VideoCapture(videopath)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # loop through frames annotating each one and storing to a list
    annotatedframes = []
    framenum = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("video read stopped on frame ", framenum)
            break

        # get the keypoints for this frame
        if kptsdf is None:
            bboxes = None
            xycs = None
        else:
            framekpts = kptsdf[kptsdf["frame"] == framenum]
            nrows = framekpts.shape[0]
            bboxlabels = [None] * nrows
            # for each row framekpts, create a label for the bounding box from person and index cols
            for idx in range(nrows):
                pers = framekpts["person"].values[idx]
                index = framekpts["index"].values[idx]
                bboxlabels[idx] = f"{pers}: {index}"
                bboxes = framekpts.iloc[:, 3:7].values
                xycs = framekpts.iloc[:, 8:].values
            frame = drawOneFrame(frame, bboxlabels, bboxes, xycs)

        if facesdf is None:
            framefaces = None
        else:
            # get the faces for this frame
            framefaces = facesdf[facesdf["frame"] == framenum]
            facelabels = framefaces["emotion"].values
            # TODO - maybe include age & gender info
            faceboxes = framefaces.iloc[:, 3:7].values
            frame = drawOneFrame(frame, facelabels, faceboxes)

        if speechjson is None:
            pass
        else:
            caption = WhisperExtractCurrentCaption(speechjson, framenum, fps)
            frame = drawOneFrame(
                frame, bboxlabels=None, bboxes=None, keyPoints=None, speechLabel=caption
            )

        # add the frame to the list
        annotatedframes.append(frame)
        framenum += 1

    # release the video
    video.release()

    # create the output video
    if videos_out is None:
        videos_out = os.path.dirname(videopath)
    videoname = os.path.basename(videopath)
    videofilename = os.path.splitext(videoname)[0] + ".annotated.mp4"
    outpath = os.path.join(videos_out, videofilename)
    out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
    print(f"Writing video to {outpath}")

    for i in range(len(annotatedframes)):
        if debug:
            clear_output(wait=True)
            color_converted = cv2.cvtColor(annotatedframes[i], cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(color_converted)
            ipydisplay(pil_image)
            time.sleep(1 / fps)
        out.write(annotatedframes[i])
    out.release()
    print(f"Number of frames: {len(annotatedframes)}")
    return outpath


def addSoundtoVideo(videopath, soundpath, out_dir=None):
    """
    Take a video and add a sound file to it using moviepy.

    Args:
        videopath (str): Path to the video file
        soundpath (str): Path to the sound file
        out_dir (str): Path to the output directory

    Returns:
        str: Path to the output video
    """
    import os

    from moviepy.editor import AudioFileClip, VideoFileClip

    try:
        videoclip = VideoFileClip(videopath)
        audioclip = AudioFileClip(soundpath)
        videoclip = videoclip.set_audio(audioclip)
        if out_dir is None:
            out_dir = os.path.dirname(videopath)
        outpath = os.path.join(out_dir, os.path.basename(videopath))
        videoclip.write_videofile(outpath, codec="libx264", audio_codec="aac")
        return outpath
    except Exception as e:
        print(f"Error occurred while adding sound to video: {e}")
        return None
