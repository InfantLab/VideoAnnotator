# functions to draw annotated frames, videos and time series plots.
# make use of ultralytics.utils where we can.

import ultralytics.utils as ultrautils
import os
import cv2

def drawOneFrame(baseImage, bboxlabels = None, bboxes = None, keyPoints = None,speechLabel = None,objectData = None):
    '''
    redraw one frame with all the annotations we provide. 
    Use ultralytics.utils.Annotator where we can.

    Args:   bboxlabels - list of labels for each bounding box, must be same length as bboxes
            bboxes - expects one row per person, each row to contain [x1,y1,x2,y2] 
            keyPoints - [nrows x 51] 
            speechLabel - string of speech happening during this frame
            objectData - similar to bboxes, but for objects [objecttype,objectinfo,x,y,w,h]
    Output: annotated image
    '''
    annotator = ultrautils.plotting.Annotator(baseImage)
    if bboxlabels is not None and bboxes is not None:
        for idx, box in enumerate(bboxes):
            annotator.box_label(box = box, label = bboxlabels[idx])
    if keyPoints is not None:
        for kpts in keyPoints:
            kpts = kpts.reshape(17,3)
            annotator.kpts(kpts)
    if speechLabel is not None:
        h, w = baseImage.shape[:2]
        #annotator quite bad when using cv2
        annotator.text([int(w/3),int(h/10)],speechLabel, anchor = 'top')
    return annotator.result()

def createAnnotatedVideo(videopath,kptsdf = None,facesdf = None,speechjson = None, videos_out = None):
    '''
    Take a processed video and go through frame by frame, adding the bounding boxes, keypoints, face/emotion and speech info.
    Then export the resulting video to a file.
    args:   
        videopath: path to the video file
        kptsdf: dataframe of the keypoints
        facesdf: dataframe of the faces
        speechdf: dataframe of the speech
        videos_out: path to the output directory
    returns:
        path to the output video
    '''
    video = cv2.VideoCapture(videopath)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    #loop through frames annotating each one and storing to a list
    annotatedframes = []
    framenum = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        #get the keypoints for this frame
        if kptsdf is None:
            bboxes = None
            xycs = None
        else:
            framekpts = kptsdf[kptsdf['frame'] == framenum]
            nrows = framekpts.shape[0]
            bboxlabels = [None] * nrows
            #for each row framekpts, create a label for the bounding box from person and index cols
            for idx in range(nrows):
                pers = framekpts["person"].values[idx]
                index = framekpts["index"].values[idx]
                bboxlabels[idx] =  f'{pers}: {index}'
                
            bboxes = framekpts.iloc[:,3:7].values

            xycs = framekpts.iloc[:,8:].values
            frame = drawOneFrame(frame, bboxlabels, bboxes, xycs)
        if facesdf is None:
            framefaces = None
        else:
            #get the faces for this frame
            framefaces = facesdf[facesdf['frame'] == framenum]
            facelabels = framefaces['emotion'].values
            #TODO - maybe include age & gender info
            faceboxes = framefaces.iloc[:,3:7].values
            frame = drawOneFrame(frame, facelabels, faceboxes)
        if speechjson is None:
            pass
        else:
            caption = WhisperExtractCurrentCaption(speechjson, framenum,fps)
            frame = drawOneFrame(frame, bboxlabels = None, bboxes = None, keyPoints = None,speechLabel = caption)
        
        #add the frame to the list
        annotatedframes.append(frame)
        framenum += 1

    #release the video
    video.release()

    #create the output video
    if videos_out is None:
        videos_out = os.path.dirname(videopath)
    videoname = os.path.basename(videopath)
    videofilename = os.path.splitext(videoname)[0] + "_annotated.mp4"
    outpath = os.path.join(videos_out, videofilename)
    out  = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
    print(f"Writing video to {outpath}")

    for i in range(len(annotatedframes)):
        out.write(annotatedframes[i])
    out.release()

    return outpath

def WhisperExtractCurrentCaption(speechjson,frame,fps):
    '''looks through 'segments' data in the json output from whisper
    and returns the caption that is current for the given frame'''
    time = frame/fps
    for seg in speechjson['segments']:
        if time >= seg['start'] and time <= seg['end']:
            return seg['text']
    return ""