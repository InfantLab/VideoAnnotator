# functions to draw annotated frames, videos and time series plots.
# make use of ultralytics.utils where we can.

import ultralytics.utils as ultrautils

def drawOneFrame(baseImage, bboxes = None, keyPoints = None,speechLabels = None,objectData = None):
    '''
    redraw one frame with all the annotations we provide. 
    Use ultralytics.utils.Annotator where we can.

    Args:   bboxes - expects one row per person, each row to contain [person,idx,x,y,w,h] 
            keyPoints - [nrows x 51] 
            speechLabel - string of speech happening during this frame
            objectData - similar to bboxes, but for objects [objecttype,objectinfo,x,y,w,h]
    Output: annotated image
    '''
    annotator = ultrautils.plotting.Annotator(baseImage)
    for box in bboxes:
        xyxy = ultrautils.ops.xywh2xyxy(box[2:]) #need to convert the bounding box to xyxy format
        annotator.box_label(box = xyxy, label = f"{box[0]}: {box[1]}")
    for kpts in keyPoints:
        kpts = kpts.reshape(17,3)
        annotator.kpts(kpts)
    return annotator.result()