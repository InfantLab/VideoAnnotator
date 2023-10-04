# functions to draw annotated frames, videos and time series plots.
# make use of ultralytics.utils where we can.

import ultralytics.utils as ultrautils

def drawOneFrame(baseImage, bboxlabels = None, bboxes = None, keyPoints = None,speechLabels = None,objectData = None):
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
    return annotator.result()