# functions to draw annotated frames, videos and time series plots.
# make use of ultralytics.utils where we can.

import ultralytics.utils as ultrautils

def drawOneFrame(baseImage,keyPoints = None,speechLabels = None,objectData = None):
    '''
    redraw one frame with all the annotations we provide. 
    Use ultralytics.utils.Annotator where we can.

    Args: keyPoints, speechLabels, objectData
    Output: annotated image
    '''
    annotator = ultrautils.plotting.Annotator(baseImage)
    for row in keyPoints:
        if keyPoints is not None:
            annotator.kpts(keyPoints)
    return annotator.result()