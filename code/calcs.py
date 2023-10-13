# functions to help with calculations. 
# Note: where possible we use ultralytics.utils functions for keypoint
# and bounding box calculations, but we also have some of our own.

# several adapted from vasc.py in github.com/infantlab/vasc

import ultralytics.utils as ultrautils
import numpy as np

def centreOfGravity(df,frames = [],people = "all",bodypart = "whole"):
    '''find average position of a bodypart across frames and people, 
    and add these as new column in the dataframe
    useful for plotting  time series of movement.

    args:   df - dataframe of keypoints
            frames - list of frames to include
            people - list of people to include
            bodypart - which bodypart to use, default is "whole" for all keypoints
    returns:
            dataframe of average positions 
    '''

    if len(frames) == 0:
        frames = df.frame.unique()
    
    if people == "all":
        people = df.person.unique()

    if bodypart != "whole":
        raise NotImplementedError("Only whole body implemented for now")
    
    threshold = 0.5

    #create new columns for the centre of gravity
    df["cog.x"] = np.nan
    df["cog.y"] = np.nan                

    for frame in frames:
        for person in people:
            #get the keypoints for this person in this frame
            kpts = df[(df['frame'] == frame) & (df['person'] == person)]
            
            if kpts.shape[0] > 0:
                #get the average position of the bodypart
                #print(kpts)
                if bodypart == "whole":
                    xyc = kpts[:,8:]
                    avgx, avgy = avgxys(xyc,threshold)
                
                df.loc[(df['frame'] == frame) & (df['person'] == person), "cog.x"] = avgx
                df.loc[(df['frame'] == frame) & (df['person'] == person), "cog.y"] = avgy   

    return df

def avgxys(xyc,threshold = 0.5):
    '''
    Given a set of x,y,conf values (a n x 3 array) calculate the average x,y values
    for all those with a confidence above the threshold.
    args:   xyc - [nrows x 3] array of x,y,conf values
            threshold - confidence threshold
    returns:    avgx, avgy
    '''
    #get the x,y values where conf > threshold
    x = xyc[:,0]
    y = xyc[:,1]
    conf = xyc[:,2]
    x = x[conf > threshold]
    y = y[conf > threshold]
    #calculate the average
    avgx = np.mean(x)
    avgy = np.mean(y)
    return avgx, avgy



