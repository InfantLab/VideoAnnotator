#helper functions
import os
import torch
import pandas as pd
import moviepy.editor as mp

def getprocessedvideos(data_out):
    #check if we have already processed some videos
    if os.path.exists(data_out + "\\processedvideos.xlsx"):
        print("found existing processedvideos.xlsx")
        processedvideos = pd.read_excel(data_out + "\\processedvideos.xlsx")
    else:
        #create new dataframe for info about processed videos
        print("creating new processedvideos.xlsx")
        cols = ["VideoID","ChildID", "JokeType","JokeNum","JokeRep","JokeTake", "HowFunny","LaughYesNo", "Frames", "FPS", "Width", "Height", "Duration","Keypoints.when", "Keypoints.file","Audio.when","Audio.file","Faces.when","Faces.file","Speech.when","Speech.file","LastError"]
        processedvideos = pd.DataFrame(columns=cols)
    return processedvideos

def createkeypointsdf():
    #create empty dataframe to store keypoints, one per person per frame
    bodyparts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    coords = ['x', 'y']
    conf = ['c']
    bodypartsxy = [f"{bp}.{c}" for bp in bodyparts for c in coords]
    bodypartsc = [f"{bp}.{c}" for bp in bodyparts for c in conf]
    boundingbox = [ 'bboxcent.x', 'bboxcent.y', 'bbox.width', 'bbox.height', 'bbox.c' ]
    cols = ['frame', 'person'] + boundingbox + bodypartsxy + bodypartsc
    df = pd.DataFrame(columns=cols)
    return df

def addkeypointstodf(df, framenumber, bbox,bconf, keypoints, kconf):
    for idx in range(len(bbox)):
        row = [framenumber, idx]
        row += torch.flatten(bbox[idx]).tolist()
        row += torch.flatten(bconf[idx]).tolist()
        row += torch.flatten(keypoints[idx]).tolist()
        row += torch.flatten(kconf[idx]).tolist()
        #add row to dataframe
        #print(row)
        df.loc[len(df)] = row
    return df

def videotokeypoints(model, videopath, track = False):
    # Run inference on the source
    if track:
        results = model.track(videopath,stream=True)  
    else:
        results = model(videopath, stream=True)  # generator of Results objects    
    df = createkeypointsdf()
    frame = 0
    for r in results:
        #print(torch.flatten(r.keypoints.xy[0]).tolist())
        df = addkeypointstodf(df,frame,r.boxes.xywh,r.boxes.conf,r.keypoints.xy, r.keypoints.conf)  
        frame += 1
    return df

def convert_video_to_audio_moviepy(videos_in, video_file, out_path, output_ext="mp3"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    try:
        filename = os.path.splitext(video_file)[0]
        clip = mp.VideoFileClip(videos_in + video_file)
        audio_file = f"{out_path}\\{filename}.{output_ext}"
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
