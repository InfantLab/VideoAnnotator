# Data Formats


This document explains the data formats created and saved by the project. 

This data is found in the folder `/data/1_interim`

1. Metadata dataframe `processedvideos` saved as `processedvideos.xlsx`
2. Keypoints from YOLOv8. One datafram per video each row is a unique person detected in a single frame of video
    a. Keypoints DataFrames - keypoints x,y locations in frame pixels `{videoid}.csv`, 
    b. Normed Keypoints DataFrames - keypoints scaled to range (0, 1) `{videoid}_normed.csv`
3. Face recognition data from DeepFace - potentially several dataframes per video (One for each backend model used).
    Each dataframe has a row for each face detected in a frame of video.
    a. Face DataFrames - face embeddings `{videoid}.faces.{facemodel}.csv`
    b. Normed faces - face embeddings scaled to range (0, 1) `{videoid}.faces.{facemodel}_normed.csv`
4. Speech recognition from OpenAI Whisper model. Saved a json file.

## 1. Metadata  - `processedvideos` DataFrame

The `processedvideos` DataFrame is used to store metadata about processed videos. It includes various columns to store information about different aspects of the videos, such as keypoints, audio, faces, speech, and diary entries.

### Columns

The `processedvideos` DataFrame has the following columns:

- `Duration`: The duration of the video.
- `Keypoints.when`: Timestamp for when keypoints were detected.
- `Keypoints.file`: File path to the keypoints data.
- `Audio.when`: Timestamp for when audio was processed.
- `Audio.file`: File path to the audio data.
- `Faces.when`: Timestamp for when faces were detected.
- `Faces.file`: File path to the faces data.
- `Speech.when`: Timestamp for when speech was detected.
- `Speech.file`: File path to the speech data.
- `Diary.file`: File path to the diary data.
- `Diary.when`: Timestamp for when the diary entry was made.
- `LastError`: Information about the last error encountered during processing.
- `annotatedVideo`: File path to the annotated video.
- `annotated.when`: Timestamp for when the video was annotated.

### Saving the `processedvideos` DataFrame

The `processedvideos` DataFrame can be saved to an Excel file using the `saveProcessedVideos` function:

```python
def saveProcessedVideos(processedvideos, data_dir, filename="processedvideos.xlsx"):
    filepath = os.path.join(data_dir, filename)
    processedvideos.to_excel(filepath, index=False)
```

## 2. Keypoints DataFrames

The Keypoints DataFrame is used to store keypoints data for each person detected in each frame of the video. It includes columns for various body parts and their coordinates, as well as bounding box information.


The Keypoints DataFrame has the following columns:
### Columns
- `frame`: The frame number in the video.
- `person`: The person identifier.
- `bbox.x1`: The x-coordinate of the top-left corner of the bounding box.
- `bbox.y1`: The y-coordinate of the top-left corner of the bounding box.
- `bbox.x2`: The x-coordinate of the bottom-right corner of the bounding box.
- `bbox.y2`: The y-coordinate of the bottom-right corner of the bounding box.
- `bbox.c`: The confidence score of the bounding box.
- body keypoints have an x, y, and confidence score for each body part.
```
nose.x, nose.y, nose.c
left_eye.x, left_eye.y, left_eye.c
right_eye.x, right_eye.y, right_eye.c
left_ear.x, left_ear.y, left_ear.c
right_ear.x, right_ear.y, right_ear.c
left_shoulder.x, left_shoulder.y, left_shoulder.c
right_shoulder.x, right_shoulder.y, right_shoulder.c
left_elbow.x, left_elbow.y, left_elbow.c
right_elbow.x, right_elbow.y, right_elbow.c
left_wrist.x, left_wrist.y, left_wrist.c
right_wrist.x, right_wrist.y, right_wrist.c
left_hip.x, left_hip.y, left_hip.c
right_hip.x, right_hip.y, right_hip.c
left_knee.x, left_knee.y, left_knee.c
right_knee.x, right_knee.y, right_knee.c
left_ankle.x, left_ankle.y, left_ankle.c
right_ankle.x, right_ankle.y, right_ankle.c
```

## 3. Face Recognition Data Format

The face recognition data format is used to store information about faces detected in each frame of the video. This includes details such as the bounding box coordinates, emotion, age, and gender of each detected face.

### Columns

The face recognition DataFrame has the following columns:

- `frame`: The frame number in the video.
- `person`: The person identifier (default is "unknown").
- `index`: The index of the face within the frame.
- `bbox.x1`: The x-coordinate of the top-left corner of the bounding box.
- `bbox.y1`: The y-coordinate of the top-left corner of the bounding box.
- `bbox.x2`: The x-coordinate of the bottom-right corner of the bounding box.
- `bbox.y2`: The y-coordinate of the bottom-right corner of the bounding box.
- `emotion`: The dominant emotion detected for the face.
- `age`: The estimated age of the person.
- `gender`: The dominant gender detected for the face.

### Example Function: `addfacestodf`

The `addfacestodf` function adds the faces identified by the face detection model to the DataFrame, along with emotion, age, and gender information. Note that unlike YOLO bounding boxes, these are returned with the top-left corner and width/height, not the center and width/height.

```python
def addfacestodf(facesdf, frameidx, facedata):
    # add the faces identified by face detection model to the dataframe, along with emotion, age and gender.
    # note that unlike YOLO bounding boxes, these are returned top left corner and width/height not centre and width/height
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
```

## 4. Speech Recognition Data Format

The output from whisper is saved directly as a json file from the following fucntion call:

```
    result = model.transcribe(audio_file, verbose = True)
```

# Speech Recognition JSON Format

The JSON file contains the following structure:

- **text**: The full transcribed text of the audio.
- **segments**: A list of segments, each representing a portion of the audio with detailed information.
  - **id**: The segment identifier.
  - **seek**: The seek position in the audio file.
  - **start**: The start time of the segment in seconds.
  - **end**: The end time of the segment in seconds.
  - **text**: The transcribed text for the segment.
  - **tokens**: A list of token IDs representing the transcribed text.
  - **temperature**: The temperature used for the transcription.
  - **avg_logprob**: The average log probability of the tokens.
  - **compression_ratio**: The compression ratio of the segment.
  - **no_speech_prob**: The probability that no speech is present in the segment.
- **language**: The detected language of the audio.

## Example

```json
{
    "text": " Hey, excuse me. Look. Ah, I can't handle this. I'm just going to put it on. You know, peek-a-boo! Hey.",
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 4.0,
            "text": " Hey, excuse me. Look.",
            "tokens": [
                50364,
                1911,
                11,
                8960,
                385,
                13,
                2053,
                13,
                50564
            ],
            "temperature": 0.2,
            "avg_logprob": -0.9854611106540846,
            "compression_ratio": 1.0515463917525774,
            "no_speech_prob": 0.2712443768978119
        },
        {
            "id": 1,
            "seek": 0,
            "start": 4.0,
            "end": 7.0,
            "text": " Ah, I can't handle this.",
            "tokens": [
                50564,
                2438,
                11,
                286,
                393,
                380,
                4813,
                341,
                13,
                50714
            ],
            "temperature": 0.2,
            "avg_logprob": -0.9854611106540846,
            "compression_ratio": 1.0515463917525774,
            "no_speech_prob": 0.2712443768978119
        },
        {
            "id": 2,
            "seek": 0,
            "start": 7.0,
            "end": 9.0,
            "text": " I'm just going to put it on.",
            "tokens": [
                50714,
                286,
                478,
                445,
                516,
                281,
                829,
                309,
                322,
                13,
                50814
            ],
            "temperature": 0.2,
            "avg_logprob": -0.9854611106540846,
            "compression_ratio": 1.0515463917525774,
            "no_speech_prob": 0.2712443768978119
        },
        {
            "id": 3,
            "seek": 0,
            "start": 9.0,
            "end": 11.0,
            "text": " You know, peek-a-boo!",
            "tokens": [
                50814,
                509,
                458,
                11,
                19604,
                12,
                64,
                12,
                19985,
                0,
                50914
            ],
            "temperature": 0.2,
            "avg_logprob": -0.9854611106540846,
            "compression_ratio": 1.0515463917525774,
            "no_speech_prob": 0.2712443768978119
        },
        {
            "id": 4,
            "seek": 0,
            "start": 12.0,
            "end": 13.0,
            "text": " Hey.",
            "tokens": [
                50964,
                1911,
                13,
                51014
            ],
            "temperature": 0.2,
            "avg_logprob": -0.9854611106540846,
            "compression_ratio": 1.0515463917525774,
            "no_speech_prob": 0.2712443768978119
        }
    ],
    "language": "en"
}```