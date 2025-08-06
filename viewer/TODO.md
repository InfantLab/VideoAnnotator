# Version 0.1.0 > 0.2.0

## Video Playback Controls 
- show time with 100ths of second
- Show current frame & total frame numbers. 

## Overlays Controls 
- Include a toggle all button.
- Controls should include all possible VideoAnnotator 1.1.1 pipelines - Greyed out if not in the current annotation set. active if json file exists
- Would be good to be able to double click on each option to view the associated Json file.
- Person tracking should have sub-options for showing personIDs, bounding-boxes and coco body wireframes 
- Faces should have options for different models 
- Deepface should have suboptions for age, gender, facial emotion
- Openface3 shoudl have it's options (see VideoAnnotator 1.1.1)
# Timeline Settings
+ Let's call this Timeline Controls
+ Use same slider buttons as Overlay controls
+ Include a toggle all button.
+ Include an option to "Lock to Overlay" - which greys out TImeline Options and syncs their statuts (on vs off) to the same as Overlay controls. 
+ Use consistent naming between Timeline and Overlay controls.

## Video 
+ Person tracking overlays not currently show
- Speaker Diarization - Let's find a better place to display this - I think we could combine Speaker IDs into the Speech Recognition subtitles: e.g "SPEAKER_00: Hello, baby how are you" Colour code so that speech is easier to read. 
- Speech recognition caption not currently aligning with the video playback frame - it's below and offset to the right
- Scene boundary marker label should appear for 1st second of each scene
- Person tracking overlays not currently show
- Speaker Diarization - Let's find a better place to display this - I think we could combine Speaker IDs into the Speech Recognition subtitles: e.g "SPEAKER_00: Hello, baby how are you" Colour code so that speech is easier to read. 
- Face detection needs options for all supported 

