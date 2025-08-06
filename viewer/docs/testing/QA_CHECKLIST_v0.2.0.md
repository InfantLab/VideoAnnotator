# Quality Assurance Checklist - Video Action Viewer v 0.2.0

## Manual Testing Checklist for Human Testers

### Testing checkbox format
- [ ] unchecked boxes for tests that need doing
- [x] ticked boxes for passed tests ✅ 2025-08-06
- [f] f is for fail
      with explanatory comments below
- [>] > for next minor version
- [>>] >> for next major version

 

### Pre-Testing Setup
- [x] Ensure `bun run dev` starts successfully on port 8080 ✅ 2025-08-06
- [x] Verify demo files are available in `demo/videos_out/` directory ✅ 2025-08-06
- [x] Confirm browser console is open for debugging information ✅ 2025-08-06

---

## 1. Demo Video Loading Tests

### Load Demo Dataset
- [x] **Load peekaboo-rep3-v1.1.1**: Click "View Demo" button ✅ 2025-08-06
- [x] **Load peekaboo-rep2-v1.1.1**: Use `window.debugUtils.loadDemoAnnotations('peekaboo-rep2-v1.1.1')` in console

- [x] **Load tearingpaper-rep1-v1.1.1**: Use `window.debugUtils.loadDemoAnnotations('tearingpaper-rep1-v1.1.1')` in console
- [x] **Load thatsnotahat-rep1-v1.1.1**: Use `window.debugUtils.loadDemoAnnotations('thatsnotahat-rep1-v1.1.1')` in console


### Verify Data Loading
- [x] Console shows "✅ complete_results.json detected" message ✅ 2025-08-06
- [x] Console shows person tracking data count > 0 ✅ 2025-08-06
- [x] No "❌ No person tracking data" errors in console ✅ 2025-08-06
- [x] Video plays without errors ✅ 2025-08-06

---

## 2. Person Tracking Overlay Tests (CRITICAL)

### YOLO/Ultralytics Skeleton Rendering
- [x] **Keypoints visible**: 17 keypoints render as colored circles ✅ 2025-08-06
- [x] **Skeleton connections**: Lines connect keypoints following YOLO standard ✅ 2025-08-06 
- [x] **Color scheme**: Uses YOLO pose palette colors (oranges, blues, greens, etc.) ✅ 2025-08-06
- [x] **Proper connections**: ✅ 2025-08-06
  - Nose connects to eyes/ears
  - Arms: shoulder → elbow → wrist
  - Legs: hip → knee → ankle
  - Torso connections present
    
### Tracking Verification
- [x] **Multiple people**: If >1 person, each has different colored bounding box ✅ 2025-08-06
- [x] **Track IDs**: Person bounding boxes show "ID:X" labels ✅ 2025-08-06
- [x] **Temporal consistency**: Person tracking follows individuals across frames ✅ 2025-08-06
- [x] **No phantom overlays**: Overlays disappear when people leave frame ✅ 2025-08-06

---

## 3. File Format Support Tests

### VideoAnnotator V1.1.1 Complete Results
- [x] `complete_results.json` files detected correctly ✅ 2025-08-06
- [x] Person tracking data extracts from `pipeline_results.person.results` ✅ 2025-08-06
- [x] Face analysis data extracts from `pipeline_results.face.results` ✅ 2025-08-06
- [x] Scene detection data extracts from `pipeline_results.scene.results` ✅ 2025-08-06

### Legacy Format Support - not required
- [x] Individual JSON files for person tracking still work
- [x] WebVTT files for speech recognition load correctly
- [x] RTTM files for speaker diarization parse successfully

---

## 4. Overlay Controls Tests

### Toggle Functionality
- [x] **Pose toggle**: ON/OFF switches skeleton rendering ✅ 2025-08-06
- [x] **Subtitles toggle**: ON/OFF switches speech text overlay ✅ 2025-08-06
- [x] **Speakers toggle**: ON/OFF switches speaker diarization display ✅ 2025-08-06
- [x] **Scenes toggle**: ON/OFF switches scene detection labels ✅ 2025-08-06
- [x] **Faces toggle**: ON/OFF switches face detection boxes ✅ 2025-08-06
- [x] **Emotions toggle**: ON/OFF switches emotion labels ✅ 2025-08-06


### Visual Verification
- [x] All enabled overlays render simultaneously without conflicts ✅ 2025-08-06
- [x] Overlay settings persist during video playback ✅ 2025-08-06
- [x] No performance degradation with all overlays enabled ✅ 2025-08-06

However there is different behaviour between pipelines.  Speaker and subtitle stay present for duration of event. This is good. The skeleton renders for a give frame and then stays present until the next predicted frame. 
All other pipelines only seem to render on the frame and next 6 frames. Unclear if this is because there are sparse predictions.  Think we make this consistent with our PredictionsPerSecond rates from the pipelines. But this information doesn't seem to be available to the viewer is . we need to raise an github issue in VideoAnnotator. 
Speaker ID and Subtitles not yet combined
Subtitles still rendering below the screen

---

## 5. Timeline and Playback Tests

### Video Controls
- [x] **Play/Pause**: Space bar or button toggles playback ✅ 2025-08-06
- [x] **Seeking**: Click timeline or use slider to jump to time ✅ 2025-08-06
- [x] **Frame stepping**: Arrow keys advance/rewind by single frames ✅ 2025-08-06
- [x] **Playback speed**: 0.25x, 0.5x, 1x, 1.5x, 2x options work ✅ 2025-08-06
- The speaker button on controls doesn't toggle sound on/off.

### Timeline Tracks
- [x] **Subtitle track**: Shows WebVTT speech recognition segments ✅ 2025-08-06
      Next we should put the words on the timeline too.
      Audio wave form is already on the list but we should group it with other audio tracks
- [x] **Speaker track**: Shows RTTM diarization segments ✅ 2025-08-06
- [x] **Scene track**: Shows scene detection boundaries ✅ 2025-08-06
- [>] **Motion track**: Shows person tracking activity (if implemented)
      Should have one track per person - that tracks their average x ordinate (centre of bbox)
      Partially - we ought to create a motion intensity algorithm (something industry standard)

---

## 6. Error Handling and Edge Cases

### File Detection Issues
- [x] **Unknown files**: Gracefully handle unsupported file types ✅ 2025-08-06
- [x] **Malformed JSON**: Show helpful error messages ✅ 2025-08-06
	almost - currently says 'unknown file type'
- [f] **Missing data**: Handle empty pipeline results
      Need to improve loading screen (see above)
- [ ] **Large files**: Performance acceptable for >10MB annotation files

### Playback Edge Cases
- [x] **Video start/end**: Overlays behave correctly at 0s and end time ✅ 2025-08-06
- [x] **Rapid seeking**: No overlay rendering artifacts during fast seeks ✅ 2025-08-06
- [x] **Browser resize**: Canvas overlays scale correctly with window size

---

## 7. Performance Tests

### Loading Performance
- [x] **Demo loading**: Completes within 5 seconds ✅ 2025-08-06
- [>] **Large datasets**: >1000 person tracking entries load without freezing
- [>] **Memory usage**: No obvious memory leaks during extended use

### Rendering Performance
- [f] **Smooth playback**: 30fps video maintains smooth overlay rendering
- [f] **Multiple people**: Performance acceptable with 3+ people in frame
- [f] **All overlays**: No significant lag with all overlay types enabled
- Some timing issues with the VEATIC 3.mp4 Patch Adams video. It also doesn't show good tracking for all the people in the video. But this might be a problem with the VideoAnnotator pipeline. So needs further investigation.
---

## 8. Browser Compatibility

### Desktop Browsers
- [ ] **Chrome**: Full functionality
- [x] **Firefox**: Full functionality  
- [ ] **Safari**: Full functionality (if macOS available)
- [x] **Edge**: Full functionality ✅ 2025-08-06

### Console Debugging
- [x] `window.debugUtils` available for testing
- [x] `window.debugUtils.DEMO_DATA_SETS` shows available datasets
- [x] No critical JavaScript errors in console

---

## 9. Data Accuracy Tests

### COCO Person Tracking
- [x] **17 keypoints**: Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles ✅ 2025-08-06
- [x] **Keypoint visibility**: Only visible keypoints (visibility=2) render ✅ 2025-08-06
- [x] **Bounding boxes**: Accurate to person locations in video ✅ 2025-08-06
- [x] **Timestamps**: Overlays sync correctly with video time ✅ 2025-08-06

### Speech Recognition
- [x] **WebVTT timing**: Subtitles appear/disappear at correct times ✅ 2025-08-06
- [x] **Text accuracy**: Readable text content displays ✅ 2025-08-06
- [x] **Positioning**: Subtitles appear at bottom of video ✅ 2025-08-06

---

## 10. Integration Tests

### VideoAnnotator V1.1.1 Pipeline
- [x] **Config preservation**: Processing settings show in metadata ✅ 2025-08-06
- [x] **Multi-pipeline**: Person + face + scene data from single file ✅ 2025-08-06
- [x] **Version compatibility**: v1.1.1 format fully supported ✅ 2025-08-06
- [x] **Backwards compatibility**: Earlier formats still work
- [>] let's remove support for obsolete versions  - we have no users yet so no need for support.

---
## 11. Additional changes
- [ ] Change the browser icon from lovable icon to something video related (use cc licenced icons)
- [ ] Once viewer works we will replace images in landing with screenshots of the viewer
- [ ] Add a link to the VideoAnnotator documentation in the footer with explanation taht it gnereates the data files used by this viewer.
- [ ] Add option to go back to landing page from viewer
- [ ] Refreshing page should reload the viewer with the last loaded video and annotations, NOt go back to landing page as at present.
- [ ] Ought to have some online docmentation for the viewer. And explanation of VidoeAnnoatator and its pipelines .

## Bug Reporting Format

When issues are found, report using this format:

```
**Bug**: [Brief description]
**Steps**: 1. Action 2. Action 3. Result
**Expected**: What should happen
**Actual**: What actually happens
**Browser**: Chrome/Firefox/Safari version
**Dataset**: Which demo video was used
**Console**: Any error messages
**Priority**: High/Medium/Low
```

---

## Testing Notes

- Test each demo dataset individually 
- Pay special attention to YOLO skeleton connections and colors
- Verify person tracking data loads from complete_results.json format
- Report any performance issues or visual artifacts
- Check console for debugging information during tests