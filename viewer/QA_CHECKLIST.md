# Quality Assurance Checklist - Video Action Viewer

## Manual Testing Checklist for Human Testers

### Pre-Testing Setup
- [ ] Ensure `bun run dev` starts successfully on port 8081
- [ ] Verify demo files are available in `demo/videos_out/` directory
- [ ] Confirm browser console is open for debugging information

---

## 1. Demo Video Loading Tests

### Load Demo Dataset
- [ ] **Load peekaboo-rep3-v1.1.1**: Click "View Demo" button
- [ ] **Load peekaboo-rep2-v1.1.1**: Use `window.debugUtils.loadDemoAnnotations('peekaboo-rep2-v1.1.1')` in console
- [ ] **Load tearingpaper-rep1-v1.1.1**: Use `window.debugUtils.loadDemoAnnotations('tearingpaper-rep1-v1.1.1')` in console
- [ ] **Load thatsnotahat-rep1-v1.1.1**: Use `window.debugUtils.loadDemoAnnotations('thatsnotahat-rep1-v1.1.1')` in console

### Verify Data Loading
- [ ] Console shows "✅ complete_results.json detected" message
- [ ] Console shows person tracking data count > 0
- [ ] No "❌ No person tracking data" errors in console
- [ ] Video plays without errors

---

## 2. Person Tracking Overlay Tests (CRITICAL)

### YOLO/Ultralytics Skeleton Rendering
- [ ] **Keypoints visible**: 17 keypoints render as colored circles
- [ ] **Skeleton connections**: Lines connect keypoints following YOLO standard
- [ ] **Color scheme**: Uses YOLO pose palette colors (oranges, blues, greens, etc.)
- [ ] **Proper connections**: 
  - Nose connects to eyes/ears
  - Arms: shoulder → elbow → wrist
  - Legs: hip → knee → ankle
  - Torso connections present

### Tracking Verification
- [ ] **Multiple people**: If >1 person, each has different colored bounding box
- [ ] **Track IDs**: Person bounding boxes show "ID:X" labels
- [ ] **Temporal consistency**: Person tracking follows individuals across frames
- [ ] **No phantom overlays**: Overlays disappear when people leave frame

---

## 3. File Format Support Tests

### VideoAnnotator v1.1.1 Complete Results
- [ ] `complete_results.json` files detected correctly
- [ ] Person tracking data extracts from `pipeline_results.person.results`
- [ ] Face analysis data extracts from `pipeline_results.face.results` 
- [ ] Scene detection data extracts from `pipeline_results.scene.results`

### Legacy Format Support
- [ ] Individual JSON files for person tracking still work
- [ ] WebVTT files for speech recognition load correctly
- [ ] RTTM files for speaker diarization parse successfully

---

## 4. Overlay Controls Tests

### Toggle Functionality
- [ ] **Pose toggle**: ON/OFF switches skeleton rendering
- [ ] **Subtitles toggle**: ON/OFF switches speech text overlay
- [ ] **Speakers toggle**: ON/OFF switches speaker diarization display
- [ ] **Scenes toggle**: ON/OFF switches scene detection labels
- [ ] **Faces toggle**: ON/OFF switches face detection boxes
- [ ] **Emotions toggle**: ON/OFF switches emotion labels

### Visual Verification
- [ ] All enabled overlays render simultaneously without conflicts
- [ ] Overlay settings persist during video playback
- [ ] No performance degradation with all overlays enabled

---

## 5. Timeline and Playback Tests

### Video Controls
- [ ] **Play/Pause**: Space bar or button toggles playback
- [ ] **Seeking**: Click timeline or use slider to jump to time
- [ ] **Frame stepping**: Arrow keys advance/rewind by single frames
- [ ] **Playback speed**: 0.25x, 0.5x, 1x, 1.5x, 2x options work

### Timeline Tracks
- [ ] **Subtitle track**: Shows WebVTT speech recognition segments
- [ ] **Speaker track**: Shows RTTM diarization segments
- [ ] **Scene track**: Shows scene detection boundaries
- [ ] **Motion track**: Shows person tracking activity (if implemented)

---

## 6. Error Handling and Edge Cases

### File Detection Issues
- [ ] **Unknown files**: Gracefully handle unsupported file types
- [ ] **Malformed JSON**: Show helpful error messages
- [ ] **Missing data**: Handle empty pipeline results
- [ ] **Large files**: Performance acceptable for >10MB annotation files

### Playback Edge Cases
- [ ] **Video start/end**: Overlays behave correctly at 0s and end time
- [ ] **Rapid seeking**: No overlay rendering artifacts during fast seeks
- [ ] **Browser resize**: Canvas overlays scale correctly with window size

---

## 7. Performance Tests

### Loading Performance
- [ ] **Demo loading**: Completes within 5 seconds
- [ ] **Large datasets**: >1000 person tracking entries load without freezing
- [ ] **Memory usage**: No obvious memory leaks during extended use

### Rendering Performance
- [ ] **Smooth playback**: 30fps video maintains smooth overlay rendering
- [ ] **Multiple people**: Performance acceptable with 3+ people in frame
- [ ] **All overlays**: No significant lag with all overlay types enabled

---

## 8. Browser Compatibility

### Desktop Browsers
- [ ] **Chrome**: Full functionality
- [ ] **Firefox**: Full functionality  
- [ ] **Safari**: Full functionality (if macOS available)
- [ ] **Edge**: Full functionality

### Console Debugging
- [ ] `window.debugUtils` available for testing
- [ ] `window.debugUtils.DEMO_DATA_SETS` shows available datasets
- [ ] No critical JavaScript errors in console

---

## 9. Data Accuracy Tests

### COCO Person Tracking
- [ ] **17 keypoints**: Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- [ ] **Keypoint visibility**: Only visible keypoints (visibility=2) render
- [ ] **Bounding boxes**: Accurate to person locations in video
- [ ] **Timestamps**: Overlays sync correctly with video time

### Speech Recognition
- [ ] **WebVTT timing**: Subtitles appear/disappear at correct times
- [ ] **Text accuracy**: Readable text content displays
- [ ] **Positioning**: Subtitles appear at bottom of video

---

## 10. Integration Tests

### VideoAnnotator v1.1.1 Pipeline
- [ ] **Config preservation**: Processing settings show in metadata
- [ ] **Multi-pipeline**: Person + face + scene data from single file
- [ ] **Version compatibility**: v1.1.1 format fully supported
- [ ] **Backwards compatibility**: Earlier formats still work

---

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