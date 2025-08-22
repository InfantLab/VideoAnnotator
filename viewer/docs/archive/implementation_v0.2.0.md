# Video Action Viewer v0.2.0 Implementation Plan
### Spec [[2025-05-05]]

## Overview

Video Action Viewer v0.2.0 builds upon a successful v0.1.0 foundation to deliver a complete VideoAnnotator v1.1.1 integration with enhanced UI/UX, full pipeline support, and advanced visualization features.

**Current Status**: v0.1.0 completed with working dev server, ready for v0.2.0 feature development.

## v0.1.0 Foundation ✅ COMPLETED

- ✅ **Core Architecture**: React + TypeScript + Vite + Tailwind CSS + shadcn/ui
- ✅ **Data System**: StandardAnnotationData with COCO, WebVTT, RTTM, Scene support
- ✅ **Parser Engine**: Multi-format file detection and parsing
- ✅ **Basic UI**: File upload, video player, timeline, overlay controls
- ✅ **Demo Mode**: Working demonstration with VideoAnnotator sample data
- ✅ **Development Environment**: Bun dev server operational at http://localhost:8081

## v0.2.0 Goals

1. **Complete VideoAnnotator v1.1.1 Integration** - Full pipeline support including face analysis
2. **Enhanced User Experience** - Improved controls, timeline, and visualization
3. **Professional UI Polish** - Consistent design, better usability, advanced features
4. **Performance Optimization** - Smooth playback with large annotation datasets
5. **Export Capabilities** - Support for multiple annotation tool formats

---

## Phase 1: VideoAnnotator v1.1.1 Complete Integration

### 1.1 Face Analysis Pipeline Support

**Priority**: HIGH | **Estimated**: 3-4 days

**Files**: 
- `src/lib/parsers/face.ts` (enhance existing)
- `src/types/annotations.ts` (already updated)
- `src/lib/parsers/merger.ts` (update detection)

**Tasks**:
- [ ] Complete LAION face analysis parser implementation
- [ ] Add support for DeepFace pipeline outputs
- [ ] Add support for OpenFace3 pipeline outputs  
- [ ] Integrate face analysis into unified `complete_results.json` parsing
- [ ] Add face detection model switching support

**Face Analysis Features** (Multiple Models Simultaneously Supported):
```typescript
interface FaceAnalysisOptions {
  // Multiple models can be active simultaneously
  models: {
    laion: {
      enabled: boolean;
      showBoundingBoxes: boolean;
      showLandmarks: boolean;
      showEmotions: boolean;
    };
    deepface: {
      enabled: boolean;
      showBoundingBoxes: boolean;
      showAge: boolean;
      showGender: boolean;
      showEmotions: boolean;
    };
    openface3: {
      enabled: boolean;
      showBoundingBoxes: boolean;
      showActionUnits: boolean;
      showGaze: boolean;     // NEW: Gaze direction visualization
    };
  };
}
```

### 1.2 Complete Results Format Support

**Priority**: HIGH | **Estimated**: 2-3 days

**Files**:
- `src/lib/parsers/merger.ts`
- `src/components/FileUploader.tsx`

**Tasks**:
- [ ] Replace individual file parsing with unified `complete_results.json` support
- [ ] Handle VideoAnnotator v1.1.1 directory structure detection
- [ ] Parse processing metadata and configuration information
- [ ] Update file type detection for v1.1.1 naming patterns
- [ ] **VideoAnnotator v1.1.1 ONLY** - No backward compatibility needed (simplifies implementation)

### 1.3 Enhanced Person Tracking

**Priority**: MEDIUM | **Estimated**: 1-2 days

**Files**:
- `src/components/VideoPlayer.tsx`
- `src/lib/parsers/coco.ts`

**Tasks**:
- [ ] Implement person ID display overlays
- [ ] Add person labeling confidence visualization
- [ ] Display labeling method information
- [ ] Enhanced track ID persistence across frames
- [ ] Person-to-face association visualization

---

## Phase 2: Video Player Enhancements

### 2.1 Advanced Playback Controls

**Priority**: HIGH | **Estimated**: 2-3 days

**From TODO.md**:
- [ ] Show time with 100ths of second precision
- [ ] Display current frame & total frame numbers
- [ ] Frame-by-frame navigation controls
- [ ] Improved seek bar with frame accuracy

**Files**: `src/components/VideoPlayer.tsx`, `src/components/VideoControls.tsx` (new)

### 2.2 Enhanced Overlay System

**Priority**: HIGH | **Estimated**: 3-4 days

**From TODO.md**:
- [ ] Person tracking overlays currently not showing - **FIX CRITICAL**
- [ ] Person tracking sub-options:
  - [ ] Toggle person IDs display
  - [ ] Toggle bounding boxes
  - [ ] Toggle COCO body wireframes
- [ ] Face analysis options for different models (LAION, DeepFace, OpenFace3)
- [ ] DeepFace sub-options: age, gender, facial emotion
- [ ] OpenFace3 action unit options

**Files**: `src/components/VideoPlayer.tsx`, `src/components/OverlayControls.tsx`

### 2.3 Improved Text Overlays

**Priority**: MEDIUM | **Estimated**: 1-2 days

**From TODO.md**:
- [ ] Fix speech recognition caption alignment (currently below and offset)
- [ ] Combine Speaker IDs into speech recognition subtitles: "SPEAKER_00: Hello, baby"
- [ ] Color-code speaker text for easier reading
- [ ] Scene boundary marker labels (show for 1st second of each scene)

**Files**: `src/components/VideoPlayer.tsx`

---

## Phase 3: Control Panel Overhaul

### 3.1 Unified Overlay Controls

**Priority**: HIGH | **Estimated**: 2-3 days

**From TODO.md + Enhancements**:
- [ ] Include "Toggle All" button
- [ ] Support all possible VideoAnnotator v1.1.1 pipelines
- [ ] Grey out options if not in current annotation set
- [ ] Activate controls if JSON file exists
- [ ] Double-click option to view associated JSON file
- [ ] Hierarchical controls for complex pipelines
- [ ] **Color-coded predictor types** for easy identification across timeline and controls
- [ ] **Consistent emoji system** throughout UI for visual clarity

**Files**: `src/components/OverlayControls.tsx`

**New Control Structure with Color Coding**:
```
📹 Video Overlays                    [Toggle All]
├── 👤 Person Tracking (Orange)     [🔲] [📄]
│   ├── Person IDs                  [☑️]
│   ├── Bounding Boxes              [☑️]
│   └── COCO Wireframes            [☑️]
├── 😊 Face Analysis (Green/Red/Yellow) [🔲] [📄]
│   ├── 🟢 LAION Faces (Green)      [🔲]
│   ├── 🔴 DeepFace Analysis (Red)  [🔲]
│   │   ├── Age Estimation         [☑️]
│   │   ├── Gender Detection       [☑️]
│   │   └── Facial Emotions        [☑️]
│   └── 🟡 OpenFace3 Analysis (Yellow) [🔲]
│       ├── Action Units           [☑️]
│       └── Gaze Direction         [☑️] (NEW)
├── 🔵 Speech Recognition (Blue)    [☑️] [📄]
├── 🟣 Speaker Diarization (Purple) [☑️] [📄]
└── 🎬 Scene Detection (Teal)       [☑️] [📄]
```

**Color Scheme**:
- 🟠 **Orange**: Person Tracking
- 🟢 **Green**: LAION Face Analysis  
- 🔴 **Red**: DeepFace Analysis
- 🟡 **Yellow**: OpenFace3 Analysis
- 🔵 **Blue**: Speech Recognition
- 🟣 **Purple**: Speaker Diarization
- 🟦 **Teal**: Scene Detection

### 3.2 Enhanced Timeline Controls

**Priority**: MEDIUM | **Estimated**: 2 days

**From TODO.md**:
- [ ] Rename "Timeline Settings" to "Timeline Controls"
- [ ] Use same slider buttons as Overlay controls
- [ ] Include "Toggle All" button
- [ ] Add "Lock to Overlay" option - greys out Timeline options and syncs with Overlay controls
- [ ] Use consistent naming between Timeline and Overlay controls

**Files**: `src/components/Timeline.tsx`, `src/components/TimelineControls.tsx` (new)

---

## Phase 4: Timeline & Visualization Improvements

### 4.1 Enhanced Timeline Features

**Priority**: MEDIUM | **Estimated**: 2-3 days

**Tasks**:
- [ ] 📊 **Multi-track visualization improvements**: Enhanced visual separation of tracks with emoji labels matching control panel
- [ ] 🔍 **Better hover interactions** with detailed info popups
- [ ] 🎯 **Timeline navigation enhancements**: Improved precision and visual feedback for seeking
- [ ] 📂 **Track grouping and filtering** by pipeline type
- [ ] 🔍 **Timeline zoom controls** - zoom in/out for detailed view of annotations

**Files**: `src/components/Timeline.tsx`

### 4.2 Audio Visualization

**Priority**: LOW | **Estimated**: 2-3 days

**Tasks**:
- [ ] Waveform generation from audio files
- [ ] Synchronized audio-visual playback
- [ ] Audio timeline integration
- [ ] Speaker diarization audio visualization

**Files**: `src/components/AudioVisualizer.tsx` (new), `src/hooks/useAudioWaveform.ts` (new)

---

## Phase 5: Export & Integration Features

### 5.1 JSON File Viewer & Export

**Priority**: HIGH (JSON Viewer) / MEDIUM (Export) | **Estimated**: 2-3 days

**JSON Viewer Priority** (Higher than export):
- [ ] 📄 **JSON file viewer** for debugging and validation
- [ ] Quick access to raw annotation files underlying any pipeline
- [ ] Syntax highlighting and collapsible JSON structure
- [ ] Search and filter within JSON files

**Export Capabilities** (Lower priority):
- [ ] Export to CVAT format
- [ ] Export to LabelStudio format  
- [ ] Export to ELAN format
- [ ] Maintain VideoAnnotator compatibility
- [ ] Batch export options

**Files**: 
- `src/components/FileViewer.tsx` (new) - JSON viewer component
- `src/lib/exporters/` (new directory)
  - `cvat.ts`
  - `labelstudio.ts` 
  - `elan.ts`
  - `videoannotator.ts`

### 5.2 Advanced File Operations

**Priority**: LOW | **Estimated**: 2 days

**Tasks**:
- [ ] JSON file viewer/editor
- [ ] Configuration file management
- [ ] Annotation validation and repair
- [ ] File format conversion utilities

**Files**: `src/components/FileViewer.tsx` (new), `src/lib/utils/fileRepair.ts` (new)

---

## Phase 6: Performance & Polish

### 6.1 Performance Optimization

**Priority**: MEDIUM | **Estimated**: 2-3 days

**Tasks**:
- [ ] Large file handling optimization
- [ ] Memory management for long videos
- [ ] Canvas rendering performance
- [ ] Lazy loading for timeline tracks
- [ ] Efficient time-based data lookups

**Files**: Multiple components, focus on `VideoPlayer.tsx` and `Timeline.tsx`

### 6.2 UI/UX Polish

**Priority**: MEDIUM | **Estimated**: 2-3 days

**Tasks**:
- [ ] Consistent design system
- [ ] Loading states and progress indicators
- [ ] Error handling improvements
- [ ] Keyboard shortcuts
- [ ] Responsive design improvements
- [ ] Accessibility enhancements

**Files**: UI components across the application

---

## Implementation Priority Matrix

### Critical Fixes (Week 1)
1. **Person tracking overlays not showing** - CRITICAL BUG
2. **Complete VideoAnnotator v1.1.1 integration** (v1.1.1 ONLY - no backward compatibility)
3. **Enhanced overlay controls with toggle all** + color coding + emojis
4. **Improved playback controls with frame precision**

### Core Features (Week 2)
1. **Face analysis pipeline support** (multiple models simultaneously + gaze)
2. **Unified timeline controls** with lock-to-overlay
3. **Text overlay improvements** (speaker integration)
4. **Complete results format support**
5. **JSON file viewer** (higher priority than export)

### Polish & Advanced (Week 3-4)
1. **Timeline zoom controls** (high value feature)
2. **Performance optimization**
3. **UI/UX enhancements** with consistent color scheme
4. **Export capabilities** (lower priority)
5. **Audio visualization** (low priority)

---

## Technical Requirements

### Dependencies to Add
```json
{
  "webvtt-parser": "^2.2.0",
  "@types/webvtt-parser": "^2.2.0",
  "file-saver": "^2.0.5",
  "@types/file-saver": "^2.0.5"
}
```

### File Structure Additions
```
src/
├── components/
│   ├── VideoControls.tsx (new)
│   ├── TimelineControls.tsx (new)
│   ├── FileViewer.tsx (new)
│   └── AudioVisualizer.tsx (new)
├── lib/
│   ├── exporters/ (new)
│   │   ├── cvat.ts
│   │   ├── labelstudio.ts
│   │   ├── elan.ts
│   │   └── videoannotator.ts
│   └── utils/
│       ├── fileRepair.ts (new)
│       └── performance.ts (new)
└── hooks/
    ├── useAudioWaveform.ts (new)
    ├── usePerformance.ts (new)
    └── useKeyboardShortcuts.ts (new)
```

---

## Success Criteria

### Functional Requirements
- [ ] All VideoAnnotator v1.1.1 pipelines fully supported
- [ ] Person tracking overlays working correctly
- [ ] Face analysis with model switching
- [ ] Enhanced controls with toggle all functionality
- [ ] Improved timeline with lock-to-overlay option
- [ ] Export capabilities for major annotation tools

### Performance Requirements
- [ ] Smooth playback with large annotation files (>1GB)
- [ ] Sub-100ms response time for overlay toggles
- [ ] Efficient memory usage for long videos (>1 hour)
- [ ] Fast file loading (<5s for typical datasets)

### User Experience Requirements
- [ ] Intuitive control panel layout
- [ ] Consistent design language
- [ ] Helpful error messages and loading states
- [ ] Professional appearance suitable for research use

---

## Timeline Estimate

### Phase 1: VideoAnnotator Integration (5-7 days)
- Face analysis support: 3-4 days
- Complete results format: 2-3 days  
- Enhanced person tracking: 1-2 days

### Phase 2: Video Player (4-6 days)
- Playback controls: 2-3 days
- Overlay system: 3-4 days
- Text overlays: 1-2 days

### Phase 3: Control Panel (3-4 days)
- Overlay controls: 2-3 days
- Timeline controls: 2 days

### Phase 4: Timeline & Visualization (3-5 days)
- Timeline features: 2-3 days
- Audio visualization: 2-3 days (optional)

### Phase 5: Export & Integration (4-5 days)
- Export capabilities: 3-4 days
- File operations: 2 days

### Phase 6: Performance & Polish (3-4 days)
- Performance optimization: 2-3 days
- UI/UX polish: 2-3 days

**Total Estimated Time**: 22-31 days (4-6 weeks)

**Recommended Development Approach**: Focus on Critical Fixes and Core Features first (Phases 1-3), then add Polish & Advanced features (Phases 4-6) based on user feedback and priorities.

---

## Version 0.2.0 Deliverables

1. **Complete VideoAnnotator v1.1.1 Support** - All pipelines including face analysis (multiple models + gaze)
2. **Professional Control Panel** - Color-coded overlay/timeline controls with emoji system
3. **Advanced Video Player** - Frame-precise controls and improved overlays with speaker integration
4. **JSON File Viewer** - Debug/validation tool for raw annotation files
5. **Timeline Zoom Controls** - Enhanced navigation for detailed annotation review
6. **Performance Optimizations** - Smooth operation with large files
7. **Export Capabilities** - Support for major annotation tools (lower priority)

**Target Release**: Q3 2025 (4-6 weeks from v0.1.0)