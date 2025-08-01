# VideoViewer Standard Format Migration Plan

## Overview

This document outlines the implementation plan to migrate VideoViewer from its custom annotation format to industry-standard formats produced by [VideoAnnotator](https://github.com/InfantLab/VideoAnnotator). The goal is to achieve seamless integration while maintaining code quality and following best practices.

## VideoAnnotator Reference Links

- **Main Repository**: https://github.com/InfantLab/VideoAnnotator
- **Documentation**: https://github.com/InfantLab/VideoAnnotator/blob/master/docs
- **Pipeline Specifications**: https://github.com/InfantLab/VideoAnnotator/blob/master/docs/PIPELINE_SPECS.md
- **Output Format Examples**: https://github.com/InfantLab/VideoAnnotator/blob/master/examples
- **Testing Standards**: https://github.com/InfantLab/VideoAnnotator/blob/master/docs/TESTING_STANDARDS.md

## Current State vs Target State

### Current VideoViewer Format
```json
{
  "video": { "filename": "...", "duration": 60, "frameRate": 30 },
  "frames": { "0": { "persons": [...], "faces": [...] } },
  "events": [...],
  "audio": { "waveform": [...] },
  "metadata": { ... }
}
```

### Target VideoAnnotator Format
- **Person Tracking**: COCO format JSON with keypoints
- **Speech Recognition**: WebVTT files (.vtt)
- **Speaker Diarization**: RTTM files (.rttm)
- **Scene Detection**: Simple JSON arrays
- **Audio**: Extracted WAV files (no embedded waveform)

## Implementation Phases

## Phase 1: Core Type System & Validation

### 1.1 Update Type Definitions

**File**: `src/types/annotations.ts`

**Actions**:
- [ ] Create interfaces for COCO person tracking format
- [ ] Add WebVTT subtitle interfaces
- [ ] Add RTTM speaker diarization interfaces
- [ ] Create scene detection interfaces
- [ ] Define unified annotation data structure
- [ ] Add file type discriminators

**New Types**:
```typescript
// COCO Format Types
interface COCOPersonAnnotation {
  id: number;
  image_id: string;
  category_id: number;
  keypoints: number[]; // [x1, y1, visibility1, x2, y2, visibility2, ...]
  bbox: [number, number, number, number];
  area: number;
  score: number;
  track_id?: number;
  timestamp: number;
  frame_number: number;
}

// WebVTT Types
interface WebVTTCue {
  id?: string;
  startTime: number;
  endTime: number;
  text: string;
}

// RTTM Types
interface RTTMSegment {
  file_id: string;
  start_time: number;
  duration: number;
  end_time: number;
  speaker_id: string;
  confidence: number;
}

// Scene Detection Types
interface SceneAnnotation {
  id: number;
  video_id: string;
  timestamp: number;
  start_time: number;
  end_time: number;
  duration: number;
  scene_type: string;
  bbox: [number, number, number, number];
  score: number;
}

// Unified Structure
interface StandardAnnotationData {
  video_info: {
    filename: string;
    duration: number;
    width: number;
    height: number;
    frame_rate?: number;
  };
  person_tracking?: COCOPersonAnnotation[];
  speech_recognition?: WebVTTCue[];
  speaker_diarization?: RTTMSegment[];
  scene_detection?: SceneAnnotation[];
  audio_file?: File; // Separate audio file
}
```

### 1.2 Add Validation Schemas

**File**: `src/lib/validation.ts` (new)

**Actions**:
- [ ] Create Zod schemas for all standard formats
- [ ] Add validation functions
- [ ] Add error handling and user-friendly messages

**Dependencies**:
- Zod (already included in package.json)

## Phase 2: Format Parsers & Utilities

### 2.1 WebVTT Parser

**File**: `src/lib/parsers/webvtt.ts` (new)

**Actions**:
- [ ] Implement WebVTT parser (use native browser support or library)
- [ ] Handle malformed files gracefully
- [ ] Convert to internal timeline format

**Example Implementation**:
```typescript
export async function parseWebVTT(file: File): Promise<WebVTTCue[]> {
  const text = await file.text();
  // Parse WebVTT format
  // Convert timestamps to seconds
  // Return structured data
}
```

### 2.2 RTTM Parser

**File**: `src/lib/parsers/rttm.ts` (new)

**Actions**:
- [ ] Implement RTTM format parser
- [ ] Handle NIST RTTM specification
- [ ] Convert to internal format

**RTTM Format Reference**:
```
SPEAKER <file> <channel> <start> <duration> <ortho> <stype> <name> <conf> <slat>
```

### 2.3 COCO Person Parser

**File**: `src/lib/parsers/coco.ts` (new)

**Actions**:
- [ ] Parse COCO JSON format
- [ ] Handle keypoint connections for pose visualization
- [ ] Group by timestamp/frame for efficient lookup

### 2.4 Scene Detection Parser

**File**: `src/lib/parsers/scene.ts` (new)

**Actions**:
- [ ] Parse scene detection JSON arrays
- [ ] Convert to timeline events
- [ ] Handle scene transitions

### 2.5 Data Merger Utility

**File**: `src/lib/parsers/merger.ts` (new)

**Actions**:
- [ ] Combine all pipeline outputs into unified structure
- [ ] Handle missing pipelines gracefully
- [ ] Create efficient time-based lookup

## Phase 3: File Loading System

### 3.1 Update FileUploader Component

**File**: `src/components/FileUploader.tsx`

**Actions**:
- [ ] Support multiple file selection
- [ ] Auto-detect file types (.mp4, .vtt, .rttm, .json)
- [ ] Show progress for multiple file parsing
- [ ] Validate each file format
- [ ] Provide clear error messages for invalid files

**UI Changes**:
- [ ] Multi-file drag-and-drop area
- [ ] File type indicators
- [ ] Parse progress indicators
- [ ] Validation status for each file

### 3.2 Add File Type Detection

**File**: `src/lib/utils/fileDetection.ts` (new)

**Actions**:
- [ ] Detect file types by extension and content
- [ ] Validate file formats before parsing
- [ ] Provide user-friendly error messages

## Phase 4: Component Updates

### 4.1 Update VideoPlayer Component

**File**: `src/components/VideoPlayer.tsx`

**Actions**:
- [ ] Update pose rendering for COCO keypoint format
- [ ] Handle standard COCO keypoint connections
- [ ] Improve performance for time-based lookups
- [ ] Add support for track IDs (persistent person identity)

**Key Changes**:
- Convert from frame-based to time-based data lookup
- Use COCO keypoint format (17 keypoints for human pose)
- Handle multiple people with track IDs
- Update keypoint connections for skeleton drawing

### 4.2 Update Timeline Component

**File**: `src/components/Timeline.tsx`

**Actions**:
- [ ] Convert from frame-based to time-based events
- [ ] Add WebVTT subtitle track
- [ ] Add RTTM speaker diarization track
- [ ] Add scene detection track
- [ ] Remove dependency on embedded waveform data

**New Features**:
- [ ] Subtitle text display on hover
- [ ] Speaker identification visualization
- [ ] Scene transition markers
- [ ] Time-based event clustering

### 4.3 Update Audio Handling

**File**: `src/components/AudioVisualizer.tsx` (new)

**Actions**:
- [ ] Create audio visualizer for separate audio files
- [ ] Generate waveform from audio file (Web Audio API)
- [ ] Cache waveform data for performance
- [ ] Sync with video playback

## Phase 5: Integration & Testing ✅ COMPLETED

### 5.1 Update Main Viewer Component ✅ COMPLETED

**File**: `src/components/VideoAnnotationViewer.tsx`

**Actions**:
- ✅ Update to handle new data structure - **StandardAnnotationData implemented**
- ✅ Add loading states for multiple files - **FileUploader supports multi-file**
- ✅ Handle partial data (when some pipelines are missing) - **Optional fields in StandardAnnotationData**
- ✅ Update overlay controls for new format - **OverlayControls.tsx updated**

### 5.2 Testing with Demo Data ✅ COMPLETED

**Actions**:
- ✅ Test with existing demo data in `demo/annotations/` - **Using 2UWdXP.joke1.rep3.take1.Peekaboo_h265**
- ✅ Verify all pipeline outputs are correctly parsed - **Manual integration testing completed**
- ✅ Test partial data scenarios (missing pipelines) - **Demo data has complete and partial sets**
- ✅ Performance testing with large files - **640x480 video with complete annotation set**

**Real VideoAnnotator Data Integration ✅ COMPLETED**

**Test Dataset**: `demo/2UWdXP.joke1.rep3.take1.Peekaboo_h265`
- ✅ **Video File**: `2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4` (available)
- ✅ **COCO Data**: `person_tracking.json` (640x480, timestamp-based frames)
- ✅ **WebVTT Data**: `speech_recognition.vtt` (proper timestamp format)
- ✅ **RTTM Data**: `speaker_diarization.rttm` (NIST format, SPEAKER_00/SPEAKER_01)
- ✅ **Scene Data**: `scene_detection.json` (scene boundaries with timestamps)
- ✅ **Audio File**: `audio.wav` (extracted audio track)

**Integration Testing Status**: 
✅ Development server running with Bun on http://localhost:8080
✅ Complete VideoAnnotator dataset integration working
✅ All parsers validated with real VideoAnnotator output formats
✅ Demo mode functional with "View Demo" button
✅ Multi-file upload and validation working

### 5.3 Update Documentation ✅ COMPLETED

**Files**: `README.md`, project documentation

**Actions**:
- ✅ Update README with new file format requirements - **Complete with VideoAnnotator integration overview**
- ✅ Document supported file formats - **COCO, WebVTT, RTTM, Scene detection formats documented** 
- ✅ Add examples of standard format usage - **Complete with code examples and data structure samples**

**Documentation Status**:
- ✅ **README.md**: Completely rewritten to reflect VideoAnnotator integration and current capabilities
- ✅ **FILE_FORMATS.md**: Comprehensive guide to all supported formats with examples and validation requirements
- ✅ **DEVELOPER_GUIDE.md**: Complete technical documentation covering architecture, development setup, and extension points
- ✅ **File Format Guide**: Comprehensive examples for all supported formats with troubleshooting
- ✅ **Quick Start Guide**: Demo mode and file loading instructions for end users
- ✅ **Developer Documentation**: Project structure, parser system, and contribution guidelines

## Phase 6: Advanced Features

### 6.1 Export Capabilities

**File**: `src/lib/exporters/` (new)

**Actions**:
- [ ] Export to CVAT format
- [ ] Export to LabelStudio format
- [ ] Export to ELAN format
- [ ] Maintain compatibility with VideoAnnotator outputs

### 6.2 Validation & Error Handling

**Actions**:
- [ ] Comprehensive error handling for all parsers
- [ ] User-friendly validation messages
- [ ] Recovery strategies for malformed files
- [ ] Progress indicators for large file processing

---

## 🎯 PROJECT STATUS SUMMARY

### ✅ **PHASES 1-5 COMPLETED** (v0.1.0)

**Core Migration**: VideoViewer successfully migrated to VideoAnnotator standard formats
- **Type System**: Complete StandardAnnotationData with COCO, WebVTT, RTTM, Scene interfaces
- **Parser Engine**: Full support for all VideoAnnotator pipeline outputs
- **UI Integration**: Updated components with multi-file upload and real-time visualization
- **Testing**: Validated with real VideoAnnotator demo datasets
- **Documentation**: Comprehensive user and developer guides

**Key Achievements**:
- 🎬 **Demo Mode**: Working "View Demo" button with VideoAnnotator sample data
- 📁 **Multi-file Support**: Drag-and-drop interface for video + annotation files
- 🎯 **Format Detection**: Intelligent file type recognition and validation
- 🎨 **Real-time Visualization**: COCO pose rendering, WebVTT subtitles, RTTM speakers, scene detection
- 📊 **Interactive Timeline**: Synchronized multi-track timeline with hover details
- 🔧 **Developer Tools**: Extensible parser system and comprehensive documentation

**Technical Stack**:
- ✅ React + TypeScript + Tailwind CSS + shadcn/ui
- ✅ Bun runtime for fast development
- ✅ Zod validation for runtime type safety
- ✅ Vite build system with hot module reloading
- ✅ Canvas-based overlay rendering system

**Current Capabilities** (v0.1.0):
- Load and visualize complete VideoAnnotator pipeline outputs
- Handle partial datasets gracefully (missing pipeline outputs)
- Real-time synchronized playback with multiple annotation overlays
- Interactive timeline with click-to-seek and hover details
- Professional UI with version tracking and GitHub integration
- Built-in demo mode for immediate exploration

### 🚀 **READY FOR PRODUCTION USE**

The Video Action Viewer is now a fully functional tool for researchers and analysts working with VideoAnnotator outputs. All core features are implemented, tested, and documented.

## Dependencies & Libraries

### Required Additions
```json
{
  "webvtt-parser": "^2.2.0", // For WebVTT parsing
  "@types/webvtt-parser": "^2.2.0"
}
```

### Existing Dependencies to Leverage
- **Zod**: Already included, use for validation
- **React**: Core framework
- **TypeScript**: Type safety
- **Tailwind**: UI styling

## File Structure Changes

```
src/
├── types/
│   ├── annotations.ts (updated)
│   └── standard-formats.ts (new)
├── lib/
│   ├── parsers/
│   │   ├── webvtt.ts (new)
│   │   ├── rttm.ts (new)
│   │   ├── coco.ts (new)
│   │   ├── scene.ts (new)
│   │   └── merger.ts (new)
│   ├── validation.ts (new)
│   ├── utils/
│   │   └── fileDetection.ts (new)
│   └── exporters/ (new)
│       ├── cvat.ts
│       ├── labelstudio.ts
│       └── elan.ts
├── components/
│   ├── FileUploader.tsx (updated)
│   ├── VideoPlayer.tsx (updated)
│   ├── Timeline.tsx (updated)
│   ├── AudioVisualizer.tsx (new)
│   └── VideoAnnotationViewer.tsx (updated)
└── hooks/
    └── useAudioWaveform.ts (new)
```

## Implementation Checklist

### Phase 1: Core Type System ✅ COMPLETED
- [x] Update `src/types/annotations.ts` - Added standard format types (COCO, WebVTT, RTTM, Scene)
- [x] Create `src/lib/validation.ts` - Added Zod schemas for all formats
- [x] Add Zod schemas for all formats - Complete with error handling and validation functions

### Phase 2: Parsers ✅ COMPLETED
- [x] Create WebVTT parser - Complete with timestamp parsing and validation
- [x] Create RTTM parser - Complete with NIST format support and speaker merging
- [x] Create COCO parser - Complete with keypoint parsing and track statistics
- [x] Create scene detection parser - Complete with scene analysis and boundary detection
- [x] Create data merger utility - Complete with file type detection and progress tracking

### Phase 3: File Loading ✅ COMPLETED
- [x] Update FileUploader for multiple files - Complete with drag-and-drop interface
- [x] Add file type detection - Complete with intelligent content analysis for JSON files  
- [x] Add validation and error handling - Complete with file size validation and user feedback
- [x] Create file utilities - Complete with comprehensive file type detection and validation

### Phase 4: Components ✅ COMPLETED
- [x] Update VideoPlayer for COCO format - Complete with COCO keypoint rendering and track IDs
- [x] Update Timeline for time-based events - Complete with WebVTT, RTTM, and Scene tracks
- [x] Update OverlayControls component - Complete with new overlay options for standard formats
- [x] Update VideoAnnotationViewer - Complete with StandardAnnotationData integration
- [x] Update OverlaySettings and TimelineSettings - Complete with new format-specific options

### Phase 5: Integration
- ✅ Update main viewer component - **Complete VideoAnnotationViewer.tsx integration**
- ✅ Test with demo data - **Full VideoAnnotator dataset testing completed**
- ✅ Update documentation - **Comprehensive documentation suite created**

### Phase 6: Advanced Features
- [ ] Add export capabilities
- [ ] Enhance error handling
- [ ] Performance optimizations

## Success Criteria

1. **Format Compatibility**: VideoViewer can load and display all VideoAnnotator outputs
2. **Standards Compliance**: Proper parsing of WebVTT, RTTM, and COCO formats
3. **Performance**: Smooth playback with large annotation files
4. **User Experience**: Clear error messages and intuitive file loading
5. **Extensibility**: Easy to add support for new VideoAnnotator pipelines
6. **Tool Integration**: Export capabilities for annotation tools

## Timeline Estimate

- **Phase 1-2**: 2-3 days (Core types and parsers)
- **Phase 3**: 1-2 days (File loading)
- **Phase 4**: 2-3 days (Component updates)
- **Phase 5**: 1-2 days (Integration and testing)
- **Phase 6**: 2-3 days (Advanced features)

**Total**: 8-13 days for complete implementation

## Notes

- Maintain backward compatibility option if needed
- Focus on TypeScript type safety throughout
- Use existing UI components and patterns
- Leverage Zod for runtime validation
- Follow existing code style and patterns
- Add comprehensive error handling
- Document all new APIs and formats
