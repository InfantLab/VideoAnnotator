# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Annotation Viewer is a React-based web application for analyzing multimodal video annotations. It displays VideoAnnotator pipeline outputs including COCO pose keypoints, WebVTT speech transcripts, RTTM speaker segments, and scene detection data synchronized with video playback.

**Key Architecture**: React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui components

## Essential Development Commands

```bash
# Development
bun run dev          # Start dev server at http://localhost:8080
bun run build        # Production build 
bun run build:dev    # Development build
bun run lint         # ESLint validation
bun run preview      # Preview production build

# Testing
# No test framework configured - manual testing with demo data
```

## Core Architecture

### Data Flow Pattern
1. **File Upload**: `FileUploader.tsx` → `merger.ts` file detection
2. **Parsing**: Format-specific parsers in `src/lib/parsers/`
3. **Unification**: `mergeAnnotationData()` creates `StandardAnnotationData`
4. **Rendering**: `VideoPlayer.tsx` canvas overlays + `Timeline.tsx` tracks
5. **State Management**: React hooks in `VideoAnnotationViewer.tsx`

### Critical File Structure
```
src/
├── components/
│   ├── VideoAnnotationViewer.tsx    # Main container component
│   ├── VideoPlayer.tsx              # Video + canvas overlay rendering
│   ├── Timeline.tsx                 # Multi-track timeline
│   ├── UnifiedControls.tsx          # v0.2.0: Combined overlay/timeline controls
│   ├── DebugPanel.tsx               # v0.2.0: Professional debugging interface
│   ├── FileUploader.tsx             # Multi-file drag/drop with enhanced detection
│   ├── FileViewer.tsx               # JSON data inspection components
│   ├── VideoControls.tsx            # Video playback controls
│   ├── WelcomeScreen.tsx            # Landing page interface
│   └── Footer.tsx                   # Footer with VideoAnnotator link
├── lib/parsers/
│   ├── merger.ts                    # Enhanced file detection & data unification
│   ├── coco.ts                      # COCO pose parsing
│   ├── webvtt.ts                    # Speech transcript parsing
│   ├── rttm.ts                      # Speaker diarization parsing
│   ├── scene.ts                     # Scene detection parsing
│   └── face.ts                      # Face analysis parsing
├── lib/fileUtils.ts                 # File type detection utilities
├── types/annotations.ts             # All TypeScript interfaces
└── utils/debugUtils.ts              # Enhanced demo data + integrity checking
```

### Data Types System
- `StandardAnnotationData`: Unified data structure for all pipelines
- `COCOPersonAnnotation`: 17-keypoint human pose data
- `WebVTTCue`: Speech recognition segments
- `RTTMSegment`: Speaker diarization segments
- `SceneAnnotation`: Scene boundary detection

### v0.2.0 UI Architecture
**Two-Column Layout**: Elegant responsive design
- **Left Column**: Video Player + Video Controls (play/pause, seeking, frame step)
- **Right Column**: Unified Controls Panel (w-96 fixed width)
- **Bottom**: Timeline Section (h-64 fixed height)

**Unified Controls Panel**: Combines overlay and timeline controls
- **Color-coded Components**: Each pipeline has distinct colored circle buttons
- **Lock Functionality**: Padlock button synchronizes overlay/timeline settings
- **JSON Viewers**: Individual data inspection for each component
- **Bulk Controls**: "All On/All Off" buttons for quick management

**Debug System**: Professional debugging interface
- **Access**: Ctrl+Shift+D keyboard shortcut or 🐛 button
- **Terminal UI**: Black background with green text logging
- **Automated Tests**: File detection and data integrity checking
- **VEATIC Support**: Specialized testing for VEATIC dataset files

### Canvas Overlay Rendering
Video overlays use HTML5 Canvas with real-time sync:
- COCO keypoints: 17-point skeleton connections
- Track IDs: Persistent person identification
- Bounding boxes: Person detection regions
- Text overlays: Subtitles and speaker labels

## File Format Support

**Input Formats**:
- Video: MP4, WebM, AVI, MOV
- COCO: JSON with keypoints array (51 values = 17 × 3)
- Speech: WebVTT format starting with "WEBVTT"
- Speakers: RTTM format (10 space-separated fields)
- Scenes: JSON array with start_time/end_time

**Critical**: File type detection in `merger.ts` uses extension + content analysis

## Development Patterns

### Adding New Annotation Types
1. Define interfaces in `types/annotations.ts`
2. Create parser in `lib/parsers/yourformat.ts`
3. Add detection logic in `merger.ts:detectFileType()`
4. Update rendering in `VideoPlayer.tsx:renderOverlays()`
5. Add timeline track in `Timeline.tsx`

### State Management Pattern
All state flows through `VideoAnnotationViewer.tsx`:
```typescript
const [currentTime, setCurrentTime] = useState(0);
const [annotationData, setAnnotationData] = useState<StandardAnnotationData | null>(null);
const [overlaySettings, setOverlaySettings] = useState<OverlaySettings>({...});
```

### Time-based Data Lookup
```typescript
// Pattern used throughout codebase for real-time sync
function getDataAtTime<T extends { timestamp: number }>(
  data: T[], 
  currentTime: number,
  tolerance = 0.1
): T[]
```

## Code Quality & Style

- **TypeScript**: Strict mode with relaxed unused variables (`tsconfig.json`)
- **ESLint**: React hooks rules enabled, unused vars disabled
- **Import aliases**: `@/` → `./src/`
- **Component naming**: PascalCase for components, camelCase for functions
- **No testing framework**: Use demo data for manual validation

## Common Development Tasks

### Working with Demo Data
```typescript
// Browser console debugging
window.debugUtils.DEMO_DATA_SETS  // Available datasets
window.debugUtils.loadDemoAnnotations('datasetKey')
window.version.getAppTitle()
```

### Parser Development
1. Study existing parsers (`coco.ts`, `webvtt.ts`, `rttm.ts`)
2. Use Zod schemas for validation (`lib/validation.ts`)
3. Transform to standard interfaces in `types/annotations.ts`
4. Handle parsing errors gracefully with user feedback

### Canvas Rendering
- Use `getCanvasContext()` helper for setup
- Clear canvas before each frame: `ctx.clearRect()`
- Coordinate system: Canvas pixels, not video coordinates
- Scale overlays based on video dimensions vs canvas display size

## Dependencies & Build

**Key Dependencies**:
- React 18 + React DOM for UI
- @radix-ui components via shadcn/ui
- Tailwind CSS for styling
- Vite as build tool with SWC React plugin
- Zod for runtime validation

**Build Output**: Static files in `dist/` suitable for any web server

## Troubleshooting

**File Upload Issues**: Check `merger.ts:detectFileType()` logic
**Rendering Performance**: Canvas operations in `VideoPlayer.tsx`
**Type Errors**: Verify interfaces in `types/annotations.ts`
**Timeline Sync**: Time-based lookups in component render methods