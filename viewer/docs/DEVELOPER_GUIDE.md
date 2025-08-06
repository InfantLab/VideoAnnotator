# Developer Guide

This guide covers the technical architecture, development setup, and extension points for Video Action Viewer.

## Architecture Overview

### Tech Stack
- **Runtime**: Bun (preferred) or Node.js 18+
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS + shadcn/ui components
- **Build Tool**: Vite
- **State Management**: React hooks (useState, useCallback, useRef)
- **Validation**: Zod schemas for runtime type checking

### Project Structure

```
src/
├── components/              # React components
│   ├── VideoPlayer.tsx      # Main video player with overlays
│   ├── Timeline.tsx         # Interactive timeline
│   ├── FileUploader.tsx     # Multi-file upload interface
│   ├── OverlayControls.tsx  # Annotation toggle controls
│   ├── WelcomeScreen.tsx    # Landing page
│   ├── Footer.tsx           # Version and GitHub info
│   └── ui/                  # shadcn/ui base components
├── lib/                     # Core business logic
│   ├── parsers/             # Format-specific parsers
│   │   ├── coco.ts          # COCO keypoint parsing
│   │   ├── webvtt.ts        # WebVTT subtitle parsing
│   │   ├── rttm.ts          # RTTM speaker parsing
│   │   ├── scene.ts         # Scene detection parsing
│   │   └── merger.ts        # Data integration
│   └── validation.ts        # Zod schemas
├── types/                   # TypeScript definitions
│   └── annotations.ts       # Standard format interfaces
├── utils/                   # Utility functions
│   ├── debugUtils.ts        # Demo data loading
│   └── version.ts           # Version management
├── hooks/                   # Custom React hooks
│   ├── use-mobile.tsx       # Mobile detection
│   └── use-toast.ts         # Toast notifications
└── assets/                  # Static assets
    └── VideoAnnotationViewer.png
```

## Development Setup

### Prerequisites
```bash
# Install Bun (recommended)
curl -fsSL https://bun.sh/install | bash

# Or use Node.js 18+
node --version  # Should be 18+
```

### Getting Started
```bash
# Clone the repository
git clone https://github.com/InfantLab/video-annotation-viewer.git
cd video-annotation-viewer

# Install dependencies
bun install
# or: npm install

# Start development server
bun run dev
# or: npm run dev

# Build for production
bun run build
# or: npm run build
```

### Available Scripts
```bash
bun run dev        # Start development server (http://localhost:8080)
bun run build      # Production build
bun run build:dev  # Development build
bun run lint       # ESLint checking
bun run preview    # Preview production build
```

## Core Components

### VideoPlayer.tsx
Main video component with overlay rendering.

**Key Features**:
- Canvas-based overlay system
- COCO keypoint rendering (17-point human pose)
- Real-time synchronization with video playback
- Person tracking with persistent IDs

**Extension Points**:
```typescript
// Add new overlay types
interface OverlaySettings {
  pose: boolean;
  subtitles: boolean;
  speakers: boolean;
  scenes: boolean;
  // Add your overlay here
  newOverlay: boolean;
}

// Implement rendering in renderOverlays()
const renderOverlays = () => {
  if (overlaySettings.newOverlay) {
    renderNewOverlay(ctx, currentTime);
  }
};
```

### Timeline.tsx
Interactive timeline with multiple tracks.

**Tracks**:
- Speech recognition (WebVTT)
- Speaker diarization (RTTM)
- Scene detection (JSON)
- Custom tracks (extensible)

**Extension Points**:
```typescript
// Add new timeline tracks
interface TimelineSettings {
  speech: boolean;
  speakers: boolean;
  scenes: boolean;
  // Add your track here
  newTrack: boolean;
}
```

### FileUploader.tsx
Multi-file upload with validation.

**Supported Operations**:
- Drag-and-drop file handling
- File type detection and validation
- Progress tracking for large files
- Error handling with user feedback

## Data Flow

### 1. File Loading
```
User drops files → FileUploader → detectFileType() → Validation → Parser selection
```

### 2. Parsing Pipeline
```
Raw files → Format-specific parsers → mergeAnnotationData() → StandardAnnotationData
```

### 3. Rendering Loop
```
Video time update → Data lookup by timestamp → Canvas overlay rendering → UI sync
```

## Parser System

### Adding New Parsers

1. **Create parser file**: `src/lib/parsers/yourformat.ts`
```typescript
import { z } from 'zod';

// Define validation schema
const YourFormatSchema = z.object({
  timestamp: z.number(),
  data: z.string(),
  // ... other fields
});

export async function parseYourFormat(file: File): Promise<YourFormatData[]> {
  const content = await file.text();
  const parsed = JSON.parse(content);
  
  // Validate with Zod
  const validated = z.array(YourFormatSchema).parse(parsed);
  
  // Transform to internal format
  return validated.map(item => ({
    timestamp: item.timestamp,
    // ... transform data
  }));
}
```

2. **Add to merger.ts**:
```typescript
// In detectFileType()
if (fileName.includes('yourformat')) {
  return { file, type: 'your_format', pipeline: 'your_pipeline', confidence: 0.9 };
}

// In mergeAnnotationData()
case 'your_format':
  const yourData = await parseYourFormat(detectedFile.file);
  // Store in result object
  break;
```

3. **Update types**: Add interfaces to `src/types/annotations.ts`

### File Type Detection

The system uses multiple detection methods:

```typescript
export async function detectFileType(file: File): Promise<DetectedFile> {
  const extension = file.name.toLowerCase().split('.').pop();
  const mimeType = file.type.toLowerCase();

  // 1. Video/Audio by MIME type
  if (mimeType.startsWith('video/')) {
    return { file, type: 'video', confidence: 0.95 };
  }

  // 2. Extension-based detection
  if (extension === 'vtt') {
    return await validateWebVTT(file);
  }

  // 3. Content-based detection for JSON
  if (extension === 'json') {
    return await detectJSONType(file);
  }

  return { file, type: 'unknown', confidence: 0.0 };
}
```

## UI System

### Component Architecture
Built on **shadcn/ui** components for consistency:

```typescript
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';

// Use components with Tailwind classes
<Button variant="outline" size="lg" className="custom-styles">
  Action
</Button>
```

### Theming
Uses CSS variables for theme consistency:

```css
:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  --primary: 222.2 47.4% 11.2%;
  /* ... */
}
```

### Responsive Design
Mobile-first approach with Tailwind breakpoints:

```typescript
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
  {/* Responsive grid */}
</div>
```

## State Management

### Component State Pattern
```typescript
const VideoAnnotationViewer = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [annotationData, setAnnotationData] = useState<StandardAnnotationData | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [overlaySettings, setOverlaySettings] = useState<OverlaySettings>({
    pose: true,
    subtitles: true,
    speakers: true,
    scenes: true,
  });

  // Callbacks for data flow
  const handleTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);
};
```

### Data Lookup Optimization
Time-based data lookup is optimized for real-time playback:

```typescript
// Efficient timestamp-based lookup
function getDataAtTime<T extends { timestamp: number }>(
  data: T[],
  currentTime: number,
  tolerance = 0.1
): T[] {
  return data.filter(item => 
    Math.abs(item.timestamp - currentTime) <= tolerance
  );
}
```

## Testing

### Demo Data System
Built-in demo data for testing:

```typescript
// Access demo utilities in browser console
window.debugUtils.loadDemoAnnotations('peekaboo-rep3');
window.version.logVersionInfo();
```

### Manual Testing Checklist
- [ ] File upload (single and multiple files)
- [ ] Video playback with overlays
- [ ] Timeline interaction and seeking
- [ ] Overlay toggle functionality
- [ ] Error handling for invalid files

## Build and Deployment

### Production Build
```bash
bun run build
# Output in dist/ folder

# Preview build locally
bun run preview
```

### Build Optimization
- **Code splitting**: Automatic by Vite
- **Asset optimization**: Images and fonts optimized
- **Tree shaking**: Unused code removed
- **Bundle analysis**: Use `bunx vite-bundle-analyzer`

### Deployment Options
- **Static hosting**: Netlify, Vercel, GitHub Pages
- **CDN**: Any static CDN
- **Self-hosted**: Nginx, Apache, or similar

## Extension Points

### Adding New Annotation Types
1. Define types in `annotations.ts`
2. Create parser in `lib/parsers/`
3. Add detection logic in `merger.ts`
4. Update overlay rendering in `VideoPlayer.tsx`
5. Add timeline track in `Timeline.tsx`

### Custom Timeline Tracks
```typescript
// In Timeline.tsx
const renderCustomTrack = (data: CustomData[]) => {
  return data.map(item => (
    <div
      key={item.id}
      className="absolute bg-custom-color"
      style={{
        left: `${(item.startTime / duration) * 100}%`,
        width: `${((item.endTime - item.startTime) / duration) * 100}%`,
      }}
    >
      {item.label}
    </div>
  ));
};
```

### Custom Validation
```typescript
// Add to validation.ts
export const CustomFormatSchema = z.object({
  // Define your schema
});

export function validateCustomFormat(data: unknown): CustomFormatData {
  return CustomFormatSchema.parse(data);
}
```

## Troubleshooting

### Common Issues

**Build Errors**:
```bash
# Clear Vite cache
rm -rf node_modules/.vite

# Reinstall dependencies
bun install
```

**Type Errors**:
```bash
# Check TypeScript
bunx tsc --noEmit
```

**Performance Issues**:
- Large files: Implement chunked loading
- Memory usage: Use Web Workers for parsing
- Rendering: Optimize canvas drawing operations

### Debug Tools

**Browser Console**:
```javascript
// Version info
window.version.getAppTitle();

// Demo data
window.debugUtils.DEMO_DATA_SETS;

// Load demo
await window.debugUtils.loadDemoAnnotations('peekaboo-rep3');
```

**Development Server**:
- Hot module reloading enabled
- Source maps for debugging
- Error overlay in browser

## Contributing

### Code Style
- **TypeScript**: Strict mode enabled
- **ESLint**: Configured with React rules
- **Prettier**: Not configured (use editor settings)
- **Naming**: PascalCase for components, camelCase for functions

### Git Workflow
```bash
# Feature development
git checkout -b feature/your-feature
git commit -m "feat: add new feature"
git push origin feature/your-feature
```

### Pull Request Process
1. Ensure all tests pass
2. Update documentation if needed
3. Add demo data if adding new parsers
4. Follow semantic commit messages

For questions or support, check the [GitHub repository](https://github.com/InfantLab/video-annotation-viewer) or contact the development team.
