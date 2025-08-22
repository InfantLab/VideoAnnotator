# OpenFace3 Integration Planning Document - OUTLINE

## 1. OpenFace3 Data Analysis Summary

### Data Structure Overview
The OpenFace3 pipeline produces highly complex, multi-layered facial analysis data:

```json
{
  "metadata": {
    "pipeline": "OpenFace3Pipeline",
    "model_info": { "features": { /* 6 feature flags */ } },
    "config": { /* per-feature configuration */ },
    "processing_stats": { "total_faces": 19, "avg_processing_time": 2.73 }
  },
  "faces": [
    {
      "annotation_id": 1,
      "bbox": [389, 211, 133, 158],
      "timestamp": 0.0,
      "features": {
        "confidence": 0.992,
        "landmarks_2d": [ /* 98 x,y coordinates */ ],
        "action_units": { /* 8 AU measurements */ },
        "head_pose": { "pitch": -12.38, "yaw": -0.05, "roll": 0.0 },
        "gaze": { "direction_x": -0.001, "direction_y": 0.214, "direction_z": 0.977 },
        "emotion": { "dominant": "disgust", "probabilities": { /* 8 emotions */ } }
      }
    }
  ]
}
```

### Feature Complexity Analysis
1. **Face Detection**: Simple bounding boxes (4 coordinates)
2. **2D Landmarks**: 98 facial keypoints (196 coordinates) - HIGHEST DENSITY
3. **Action Units**: 8 facial muscle measurements with intensity/presence
4. **Head Pose**: 3D orientation (pitch, yaw, roll) + confidence
5. **Gaze Direction**: 3D vector (x, y, z) + confidence  
6. **Emotions**: 8 emotion probabilities + valence/arousal + dominant

### Data Volume Considerations
- **Multiple faces per frame**: Up to 5 faces simultaneously
- **High temporal resolution**: Every ~1 second in demo data
- **Memory footprint**: ~500KB per face annotation (landmarks dominate)
- **Rendering complexity**: 98 landmarks = 196 canvas operations per face

## 2. Current Architecture Assessment

### Existing Overlay System Strengths
The v0.2.0 overlay system provides an excellent foundation:

**OverlaySettings Interface** (currently flat):
```typescript
export interface OverlaySettings {
  pose: boolean;           // COCO person tracking
  subtitles: boolean;      // WebVTT speech
  speakers: boolean;       // RTTM diarization
  scenes: boolean;         // Scene detection
  faces: boolean;          // LAION face detection
  emotions: boolean;       // Basic emotion overlays
}
```

**Control Grouping System** (from `OverlayControls.tsx`):
```typescript
const overlayGroups = [
  {
    title: '😊 Face Analysis',
    color: 'hsl(142, 76%, 36%)', // Green theme
    items: [
      { key: 'faces', label: 'Face Detection Boxes' },
      { key: 'emotions', label: 'Emotion Recognition' }
    ]
  }
];
```

### Current Limitations for OpenFace3
1. **Flat Structure**: No support for nested/hierarchical controls
2. **Single Emotion Type**: Only basic emotion overlay, no fine-grained control
3. **No Landmarks Support**: Missing 2D/3D facial keypoint rendering
4. **No Action Units**: Missing facial muscle movement analysis
5. **No Head Pose/Gaze**: Missing 3D orientation data visualization

### Architecture Strengths to Leverage
- ✅ **Canvas Overlay System**: Efficient, scalable rendering foundation
- ✅ **Time-based Data Filtering**: Proven system for temporal queries
- ✅ **Color-coded Groups**: Visual organization system ready for expansion
- ✅ **JSON Viewer Integration**: Per-feature data inspection capability
- ✅ **Toggle All Functionality**: Bulk enable/disable patterns established
- ✅ **Timeline Synchronization**: Lock-to-overlay system for consistency

## 3. Proposed Control Hierarchy

### Master Control Design
```
😊 Face Analysis                                    [EXPANDED ▼] [ALL ON] [📄]
├── 🔍 Face Detection (LAION)      [●] [📄]        # Existing
├── 🎭 Basic Emotions (LAION)      [●] [📄]        # Existing  
├── 🧬 OpenFace3 Analysis          [●] [📄] [MASTER TOGGLE]
│   ├── � Face Bounding Boxes     [●] [📄]
│   ├── 📍 2D Landmarks (98pts)    [○] [📄]        # Performance: Default OFF
│   ├── 💪 Action Units (8 AUs)    [○] [📄]        # Performance: Default OFF  
│   ├── 🧭 Head Pose (3D)          [●] [📄]
│   ├── 👁️  Gaze Direction         [●] [📄]
│   └── 😀 Emotions (Enhanced)     [●] [📄]
```

### Control Interaction Logic

#### Master Toggle Behavior
- **OpenFace3 Master ON**: 
  - Enables all sub-controls
  - Shows aggregate face count in timeline
  - Displays "OpenFace3 Active: X faces" status
  
- **OpenFace3 Master OFF**:
  - Disables all sub-controls (grayed out)
  - Hides all OpenFace3 overlays instantly
  - Preserves individual sub-control states for when re-enabled

#### Sub-Control Independence
- **Individual toggles work independently** when master is ON
- **Smart performance defaults**: Heavy features (landmarks, action units) start OFF
- **Cascade disable**: If all sub-controls OFF, master shows indeterminate state
- **JSON viewers**: Each sub-control has dedicated data inspection

### Visual Hierarchy Implementation
```typescript
// Enhanced OverlaySettings structure
export interface OverlaySettings {
  // Existing flat controls
  pose: boolean;
  subtitles: boolean;
  speakers: boolean;
  scenes: boolean;
  
  // Enhanced face analysis with hierarchy
  faces: boolean;              // LAION face detection (existing)
  emotions: boolean;           // LAION emotions (existing)
  
  // NEW: OpenFace3 nested controls
  openface3: {
    enabled: boolean;          // Master toggle
    faceBoxes: boolean;        // Face detection boxes
    landmarks2d: boolean;      // 98 facial landmarks
    actionUnits: boolean;      // Facial muscle analysis
    headPose: boolean;         // 3D head orientation
    gaze: boolean;             // Gaze direction vectors
    emotionsEnhanced: boolean; // 8-category emotion analysis
  };
}
```

### UI Component Structure
```tsx
// OverlayControls.tsx - Enhanced grouping
const overlayGroups = [
  {
    title: '😊 Face Analysis',
    color: 'hsl(142, 76%, 36%)',
    items: [
      { key: 'faces', label: 'Face Detection (LAION)', type: 'simple' },
      { key: 'emotions', label: 'Basic Emotions (LAION)', type: 'simple' },
      { 
        key: 'openface3', 
        label: 'OpenFace3 Analysis',
        type: 'hierarchical',
        children: [
          { key: 'faceBoxes', label: 'Face Bounding Boxes', default: true },
          { key: 'landmarks2d', label: '2D Landmarks (98pts)', default: false },
          { key: 'actionUnits', label: 'Action Units (8 AUs)', default: false },
          { key: 'headPose', label: 'Head Pose (3D)', default: true },
          { key: 'gaze', label: 'Gaze Direction', default: true },
          { key: 'emotionsEnhanced', label: 'Emotions (Enhanced)', default: true }
        ]
      }
    ]
  }
];
```

## 4. Technical Implementation Strategy

### Phase 1: Data Types & Parsing (Days 1-2)

#### Extended Type Definitions
```typescript
// types/annotations.ts - New OpenFace3 interfaces
export interface OpenFace3Landmark {
  x: number;
  y: number;
}

export interface OpenFace3ActionUnit {
  intensity: number;
  presence: boolean;
}

export interface OpenFace3ActionUnits {
  AU01_Inner_Brow_Raiser: OpenFace3ActionUnit;
  AU02_Outer_Brow_Raiser: OpenFace3ActionUnit;
  AU04_Brow_Lowerer: OpenFace3ActionUnit;
  AU05_Upper_Lid_Raiser: OpenFace3ActionUnit;
  AU06_Cheek_Raiser: OpenFace3ActionUnit;
  AU07_Lid_Tightener: OpenFace3ActionUnit;
  AU09_Nose_Wrinkler: OpenFace3ActionUnit;
  AU10_Upper_Lip_Raiser: OpenFace3ActionUnit;
}

export interface OpenFace3HeadPose {
  pitch: number;    // degrees
  yaw: number;      // degrees  
  roll: number;     // degrees
  confidence: number;
}

export interface OpenFace3Gaze {
  direction_x: number;
  direction_y: number;
  direction_z: number;
  confidence: number;
}

export interface OpenFace3Emotion {
  dominant: string;
  probabilities: {
    neutral: number;
    happiness: number;
    sadness: number;
    anger: number;
    fear: number;
    surprise: number;
    disgust: number;
    contempt: number;
  };
  valence: number;   // -1 to 1
  arousal: number;   // -1 to 1  
  confidence: number;
}

export interface OpenFace3FaceAnnotation {
  annotation_id: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
  timestamp: number;
  features: {
    confidence: number;
    landmarks_2d: OpenFace3Landmark[];     // 98 landmarks
    action_units: OpenFace3ActionUnits;
    head_pose: OpenFace3HeadPose;
    gaze: OpenFace3Gaze;
    emotion: OpenFace3Emotion;
  };
}

export interface OpenFace3Data {
  metadata: {
    pipeline: string;
    model_info: any;
    config: any;
    processing_stats: {
      total_faces: number;
      avg_processing_time: number;
    };
  };
  faces: OpenFace3FaceAnnotation[];
}
```

#### Parser Functions
```typescript
// lib/parsers/openface3.ts - New parser module
export const getOpenFace3FacesAtTime = (
  data: OpenFace3FaceAnnotation[], 
  currentTime: number, 
  timeWindow: number = 0.1
): OpenFace3FaceAnnotation[] => {
  return data.filter(face => 
    Math.abs(face.timestamp - currentTime) <= timeWindow
  );
};

export const getActionUnitsIntensity = (face: OpenFace3FaceAnnotation): Array<{
  name: string;
  intensity: number;
  active: boolean;
}> => {
  return Object.entries(face.features.action_units).map(([key, value]) => ({
    name: key.replace('AU', '').replace(/_/g, ' '),
    intensity: value.intensity,
    active: value.presence
  }));
};

export const getGazeVector = (face: OpenFace3FaceAnnotation): {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  confidence: number;
} => {
  const bbox = face.bbox;
  const centerX = bbox[0] + bbox[2] / 2;
  const centerY = bbox[1] + bbox[3] / 2;
  
  const gaze = face.features.gaze;
  const vectorLength = 100; // pixels
  
  return {
    startX: centerX,
    startY: centerY,
    endX: centerX + (gaze.direction_x * vectorLength),
    endY: centerY + (gaze.direction_y * vectorLength),
    confidence: gaze.confidence
  };
};
```

#### Integration with StandardAnnotationData
```typescript
// Extend existing StandardAnnotationData interface
export interface StandardAnnotationData {
  // ... existing fields ...
  openface3_analysis?: OpenFace3FaceAnnotation[];  // NEW field
}
```

### Phase 2: Enhanced Control System (Days 3-4)

#### Enhanced OverlaySettings
```typescript
// Update OverlaySettings to support nested controls
export interface OpenFace3Settings {
  enabled: boolean;
  faceBoxes: boolean;
  landmarks2d: boolean;
  actionUnits: boolean;
  headPose: boolean;
  gaze: boolean;
  emotionsEnhanced: boolean;
}

export interface OverlaySettings {
  // Existing controls
  pose: boolean;
  subtitles: boolean;
  speakers: boolean;
  scenes: boolean;
  faces: boolean;
  emotions: boolean;
  
  // NEW: Nested OpenFace3 controls
  openface3: OpenFace3Settings;
}
```

#### OverlayControls Component Enhancement
```tsx
// components/OverlayControls.tsx - Add hierarchical rendering
const renderHierarchicalControl = (item: HierarchicalItem) => {
  const [expanded, setExpanded] = useState(true);
  
  return (
    <div className="space-y-2">
      {/* Master Control */}
      <div className="flex items-center justify-between p-2 bg-muted rounded">
        <div className="flex items-center space-x-2">
          <Button 
            size="sm" 
            variant="ghost"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? '▼' : '▶'}
          </Button>
          <Label>{item.label}</Label>
        </div>
        <div className="flex items-center space-x-2">
          <Switch 
            checked={settings.openface3.enabled}
            onCheckedChange={(checked) => handleNestedToggle('openface3', 'enabled', checked)}
          />
          <Button size="sm" variant="outline">📄</Button>
        </div>
      </div>
      
      {/* Child Controls */}
      {expanded && settings.openface3.enabled && (
        <div className="ml-6 space-y-2 border-l-2 border-border pl-4">
          {item.children?.map(child => (
            <div key={child.key} className="flex items-center justify-between">
              <Label className="text-sm">{child.label}</Label>
              <div className="flex items-center space-x-2">
                <Switch 
                  checked={settings.openface3[child.key as keyof OpenFace3Settings]}
                  onCheckedChange={(checked) => handleNestedToggle('openface3', child.key, checked)}
                />
                <Button size="sm" variant="outline">📄</Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
```

### Phase 3: Overlay Rendering (Days 5-7)

#### VideoPlayer.tsx - New Rendering Functions
```tsx
// Add 6 new drawing functions to VideoPlayer.tsx
const drawOpenFace3FaceBoxes = useCallback((ctx: CanvasRenderingContext2D) => {
  if (!overlaySettings.openface3.enabled || !overlaySettings.openface3.faceBoxes) return;
  
  const faces = getOpenFace3FacesAtTime(annotationData.openface3_analysis || [], currentTime);
  
  faces.forEach(face => {
    const [x, y, width, height] = face.bbox;
    const hue = (face.annotation_id * 137.508) % 360;
    
    ctx.strokeStyle = `hsl(${hue}, 70%, 60%)`;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    
    // Confidence indicator
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(x, y - 20, 80, 18);
    ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
    ctx.font = '12px monospace';
    ctx.fillText(`OF3:${face.annotation_id} (${(face.features.confidence * 100).toFixed(0)}%)`, x + 2, y - 6);
  });
}, [overlaySettings.openface3, currentTime, annotationData]);

const drawOpenFace3Landmarks = useCallback((ctx: CanvasRenderingContext2D) => {
  if (!overlaySettings.openface3.enabled || !overlaySettings.openface3.landmarks2d) return;
  
  const faces = getOpenFace3FacesAtTime(annotationData.openface3_analysis || [], currentTime);
  
  faces.forEach(face => {
    const hue = (face.annotation_id * 137.508) % 360;
    ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
    
    face.features.landmarks_2d.forEach((landmark, index) => {
      ctx.beginPath();
      ctx.arc(landmark.x, landmark.y, 1.5, 0, 2 * Math.PI);
      ctx.fill();
    });
  });
}, [overlaySettings.openface3, currentTime, annotationData]);

const drawOpenFace3ActionUnits = useCallback((ctx: CanvasRenderingContext2D) => {
  if (!overlaySettings.openface3.enabled || !overlaySettings.openface3.actionUnits) return;
  
  const faces = getOpenFace3FacesAtTime(annotationData.openface3_analysis || [], currentTime);
  
  faces.forEach(face => {
    const [x, y, width, height] = face.bbox;
    const activeAUs = getActionUnitsIntensity(face).filter(au => au.active);
    
    activeAUs.forEach((au, index) => {
      const labelY = y + height + 30 + (index * 20);
      const intensity = Math.min(au.intensity, 5); // Cap at 5 for display
      const alpha = 0.4 + (intensity / 5) * 0.6; // Variable opacity based on intensity
      
      ctx.fillStyle = `rgba(255, 165, 0, ${alpha})`;
      ctx.fillRect(x, labelY - 15, 120, 18);
      
      ctx.fillStyle = 'white';
      ctx.font = '11px monospace';
      ctx.fillText(`${au.name}: ${intensity.toFixed(1)}`, x + 2, labelY - 3);
    });
  });
}, [overlaySettings.openface3, currentTime, annotationData]);

const drawOpenFace3HeadPose = useCallback((ctx: CanvasRenderingContext2D) => {
  if (!overlaySettings.openface3.enabled || !overlaySettings.openface3.headPose) return;
  
  const faces = getOpenFace3FacesAtTime(annotationData.openface3_analysis || [], currentTime);
  
  faces.forEach(face => {
    const [x, y, width, height] = face.bbox;
    const centerX = x + width / 2;
    const centerY = y + height / 2;
    const pose = face.features.head_pose;
    
    // Draw coordinate system based on head pose
    const lineLength = 50;
    
    // Pitch (red line - up/down)
    ctx.strokeStyle = `rgba(255, 0, 0, ${pose.confidence})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX, centerY + Math.sin(pose.pitch * Math.PI / 180) * lineLength);
    ctx.stroke();
    
    // Yaw (green line - left/right)  
    ctx.strokeStyle = `rgba(0, 255, 0, ${pose.confidence})`;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + Math.sin(pose.yaw * Math.PI / 180) * lineLength, centerY);
    ctx.stroke();
    
    // Roll indicator (blue arc)
    ctx.strokeStyle = `rgba(0, 0, 255, ${pose.confidence})`;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 30, 0, pose.roll * Math.PI / 180);
    ctx.stroke();
  });
}, [overlaySettings.openface3, currentTime, annotationData]);

const drawOpenFace3Gaze = useCallback((ctx: CanvasRenderingContext2D) => {
  if (!overlaySettings.openface3.enabled || !overlaySettings.openface3.gaze) return;
  
  const faces = getOpenFace3FacesAtTime(annotationData.openface3_analysis || [], currentTime);
  
  faces.forEach(face => {
    const gazeVector = getGazeVector(face);
    
    ctx.strokeStyle = `rgba(255, 255, 0, ${gazeVector.confidence})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(gazeVector.startX, gazeVector.startY);
    ctx.lineTo(gazeVector.endX, gazeVector.endY);
    ctx.stroke();
    
    // Arrow head
    const angle = Math.atan2(gazeVector.endY - gazeVector.startY, gazeVector.endX - gazeVector.startX);
    ctx.beginPath();
    ctx.moveTo(gazeVector.endX, gazeVector.endY);
    ctx.lineTo(gazeVector.endX - 10 * Math.cos(angle - Math.PI/6), gazeVector.endY - 10 * Math.sin(angle - Math.PI/6));
    ctx.moveTo(gazeVector.endX, gazeVector.endY);
    ctx.lineTo(gazeVector.endX - 10 * Math.cos(angle + Math.PI/6), gazeVector.endY - 10 * Math.sin(angle + Math.PI/6));
    ctx.stroke();
  });
}, [overlaySettings.openface3, currentTime, annotationData]);

const drawOpenFace3Emotions = useCallback((ctx: CanvasRenderingContext2D) => {
  if (!overlaySettings.openface3.enabled || !overlaySettings.openface3.emotionsEnhanced) return;
  
  const faces = getOpenFace3FacesAtTime(annotationData.openface3_analysis || [], currentTime);
  
  faces.forEach(face => {
    const [x, y, width, height] = face.bbox;
    const emotion = face.features.emotion;
    
    // Show dominant emotion with probability
    const emotionText = `${emotion.dominant}: ${(emotion.confidence * 100).toFixed(1)}%`;
    const valenceText = `V:${emotion.valence.toFixed(2)} A:${emotion.arousal.toFixed(2)}`;
    
    // Background for emotion text
    ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    const textWidth = Math.max(ctx.measureText(emotionText).width, ctx.measureText(valenceText).width);
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(x, y + height + 5, textWidth + 8, 35);
    
    // Emotion colors based on valence
    const emotionColor = emotion.valence > 0 ? 'hsl(120, 70%, 60%)' : 'hsl(0, 70%, 60%)';
    ctx.fillStyle = emotionColor;
    ctx.fillText(emotionText, x + 4, y + height + 20);
    ctx.fillText(valenceText, x + 4, y + height + 35);
  });
}, [overlaySettings.openface3, currentTime, annotationData]);
```

### Phase 4: Timeline Integration (Days 8-9)

#### Timeline Component Updates
```tsx
// components/Timeline.tsx - Add OpenFace3 track support
const drawOpenFace3Track = (ctx: CanvasRenderingContext2D, data: OpenFace3FaceAnnotation[]) => {
  if (!timelineSettings.showOpenFace3) return;
  
  // Group faces by timestamp for density visualization
  const facesByTime = data.reduce((acc, face) => {
    const timeKey = Math.floor(face.timestamp);
    acc[timeKey] = (acc[timeKey] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);
  
  // Draw density bars
  Object.entries(facesByTime).forEach(([time, count]) => {
    const x = parseFloat(time) * pixelsPerSecond;
    const height = Math.min(count * 5, 30); // Max height 30px
    
    ctx.fillStyle = 'hsl(142, 76%, 36%)'; // Green theme
    ctx.fillRect(x, trackY, 2, height);
  });
};
```

#### TimelineSettings Enhancement
```typescript
export interface TimelineSettings {
  // Existing settings
  showSubtitles: boolean;
  showSpeakers: boolean;
  showScenes: boolean;
  showMotion: boolean;
  showFaces: boolean;
  showEmotions: boolean;
  
  // NEW: OpenFace3 timeline track
  showOpenFace3: boolean;
}
```

## 5. User Experience Design

### Master Toggle Behavior

#### Visual States
```
[●] 🧬 OpenFace3 Analysis (6 active)     [ALL ON] [📄]    # All sub-features enabled
[◐] 🧬 OpenFace3 Analysis (3 active)     [SOME]   [📄]    # Some sub-features enabled  
[○] 🧬 OpenFace3 Analysis (0 active)     [ALL OFF] [📄]   # All sub-features disabled
[⊗] 🧬 OpenFace3 Analysis (no data)      [DISABLED] [📄]  # No OpenFace3 data available
```

#### Interaction Patterns
- **Single Click Master Toggle**: 
  - If any sub-feature is ON → Turn ALL OFF
  - If all sub-features are OFF → Turn ON smart defaults (faceBoxes, headPose, gaze, emotions)
  - Preserves individual preferences for re-enable

- **ALL ON/ALL OFF Button**:
  - ALL ON: Enables all 6 sub-features (including performance-heavy landmarks & action units)
  - ALL OFF: Disables all sub-features, sets master to OFF state
  - Independent of master toggle state preservation

### Sub-Control Features & Smart Defaults

#### Performance-Aware Defaults
```typescript
const OPENFACE3_SMART_DEFAULTS: OpenFace3Settings = {
  enabled: true,
  faceBoxes: true,        // ✅ Lightweight, always useful
  landmarks2d: false,     // ❌ Heavy rendering (98 points × faces)
  actionUnits: false,     // ❌ Complex overlay, specialized use
  headPose: true,         // ✅ Useful 3D orientation info
  gaze: true,             // ✅ Intuitive attention visualization
  emotionsEnhanced: true  // ✅ Enhanced emotion analysis valuable
};
```

#### Sub-Control Descriptions
```
📦 Face Bounding Boxes     [●] [📄]  
   "OpenFace3 face detection regions with confidence scores"

📍 2D Landmarks (98pts)    [○] [📄]  ⚠️ Performance Impact
   "Detailed facial keypoints (98 landmarks per face)"

💪 Action Units (8 AUs)    [○] [📄]  ⚠️ Specialized Analysis  
   "Facial muscle movements (AU01-AU10 intensity analysis)"

🧭 Head Pose (3D)          [●] [📄]
   "3D head orientation (pitch, yaw, roll with confidence)"

👁️ Gaze Direction         [●] [📄]
   "Eye gaze direction vectors with confidence indicators"

😀 Emotions (Enhanced)     [●] [📄]
   "8-category emotion analysis with valence/arousal metrics"
```

### JSON Viewer Integration

#### Per-Feature Data Inspection
Each sub-control has a dedicated JSON viewer showing filtered data:

- **📦 Face Boxes**: Shows bbox coordinates, confidence, annotation_id
- **📍 Landmarks**: Shows all 98 x,y coordinates for current timestamp  
- **💪 Action Units**: Shows intensity/presence for all 8 AUs
- **🧭 Head Pose**: Shows pitch/yaw/roll angles with confidence
- **👁️ Gaze**: Shows 3D direction vector with confidence
- **😀 Emotions**: Shows all 8 emotion probabilities + valence/arousal

#### Master JSON Viewer
Shows complete OpenFace3 data structure including metadata and processing stats.

### Progressive Disclosure Design

#### Collapsed State (Default)
```
😊 Face Analysis                           [EXPANDED ▼] [ALL ON] [📄]
├── 🔍 Face Detection (LAION)     [●] [📄]
├── 🎭 Basic Emotions (LAION)     [●] [📄]  
└── 🧬 OpenFace3 Analysis (3)     [●] [ALL ON] [📄]
```

#### Expanded State (On User Click)
```
😊 Face Analysis                           [EXPANDED ▼] [ALL ON] [📄]
├── 🔍 Face Detection (LAION)     [●] [📄]
├── 🎭 Basic Emotions (LAION)     [●] [📄]  
├── 🧬 OpenFace3 Analysis         [●] [ALL ON] [📄]
│   ├── 📦 Face Bounding Boxes    [●] [📄]
│   ├── 📍 2D Landmarks (98pts)   [○] [📄] ⚠️ Performance Impact
│   ├── 💪 Action Units (8 AUs)   [○] [📄] ⚠️ Specialized
│   ├── 🧭 Head Pose (3D)         [●] [📄]
│   ├── 👁️ Gaze Direction        [●] [📄]
│   └── 😀 Emotions (Enhanced)    [●] [📄]
```

### Error States & Data Availability

#### No OpenFace3 Data Available
```
🧬 OpenFace3 Analysis             [⊗] [DISABLED] [📄]
   "No OpenFace3 data found in current annotation set"
   "Upload video processed with OpenFace3 pipeline"
```

#### Partial Data Available
```
🧬 OpenFace3 Analysis (partial)   [◐] [SOME] [📄]
├── 📦 Face Boxes Available       [●] [📄]
├── 📍 Landmarks Not Available    [⊗] [📄] ❌ Feature disabled in pipeline
├── 💪 Action Units Available     [●] [📄] 
├── 🧭 Head Pose Available        [●] [📄]
├── 👁️ Gaze Not Available        [⊗] [📄] ❌ Feature disabled in pipeline  
└── 😀 Emotions Available         [●] [📄]
```

### Accessibility & Keyboard Navigation

#### Keyboard Shortcuts
- **Space**: Toggle master OpenFace3 control
- **Shift+1**: Toggle Face Boxes
- **Shift+2**: Toggle 2D Landmarks  
- **Shift+3**: Toggle Action Units
- **Shift+4**: Toggle Head Pose
- **Shift+5**: Toggle Gaze Direction
- **Shift+6**: Toggle Enhanced Emotions

#### Screen Reader Support
- **ARIA labels**: Descriptive labels for all controls
- **State announcements**: "OpenFace3 Analysis enabled, 3 of 6 features active"
- **Nested structure**: Proper heading hierarchy for sub-controls

## 6. Performance Considerations

### Data Volume Analysis

#### Memory Footprint per Face Annotation
```
Face Detection Box:    16 bytes   (4 × 4-byte coordinates)
2D Landmarks:         784 bytes   (98 × 8-byte coordinates)  🔴 HEAVY
Action Units:         144 bytes   (8 × 18-byte AU objects)
Head Pose:             28 bytes   (3 angles + confidence)
Gaze Direction:        28 bytes   (3D vector + confidence)  
Emotions:             256 bytes   (8 probabilities + metadata)
---------------------------------------------------------
Total per face:      1,256 bytes
```

#### Scaling Projections
- **Single face, 30fps video**: ~37KB/second
- **5 faces, 30fps video**: ~185KB/second  
- **10-minute video, 5 faces**: ~111MB total
- **98 landmarks rendering**: 490 canvas operations per face per frame

### Rendering Optimization Strategy

#### Smart Visibility Culling
```typescript
// Only render features that are actually visible
const optimizedRender = () => {
  const faces = getOpenFace3FacesAtTime(data, currentTime);
  
  // Pre-filter faces outside viewport
  const visibleFaces = faces.filter(face => {
    const [x, y, width, height] = face.bbox;
    return (x + width > 0 && x < canvasWidth && 
            y + height > 0 && y < canvasHeight);
  });
  
  // Render only enabled features for visible faces
  if (settings.faceBoxes) drawFaceBoxes(visibleFaces);
  if (settings.landmarks2d && visibleFaces.length <= 3) { // Limit for performance
    drawLandmarks(visibleFaces);
  }
  // ... other features
};
```

#### Performance-Aware Feature Limits
```typescript
const PERFORMANCE_LIMITS = {
  landmarks2d: {
    maxFaces: 3,           // Limit landmark rendering to 3 faces max
    maxZoomOut: 0.5,       // Disable landmarks when zoomed out
    minFaceSize: 50        // Don't render for small faces
  },
  actionUnits: {
    maxFaces: 5,           // Action units for up to 5 faces
    updateThreshold: 0.1   // Only update if intensity changes > 0.1
  },
  gaze: {
    minConfidence: 0.3     // Only show gaze vectors above 30% confidence
  }
};
```

#### Canvas Optimization Techniques
```typescript
// Batch canvas operations for efficiency
const batchedLandmarkRender = (faces: OpenFace3FaceAnnotation[]) => {
  const ctx = canvasRef.current?.getContext('2d');
  if (!ctx) return;
  
  // Single beginPath() for all landmarks
  ctx.beginPath();
  
  faces.forEach(face => {
    face.features.landmarks_2d.forEach(landmark => {
      ctx.moveTo(landmark.x + 1.5, landmark.y);
      ctx.arc(landmark.x, landmark.y, 1.5, 0, 2 * Math.PI);
    });
  });
  
  // Single fill() operation
  ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
  ctx.fill();
};
```

### Memory Management

#### Lazy Loading Strategy
```typescript
// Only load OpenFace3 data when first requested
const useOpenFace3Data = (annotationData: StandardAnnotationData) => {
  const [openface3Data, setOpenface3Data] = useState<OpenFace3FaceAnnotation[] | null>(null);
  const [loading, setLoading] = useState(false);
  
  const loadOpenFace3Data = useCallback(async () => {
    if (openface3Data || loading) return;
    
    setLoading(true);
    try {
      // Load and parse OpenFace3 data on demand
      const data = await parseOpenFace3File(annotationData.openface3_file_path);
      setOpenface3Data(data.faces);
    } catch (error) {
      console.error('Failed to load OpenFace3 data:', error);
    } finally {
      setLoading(false);
    }
  }, [annotationData.openface3_file_path, openface3Data, loading]);
  
  return { openface3Data, loadOpenFace3Data, loading };
};
```

#### Data Preprocessing for Performance
```typescript
// Pre-compute frame indices for faster temporal queries
const preprocessOpenFace3Data = (faces: OpenFace3FaceAnnotation[]) => {
  const frameIndex = new Map<number, OpenFace3FaceAnnotation[]>();
  
  faces.forEach(face => {
    const frameKey = Math.round(face.timestamp * 30); // Assume 30fps
    if (!frameIndex.has(frameKey)) {
      frameIndex.set(frameKey, []);
    }
    frameIndex.get(frameKey)!.push(face);
  });
  
  return frameIndex;
};
```

### Real-time Performance Monitoring

#### Performance Metrics Collection
```typescript
const useRenderingPerformance = () => {
  const [metrics, setMetrics] = useState({
    avgFrameTime: 0,
    droppedFrames: 0,
    activeFaces: 0,
    activeFeatures: 0
  });
  
  const measureRenderTime = useCallback((renderFn: () => void) => {
    const startTime = performance.now();
    renderFn();
    const endTime = performance.now();
    
    const frameTime = endTime - startTime;
    setMetrics(prev => ({
      ...prev,
      avgFrameTime: (prev.avgFrameTime * 0.9) + (frameTime * 0.1), // Moving average
      droppedFrames: frameTime > 16.67 ? prev.droppedFrames + 1 : prev.droppedFrames
    }));
  }, []);
  
  return { metrics, measureRenderTime };
};
```

#### Adaptive Quality Control
```typescript
// Automatically disable expensive features if performance drops
const useAdaptiveQuality = (metrics: PerformanceMetrics) => {
  const [qualityLevel, setQualityLevel] = useState<'high' | 'medium' | 'low'>('high');
  
  useEffect(() => {
    if (metrics.avgFrameTime > 25) { // > 25ms = below 40fps
      setQualityLevel('low');
    } else if (metrics.avgFrameTime > 20) { // > 20ms = below 50fps
      setQualityLevel('medium');
    } else {
      setQualityLevel('high');
    }
  }, [metrics.avgFrameTime]);
  
  const getAdaptiveSettings = (baseSettings: OpenFace3Settings): OpenFace3Settings => {
    switch (qualityLevel) {
      case 'low':
        return { ...baseSettings, landmarks2d: false, actionUnits: false };
      case 'medium':
        return { ...baseSettings, landmarks2d: false };
      default:
        return baseSettings;
    }
  };
  
  return { qualityLevel, getAdaptiveSettings };
};
```

### Debugging & Profiling Tools

#### Performance Debug Panel
```typescript
// Add to DebugPanel.tsx for OpenFace3 performance monitoring
const OpenFace3PerformancePanel = ({ metrics }: { metrics: PerformanceMetrics }) => (
  <Card className="mt-4">
    <CardHeader>
      <CardTitle className="text-sm">🧬 OpenFace3 Performance</CardTitle>
    </CardHeader>
    <CardContent className="space-y-2 text-xs">
      <div>Avg Frame Time: {metrics.avgFrameTime.toFixed(2)}ms</div>
      <div>Dropped Frames: {metrics.droppedFrames}</div>
      <div>Active Faces: {metrics.activeFaces}</div>
      <div>Active Features: {metrics.activeFeatures}/6</div>
      <div className={`font-bold ${metrics.avgFrameTime > 20 ? 'text-red-500' : 'text-green-500'}`}>
        Quality: {metrics.avgFrameTime > 25 ? 'LOW' : metrics.avgFrameTime > 20 ? 'MEDIUM' : 'HIGH'}
      </div>
    </CardContent>
  </Card>
);
```

## 7. Integration Points

### File Detection & Auto-Discovery

#### OpenFace3 File Naming Convention
Based on the demo files, OpenFace3 produces files with this pattern:
```
{video_base_name}_openface3_analysis.json     # Summary format
{video_base_name}_openface3_detailed.json     # Full format (recommended)
```

#### Enhanced File Detection Logic
```typescript
// lib/fileUtils.ts - Extend existing detection system
export const detectOpenFace3Files = async (videoFile: File): Promise<{
  analysis?: File;
  detailed?: File;
} | null> => {
  const videoBaseName = videoFile.name.replace(/\.[^/.]+$/, ''); // Remove extension
  
  // Look for both OpenFace3 file variants
  const patterns = [
    `${videoBaseName}_openface3_detailed.json`,  // Preferred - full data
    `${videoBaseName}_openface3_analysis.json`,  // Fallback - summary data
    `${videoBaseName}.openface3.json`,           // Alternative naming
  ];
  
  // Use existing directory scanning logic from VEATIC detection
  const detectedFiles = await scanForAnnotationFiles(videoFile, patterns);
  
  return {
    detailed: detectedFiles.find(f => f.name.includes('detailed')),
    analysis: detectedFiles.find(f => f.name.includes('analysis')) 
  };
};
```

#### Integration with FileUploader Component
```typescript
// components/FileUploader.tsx - Add OpenFace3 auto-detection
const handleVideoUpload = async (videoFile: File) => {
  // ... existing logic ...
  
  // Auto-detect OpenFace3 files
  const openface3Files = await detectOpenFace3Files(videoFile);
  if (openface3Files.detailed || openface3Files.analysis) {
    setAutoDetectedFiles(prev => ({
      ...prev,
      openface3: openface3Files.detailed || openface3Files.analysis
    }));
  }
};
```

### Color Coding & Visual Theme

#### Consistent Green Theme Integration
OpenFace3 features inherit the existing "😊 Face Analysis" green color scheme:

```typescript
// Extend existing color system in OverlayControls.tsx
const FACE_ANALYSIS_COLOR = 'hsl(142, 76%, 36%)'; // Existing green

const OPENFACE3_COLOR_PALETTE = {
  primary: 'hsl(142, 76%, 36%)',      // Main green (face boxes, master control)
  landmarks: 'hsl(142, 60%, 45%)',    // Slightly lighter (landmarks)
  actionUnits: 'hsl(142, 80%, 30%)',  // Darker green (action units)
  headPose: 'hsl(142, 70%, 50%)',     // Medium green (head pose)
  gaze: 'hsl(142, 85%, 40%)',         // Vibrant green (gaze vectors)
  emotions: 'hsl(142, 65%, 55%)'      // Light green (emotions)
};
```

#### Timeline Color Consistency
```typescript
// components/Timeline.tsx - OpenFace3 track styling
const drawOpenFace3Track = (ctx: CanvasRenderingContext2D) => {
  // Use same green as face analysis group
  ctx.fillStyle = FACE_ANALYSIS_COLOR;
  
  // Density visualization with opacity variation
  const density = getFaceDensityAtTime(currentTime);
  ctx.globalAlpha = 0.3 + (density / maxDensity) * 0.7;
  
  // Draw track bars
  drawTrackSegments(ctx, openface3Data);
};
```

### Lock Functionality Integration

#### Enhanced UnifiedControls Synchronization
```typescript
// components/UnifiedControls.tsx - Extend sync logic for OpenFace3
const syncTimelineWithOverlay = (overlaySettings: OverlaySettings) => {
  const syncedSettings: TimelineSettings = {
    // ... existing sync logic ...
    
    // NEW: OpenFace3 timeline sync
    showOpenFace3: overlaySettings.openface3.enabled && (
      overlaySettings.openface3.faceBoxes ||
      overlaySettings.openface3.landmarks2d ||
      overlaySettings.openface3.actionUnits ||
      overlaySettings.openface3.headPose ||
      overlaySettings.openface3.gaze ||
      overlaySettings.openface3.emotionsEnhanced
    )
  };
  
  onTimelineChange(syncedSettings);
};
```

#### Master Lock Behavior
When timeline controls are locked to overlays:
- **OpenFace3 Master Toggle** affects timeline OpenFace3 track visibility
- **Individual sub-toggles** don't affect timeline (only master controls timeline)
- **Timeline OpenFace3 toggle** reflects aggregate state of all sub-features

### Debug Panel Integration

#### Enhanced Debug Metrics
```typescript
// components/DebugPanel.tsx - Add OpenFace3 section
const DebugPanel = ({ annotationData, overlaySettings }) => {
  const openface3Stats = useMemo(() => {
    if (!annotationData.openface3_analysis) return null;
    
    const totalFaces = annotationData.openface3_analysis.length;
    const uniqueFaceIds = new Set(annotationData.openface3_analysis.map(f => f.annotation_id)).size;
    const avgConfidence = annotationData.openface3_analysis.reduce((acc, f) => acc + f.features.confidence, 0) / totalFaces;
    const timespan = Math.max(...annotationData.openface3_analysis.map(f => f.timestamp)) - 
                    Math.min(...annotationData.openface3_analysis.map(f => f.timestamp));
    
    return {
      totalFaces,
      uniqueFaceIds,
      avgConfidence,
      timespan,
      facesPerSecond: totalFaces / timespan
    };
  }, [annotationData.openface3_analysis]);
  
  return (
    <div className="space-y-4">
      {/* ... existing debug sections ... */}
      
      {openface3Stats && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">🧬 OpenFace3 Analysis</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <div>Total Face Annotations: {openface3Stats.totalFaces}</div>
            <div>Unique Face IDs: {openface3Stats.uniqueFaceIds}</div>
            <div>Average Confidence: {(openface3Stats.avgConfidence * 100).toFixed(1)}%</div>
            <div>Timespan: {openface3Stats.timespan.toFixed(1)}s</div>
            <div>Density: {openface3Stats.facesPerSecond.toFixed(1)} faces/second</div>
            
            <Separator className="my-2" />
            
            <div className="font-semibold">Active Features:</div>
            <div className="grid grid-cols-2 gap-1">
              <div className={overlaySettings.openface3.faceBoxes ? 'text-green-500' : 'text-gray-500'}>
                📦 Face Boxes
              </div>
              <div className={overlaySettings.openface3.landmarks2d ? 'text-green-500' : 'text-gray-500'}>
                📍 Landmarks
              </div>
              <div className={overlaySettings.openface3.actionUnits ? 'text-green-500' : 'text-gray-500'}>
                💪 Action Units
              </div>
              <div className={overlaySettings.openface3.headPose ? 'text-green-500' : 'text-gray-500'}>
                🧭 Head Pose
              </div>
              <div className={overlaySettings.openface3.gaze ? 'text-green-500' : 'text-gray-500'}>
                👁️ Gaze
              </div>
              <div className={overlaySettings.openface3.emotionsEnhanced ? 'text-green-500' : 'text-gray-500'}>
                😀 Emotions
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
```

### JSON Viewer Enhancement

#### Hierarchical Data Inspection
```typescript
// components/FileViewer.tsx - Add OpenFace3-specific formatting
const formatOpenFace3Data = (data: OpenFace3FaceAnnotation[], feature: string) => {
  const currentTime = getCurrentTime();
  const currentFaces = getOpenFace3FacesAtTime(data, currentTime);
  
  switch (feature) {
    case 'faceBoxes':
      return currentFaces.map(face => ({
        annotation_id: face.annotation_id,
        bbox: face.bbox,
        confidence: face.features.confidence,
        timestamp: face.timestamp
      }));
      
    case 'landmarks2d':
      return currentFaces.map(face => ({
        annotation_id: face.annotation_id,
        landmarks_count: face.features.landmarks_2d.length,
        landmarks: face.features.landmarks_2d
      }));
      
    case 'actionUnits':
      return currentFaces.map(face => ({
        annotation_id: face.annotation_id,
        action_units: face.features.action_units
      }));
      
    // ... other feature formatters
    
    default:
      return currentFaces;
  }
};
```

### Demo Integration

#### Enhanced Demo Dataset
The existing demo dataset already includes OpenFace3 files:
```
demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/
├── 2UWdXP.joke1.rep2.take1.Peekaboo_h265_openface3_analysis.json
└── 2UWdXP.joke1.rep2.take1.Peekaboo_h265_openface3_detailed.json
```

#### Demo Button Enhancement
```typescript
// components/VideoAnnotationViewer.tsx - Update demo loading
const handleViewDemo = useCallback(async () => {
  try {
    // Load video file
    const videoResponse = await fetch('/demo/videos/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4');
    const videoBlob = await videoResponse.blob();
    const videoFile = new File([videoBlob], '2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4');
    
    // Load ALL annotation files including OpenFace3
    const annotationFiles = await Promise.all([
      loadDemoFile('/demo/videos_out/.../person_tracking.json'),
      loadDemoFile('/demo/videos_out/.../face_analysis.json'),
      loadDemoFile('/demo/videos_out/.../speech_recognition.json'),
      loadDemoFile('/demo/videos_out/.../openface3_detailed.json'), // NEW
      // ... other files
    ]);
    
    // Enhanced annotation parsing with OpenFace3 support
    const annotationData = await parseAnnotationFiles(annotationFiles);
    
    setVideoFile(videoFile);
    setAnnotationData(annotationData);
    setShowWelcome(false);
  } catch (error) {
    console.error('Failed to load demo with OpenFace3 data:', error);
  }
}, []);
```

## 8. Implementation Roadmap & Next Steps

### Phase 1: Foundation (Days 1-2) - HIGH PRIORITY
**Goal**: Establish data structures and basic parsing

#### Day 1: Type Definitions & Interfaces
- [ ] **Create OpenFace3 type definitions** in `src/types/annotations.ts`
  - [ ] `OpenFace3FaceAnnotation` interface
  - [ ] `OpenFace3ActionUnits` interface with all 8 AUs
  - [ ] `OpenFace3HeadPose`, `OpenFace3Gaze`, `OpenFace3Emotion` interfaces
  - [ ] Extend `StandardAnnotationData` with `openface3_analysis` field

#### Day 2: Parser Functions & Data Access
- [ ] **Create OpenFace3 parser module** in `src/lib/parsers/openface3.ts`
  - [ ] `getOpenFace3FacesAtTime()` - temporal filtering function
  - [ ] `getActionUnitsIntensity()` - AU analysis helper  
  - [ ] `getGazeVector()` - gaze direction calculation
  - [ ] `parseOpenFace3File()` - file loading and validation

- [ ] **Integration with file detection** in `src/lib/fileUtils.ts`
  - [ ] `detectOpenFace3Files()` - auto-discovery logic
  - [ ] Update `FileUploader.tsx` to include OpenFace3 auto-detection

**Deliverable**: OpenFace3 data can be loaded and parsed successfully

### Phase 2: Enhanced Control System (Days 3-4) - HIGH PRIORITY  
**Goal**: Implement hierarchical control UI with master/child toggles

#### Day 3: OverlaySettings Extension
- [ ] **Extend OverlaySettings interface** to support nested controls
  - [ ] Define `OpenFace3Settings` sub-interface
  - [ ] Update all components that use `OverlaySettings`
  - [ ] Implement backward compatibility for existing overlays

#### Day 4: Hierarchical Control UI
- [ ] **Enhance OverlayControls.tsx** for nested controls
  - [ ] Add `renderHierarchicalControl()` component
  - [ ] Implement master toggle logic (all on/off, some on states)
  - [ ] Add expand/collapse functionality for sub-controls
  - [ ] Style with indentation and connecting lines

- [ ] **Update UnifiedControls.tsx** for synchronization
  - [ ] Extend timeline lock functionality for OpenFace3
  - [ ] Handle master/child toggle interactions

**Deliverable**: Complete hierarchical control system working in UI

### Phase 3: Overlay Rendering (Days 5-7) - MEDIUM PRIORITY
**Goal**: Implement all 6 OpenFace3 overlay types with performance optimization

#### Day 5: Basic Overlays (Face Boxes + Head Pose)
- [ ] **Add rendering functions to VideoPlayer.tsx**
  - [ ] `drawOpenFace3FaceBoxes()` - enhanced face detection
  - [ ] `drawOpenFace3HeadPose()` - 3D orientation visualization
  - [ ] Update main `renderOverlays()` function to call new drawers

#### Day 6: Advanced Overlays (Gaze + Emotions)  
- [ ] **Implement complex visualization functions**
  - [ ] `drawOpenFace3Gaze()` - gaze direction vectors with arrows
  - [ ] `drawOpenFace3Emotions()` - enhanced emotion display with valence/arousal

#### Day 7: Performance-Heavy Overlays (Landmarks + Action Units)
- [ ] **Implement data-intensive rendering**
  - [ ] `drawOpenFace3Landmarks()` - 98 facial keypoints (with performance limits)
  - [ ] `drawOpenFace3ActionUnits()` - muscle activation visualization
  - [ ] Add performance monitoring and adaptive quality control

**Deliverable**: All 6 OpenFace3 overlay types rendering correctly

### Phase 4: Timeline Integration & Polish (Days 8-9) - LOW PRIORITY
**Goal**: Complete timeline integration and performance optimization

#### Day 8: Timeline Enhancement
- [ ] **Extend Timeline component** for OpenFace3 tracks
  - [ ] Add `drawOpenFace3Track()` function for density visualization
  - [ ] Update `TimelineSettings` interface for OpenFace3 toggle
  - [ ] Implement timeline synchronization with overlay controls

#### Day 9: Polish & Optimization
- [ ] **Performance optimization**
  - [ ] Implement smart visibility culling
  - [ ] Add adaptive quality control based on frame rate
  - [ ] Optimize canvas batching for landmark rendering

- [ ] **Debug panel integration**
  - [ ] Add OpenFace3 metrics to DebugPanel.tsx
  - [ ] Include performance monitoring display
  - [ ] Add feature-specific JSON viewers

**Deliverable**: Complete, polished OpenFace3 integration

### Testing & Quality Assurance (Day 10)
- [ ] **Comprehensive testing** with demo dataset
  - [ ] Verify all 6 overlay types render correctly
  - [ ] Test hierarchical controls (master/child interactions)
  - [ ] Validate performance with multiple faces and heavy features
  - [ ] Confirm timeline synchronization works properly

- [ ] **Edge case testing**
  - [ ] Missing OpenFace3 data scenarios
  - [ ] Partial feature availability
  - [ ] Performance degradation handling

### Success Criteria & Acceptance Tests

#### Must-Have Features ✅
- [ ] **Master Toggle**: OpenFace3 pipeline can be enabled/disabled as a group
- [ ] **6 Sub-Controls**: Individual control over each OpenFace3 feature
- [ ] **Smart Defaults**: Performance-heavy features start disabled
- [ ] **JSON Viewers**: Per-feature data inspection capability
- [ ] **Demo Integration**: OpenFace3 works with existing demo dataset

#### Performance Requirements ✅
- [ ] **60 FPS playback** maintained with face boxes + head pose + gaze + emotions
- [ ] **30+ FPS playback** maintained with landmarks enabled (up to 3 faces)
- [ ] **Graceful degradation** when performance thresholds exceeded
- [ ] **Memory usage** stays under 200MB for 10-minute demo video

#### User Experience Requirements ✅
- [ ] **Intuitive hierarchy**: Master/child relationship clear in UI
- [ ] **Consistent theming**: Matches existing green face analysis color
- [ ] **Responsive controls**: Sub-controls disabled when master is off
- [ ] **Timeline sync**: OpenFace3 track appears when any feature enabled

### Risk Mitigation

#### Technical Risks
- **Performance with 98 landmarks**: Implement strict face count limits and distance-based LOD
- **Memory consumption**: Use lazy loading and data preprocessing
- **Canvas rendering complexity**: Batch operations and implement visibility culling

#### UX/UI Risks  
- **Control complexity**: Start with collapsed view, use progressive disclosure
- **Visual clutter**: Implement smart defaults favoring lightweight features
- **Learning curve**: Provide clear tooltips and performance warnings

### Future Enhancements (Post-v1.0)

#### Advanced Features
- [ ] **3D landmark support**: Extend to render 3D facial mesh
- [ ] **Action unit intensity visualization**: Heatmap overlay on face
- [ ] **Emotion tracking over time**: Timeline-based emotion graphs
- [ ] **Multi-person emotion comparison**: Side-by-side analysis

#### Performance Optimizations
- [ ] **WebGL rendering**: Hardware-accelerated landmark rendering
- [ ] **Web Workers**: Background data processing for large datasets
- [ ] **Streaming data**: Support for real-time OpenFace3 analysis

---

**Estimated Total Timeline**: 10 working days (2 weeks)
**Resource Requirements**: 1 full-time developer
**Priority Level**: HIGH (major feature enhancement)
**Risk Level**: MEDIUM (performance and complexity challenges)
