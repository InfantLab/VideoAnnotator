// Data types for video annotation viewer
// Updated to support VideoAnnotator standard formats
// Reference: https://github.com/InfantLab/VideoAnnotator

// =============================================================================
// STANDARD FORMAT TYPES (VideoAnnotator Compatible)
// =============================================================================

// COCO Format Types for Person Tracking (VideoAnnotator v1.1.1)
// Reference: https://cocodataset.org/#format-data
export interface COCOPersonAnnotation {
  id: number;
  image_id: string;
  category_id: number;
  keypoints: number[]; // [x1, y1, visibility1, x2, y2, visibility2, ...] (17 keypoints * 3)
  num_keypoints: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
  area: number;
  iscrowd: 0 | 1;
  score: number;
  track_id: number;
  timestamp: number;
  frame_number: number;
  // NEW v1.1.1 fields
  person_id: string; // e.g., "person_2UWdXP.joke1.rep3.take1.Peekaboo_h265_001"
  person_label: string; // e.g., "parent", "child"
  label_confidence: number;
  labeling_method: string; // e.g., "automatic_size_based"
}

// WebVTT Types for Speech Recognition
// Reference: https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API
export interface WebVTTCue {
  id?: string;
  startTime: number; // in seconds
  endTime: number; // in seconds
  text: string;
  settings?: string; // WebVTT cue settings
}

// RTTM Types for Speaker Diarization
// Reference: https://github.com/nryant/dscore#rttm
export interface RTTMSegment {
  file_id: string;
  start_time: number; // in seconds
  duration: number; // in seconds
  end_time: number; // calculated: start_time + duration
  speaker_id: string;
  confidence: number;
  pipeline: 'speaker_diarization';
  format: 'rttm';
}

// LAION Face Analysis Types (VideoAnnotator v1.1.1)
// Reference: LAION face analysis backend
export interface LAIONFaceAnnotation {
  id: number;
  image_id: string;
  category_id: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
  area: number;
  iscrowd: 0 | 1;
  score: number;
  face_id: number;
  timestamp: number;
  frame_number: number;
  backend: string; // e.g., "opencv", "laion"
  person_id: string; // e.g., "unknown" or associated person
  person_label: string; // e.g., "unknown", "parent", "child"
  person_label_confidence: number;
  person_labeling_method: string; // e.g., "none", "automatic_size_based"
  attributes: {
    emotions: Record<string, {
      score: number;
      rank: number;
      raw_score: number;
    }>;
    model_info: {
      model_size: string; // e.g., "small"
      embedding_dim: number; // e.g., 1152
    };
  };
}

// Scene Detection Types (Updated for v1.1.1 COCO compliance)
export interface SceneAnnotation {
  id: number;
  image_id: string;
  category_id: number;
  bbox: [number, number, number, number]; // full frame typically
  area: number;
  iscrowd: 0 | 1;
  score: number;
  video_id: string;
  timestamp: number; // scene center timestamp
  start_time: number;
  end_time: number;
  duration: number;
  scene_type: string;
  frame_start: number;
  frame_end: number;
  all_scores: Record<string, number>;
}

// Pipeline Result Wrappers (VideoAnnotator v1.1.1)
export interface PipelineResult<T> {
  results: T[];
  processing_time: number;
  status: string; // e.g., "completed", "failed"
}

// VideoAnnotator v1.1.1 Complete Results Structure
export interface VideoAnnotatorCompleteResults {
  video_path: string;
  output_dir: string;
  start_time: string; // ISO timestamp
  config: {
    scene_detection: {
      enabled: boolean;
      threshold: number;
      min_scene_length: number;
    };
    person_tracking: {
      enabled: boolean;
      model_name: string;
      confidence_threshold: number;
    };
    face_analysis: {
      enabled: boolean;
      backend: string;
      detection_confidence: number;
      enable_action_units: boolean;
      enable_head_pose: boolean;
      enable_gaze: boolean;
      max_faces: number;
    };
    audio_processing: {
      enabled: boolean;
      whisper_model: string;
      sample_rate: number;
    };
  };
  pipeline_results: {
    scene?: PipelineResult<SceneAnnotation>;
    person?: PipelineResult<COCOPersonAnnotation>;
    face?: PipelineResult<LAIONFaceAnnotation>;
    audio?: PipelineResult<any>; // Audio results structure varies
  };
  errors: any[];
  end_time: string; // ISO timestamp
  total_duration: number; // seconds
}

// Unified Standard Annotation Data Structure (Updated for v1.1.1)
export interface StandardAnnotationData {
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
  face_analysis?: LAIONFaceAnnotation[]; // NEW: Face analysis support
  audio_file?: File; // Separate WAV file
  metadata?: {
    created: string;
    version: string;
    pipelines: string[];
    source: 'videoannotator' | 'custom';
    // NEW: Processing information from VideoAnnotator
    processing_config?: VideoAnnotatorCompleteResults['config'];
    processing_time?: number;
    total_duration?: number;
  };
}

// File Type Detection (Updated for v1.1.1)
export type SupportedFileType =
  | 'video/mp4'
  | 'video/webm'
  | 'video/avi'
  | 'video/mov'
  | 'text/vtt'
  | 'text/plain' // for RTTM
  | 'application/json'
  | 'application/json+complete_results' // NEW: VideoAnnotator complete results
  | 'application/json+face_analysis' // NEW: LAION face analysis
  | 'audio/wav'
  | 'audio/mpeg';

export interface FileTypeInfo {
  type: SupportedFileType;
  extension: string;
  pipeline?: string;
  description: string;
}

// =============================================================================
// LEGACY TYPES (for backward compatibility - to be deprecated)
// =============================================================================

export interface Point {
  x: number;
  y: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PoseKeypoint {
  x: number;
  y: number;
  confidence: number;
}

export interface PersonPose {
  id: number;
  keypoints: PoseKeypoint[];
  connections: [number, number][]; // pairs of keypoint indices to connect
  confidence: number;
}

export interface FaceData {
  id: number;
  boundingBox: BoundingBox;
  emotion: string;
  confidence: number;
}

export interface FrameData {
  frameNumber: number;
  timestamp: number;
  persons: PersonPose[];
  faces: FaceData[];
  audioEmotion?: {
    emotion: string;
    confidence: number;
  };
  motionIntensity: number;
}

export interface EventData {
  id: string;
  type: 'subtitle' | 'action' | 'interaction' | 'audio_emotion';
  label: string;
  content?: string; // for subtitles
  startTime: number;
  endTime: number;
  participants?: number[]; // person IDs involved
  confidence?: number;
}

export interface AudioData {
  waveform: number[]; // amplitude values
  sampleRate: number;
  duration: number;
}

export interface AnnotationData {
  video: {
    filename: string;
    duration: number;
    frameRate: number;
    width: number;
    height: number;
  };
  frames: Record<number, FrameData>;
  events: EventData[];
  audio: AudioData;
  metadata: {
    created: string;
    version: string;
    pipeline: string;
  };
}

export interface OverlaySettings {
  pose: boolean;           // COCO person tracking keypoints
  subtitles: boolean;      // WebVTT speech recognition text
  speakers: boolean;       // RTTM speaker diarization
  scenes: boolean;         // Scene detection boundaries
  faces: boolean;          // NEW: LAION face detection and analysis
  emotions: boolean;       // NEW: Emotion recognition overlays

  // Legacy fields for backward compatibility (deprecated)
  faceEmotion?: boolean;
  audioEmotion?: boolean;
  events?: boolean;
}

export interface TimelineSettings {
  showSubtitles: boolean;    // WebVTT speech recognition
  showSpeakers: boolean;     // RTTM speaker diarization  
  showScenes: boolean;       // Scene detection
  showMotion: boolean;       // Person tracking motion data
  showFaces: boolean;        // NEW: Face analysis timeline
  showEmotions: boolean;     // NEW: Emotion analysis timeline

  // Legacy fields for backward compatibility (deprecated)
  showEvents?: boolean;
  showWaveform?: boolean;
}

// =============================================================================
// COCO KEYPOINT CONSTANTS
// =============================================================================

// COCO Human Pose Keypoints (17 keypoints)
// Reference: https://cocodataset.org/#keypoints-2017
export const COCO_KEYPOINT_NAMES = [
  'nose',           // 0
  'left_eye',       // 1
  'right_eye',      // 2
  'left_ear',       // 3
  'right_ear',      // 4
  'left_shoulder',  // 5
  'right_shoulder', // 6
  'left_elbow',     // 7
  'right_elbow',    // 8
  'left_wrist',     // 9
  'right_wrist',    // 10
  'left_hip',       // 11
  'right_hip',      // 12
  'left_knee',      // 13
  'right_knee',     // 14
  'left_ankle',     // 15
  'right_ankle'     // 16
] as const;

// YOLO/Ultralytics Skeleton Connections (converted from 1-based to 0-based indexing)
// Based on Ultralytics YOLO pose estimation standard
export const COCO_SKELETON_CONNECTIONS: [number, number][] = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], // legs: ankle->knee->hip
  [5, 11], [6, 12], [5, 6],                         // torso: shoulder->hip, shoulder-shoulder
  [5, 7], [6, 8], [7, 9], [8, 10],                  // arms: shoulder->elbow->wrist
  [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],          // head: ear-ear, nose-ear, eye-ear  
  [3, 5], [4, 6]                                    // eye to shoulder connections
];

// Visibility flags for COCO keypoints
export enum COCOVisibility {
  NOT_LABELED = 0,
  LABELED_NOT_VISIBLE = 1,
  LABELED_VISIBLE = 2
}

// YOLO/Ultralytics Color Palette for Pose Estimation
// Based on Ultralytics pose_palette colors
export const YOLO_POSE_PALETTE = [
  [255, 128, 0],   // 0: orange
  [255, 153, 51],  // 1: light orange  
  [255, 178, 102], // 2: lighter orange
  [230, 230, 0],   // 3: yellow
  [255, 153, 255], // 4: pink
  [153, 204, 255], // 5: light blue
  [255, 102, 255], // 6: magenta
  [255, 51, 255],  // 7: bright magenta
  [102, 178, 255], // 8: blue
  [51, 153, 255],  // 9: bright blue
  [255, 153, 153], // 10: light red
  [255, 102, 102], // 11: red
  [255, 51, 51],   // 12: bright red
  [153, 255, 153], // 13: light green
  [102, 255, 102], // 14: green
  [51, 255, 51],   // 15: bright green
  [0, 255, 0],     // 16: pure green
  [0, 0, 255],     // 17: pure blue
  [255, 0, 0],     // 18: pure red
];

// YOLO Limb Colors (indices into pose_palette)
export const YOLO_LIMB_COLORS = [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16];

// YOLO Keypoint Colors (indices into pose_palette)  
export const YOLO_KEYPOINT_COLORS = [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9];