// Data types for video annotation viewer
// Updated to support VideoAnnotator standard formats
// Reference: https://github.com/InfantLab/VideoAnnotator

// =============================================================================
// STANDARD FORMAT TYPES (VideoAnnotator Compatible)
// =============================================================================

// COCO Format Types for Person Tracking
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
  track_id?: number;
  timestamp: number;
  frame_number: number;
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

// Scene Detection Types
export interface SceneAnnotation {
  id: number;
  video_id: string;
  timestamp: number; // scene center timestamp
  start_time: number;
  end_time: number;
  duration: number;
  scene_type: string;
  bbox: [number, number, number, number]; // full frame typically
  score: number;
  frame_start: number;
  frame_end: number;
  all_scores: Record<string, number>;
}

// Pipeline Result Wrappers (as produced by VideoAnnotator)
export interface PipelineResult<T> {
  pipeline: string;
  format: string;
  data: T[];
  metadata: {
    pipeline_name: string;
    output_format: string;
    processed_segments: number;
  };
}

// Unified Standard Annotation Data Structure
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
  audio_file?: File; // Separate WAV file
  metadata?: {
    created: string;
    version: string;
    pipelines: string[];
    source: 'videoannotator' | 'custom';
  };
}

// File Type Detection
export type SupportedFileType =
  | 'video/mp4'
  | 'video/webm'
  | 'video/avi'
  | 'video/mov'
  | 'text/vtt'
  | 'text/plain' // for RTTM
  | 'application/json'
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

// COCO Skeleton Connections (which keypoints to connect)
export const COCO_SKELETON_CONNECTIONS: [number, number][] = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], // legs
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6],         // arms
  [5, 11], [6, 12],                                 // torso
  [0, 1], [0, 2], [1, 3], [2, 4],                  // head
  [3, 5], [4, 6]                                    // head to shoulders
];

// Visibility flags for COCO keypoints
export enum COCOVisibility {
  NOT_LABELED = 0,
  LABELED_NOT_VISIBLE = 1,
  LABELED_VISIBLE = 2
}