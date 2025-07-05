// Data types for video annotation viewer

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
  pose: boolean;
  faceEmotion: boolean;
  audioEmotion: boolean;
  subtitles: boolean;
  events: boolean;
}

export interface TimelineSettings {
  showSubtitles: boolean;
  showEvents: boolean;
  showWaveform: boolean;
  showMotion: boolean;
}