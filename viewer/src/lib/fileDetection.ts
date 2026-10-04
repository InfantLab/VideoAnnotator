/**
 * Which pipeline produced a file: the one place the viewer decides.
 *
 * There used to be four detectors (two in merger.ts, one in fileUtils.ts, and
 * type lists in FileUploader.tsx) and they disagreed. The upload screen called
 * every JSON file "unknown" while the loader called it something else, and
 * detectors that parsed only the first few KB of a file threw on anything
 * larger, so a looser check claimed it (OpenFace's export was taken for person
 * tracking). Detection now reads the extension, then classifies the whole
 * parsed JSON by the fields its annotations carry. The contract test checks
 * it against real VideoAnnotator outputs.
 */

import { isValidElanFile } from './parsers/elan';

export type DetectedFileType =
  | 'video'
  | 'audio'
  | 'person_tracking'
  | 'speech_recognition'
  | 'speaker_diarization'
  | 'scene_detection'
  | 'vlm_annotation'
  | 'elan_ground_truth'
  | 'face_analysis'
  | 'openface3_faces'
  | 'complete_results'
  | 'unknown';

export interface DetectedFile {
  file: File;
  type: DetectedFileType;
  pipeline?: string;
  /** 0-1: how sure the detection is. */
  confidence: number;
}

/** Types that carry annotations to draw (everything but media and unknown). */
export const ANNOTATION_TYPES: readonly DetectedFileType[] = [
  'person_tracking',
  'speech_recognition',
  'speaker_diarization',
  'scene_detection',
  'vlm_annotation',
  'elan_ground_truth',
  'face_analysis',
  'openface3_faces',
  'complete_results',
];

export function isAnnotationType(type: DetectedFileType): boolean {
  return ANNOTATION_TYPES.includes(type);
}

const VIDEO_EXTENSIONS = ['mp4', 'webm', 'avi', 'mov', 'mkv'];
const AUDIO_EXTENSIONS = ['wav', 'mp3', 'aac', 'ogg'];

export async function detectFileType(file: File): Promise<DetectedFile> {
  const name = file.name.toLowerCase();
  const extension = name.split('.').pop() || '';
  const mimeType = file.type.toLowerCase();

  if (VIDEO_EXTENSIONS.includes(extension) || mimeType.startsWith('video/')) {
    return { file, type: 'video', confidence: 0.95 };
  }
  if (AUDIO_EXTENSIONS.includes(extension) || mimeType.startsWith('audio/')) {
    return { file, type: 'audio', confidence: 0.95 };
  }
  if (extension === 'vtt' || mimeType === 'text/vtt') {
    const valid = (await file.slice(0, 100).text()).trim().startsWith('WEBVTT');
    return { file, type: 'speech_recognition', pipeline: 'speech_recognition', confidence: valid ? 0.9 : 0.3 };
  }
  if (extension === 'rttm') {
    const valid = (await file.slice(0, 1000).text()).includes('SPEAKER');
    return { file, type: 'speaker_diarization', pipeline: 'speaker_diarization', confidence: valid ? 0.9 : 0.3 };
  }
  if (extension === 'eaf') {
    const valid = await isValidElanFile(file);
    return { file, type: 'elan_ground_truth', pipeline: 'elan_ground_truth', confidence: valid ? 0.9 : 0.3 };
  }
  if (extension === 'json' || mimeType === 'application/json') {
    return detectJSONFile(file);
  }
  return { file, type: 'unknown', confidence: 0 };
}

type JSONStructureType = Exclude<
  DetectedFileType,
  'video' | 'audio' | 'unknown' | 'speech_recognition' | 'speaker_diarization' | 'elan_ground_truth'
>;

const STRUCTURE_CONFIDENCE: Record<JSONStructureType, number> = {
  complete_results: 0.95,
  openface3_faces: 0.95,
  vlm_annotation: 0.95,
  face_analysis: 0.9,
  person_tracking: 0.9,
  scene_detection: 0.9,
};

/** The `pipeline` a type is reported as, where it isn't the type's own name. */
const PIPELINE_OF: Partial<Record<DetectedFileType, string>> = {
  openface3_faces: 'openface3',
};

/**
 * Classifies a fully parsed VideoAnnotator JSON output by the fields its
 * annotations carry.
 */
export function detectJSONStructure(data: unknown): JSONStructureType | 'person_id_summary' | null {
  if (!data || typeof data !== 'object') return null;
  const d = data as Record<string, unknown>;

  if (d.video_path && d.pipeline_results && d.config) return 'complete_results';

  const metadata = d.metadata as Record<string, unknown> | undefined;
  if (metadata?.pipeline && Array.isArray(d.faces)) return 'openface3_faces';
  // person_tracking's *_person_tracks.json: labels per person ID, no frames to draw.
  if (Array.isArray(d.person_tracks) && !d.annotations) return 'person_id_summary';
  if (Array.isArray(d.scenes)) return 'scene_detection';

  const annotations = Array.isArray(data) ? data : (d.annotations ?? d.results);
  if (!Array.isArray(annotations) || annotations.length === 0) return null;
  const first = annotations[0];
  if (!first || typeof first !== 'object') return null;

  if ('openface3' in first) return 'openface3_faces';
  if ('sampling_mode' in first && 'reasoning' in first) return 'vlm_annotation';
  if ('scene_type' in first || ('start_time' in first && 'end_time' in first)) return 'scene_detection';
  if ('face_id' in first) return 'face_analysis';
  if ('keypoints' in first && 'bbox' in first) return 'person_tracking';
  return null;
}

/**
 * VideoAnnotator names each output `<video>_<suffix>` (the server's
 * `outputs[].file` in each pipeline's registry metadata). Used only when the
 * JSON itself doesn't say, e.g. a pipeline that found nothing to annotate.
 */
const FILE_SUFFIXES: Array<[string, JSONStructureType]> = [
  ['_person_tracking.json', 'person_tracking'],
  ['_scene_detection.json', 'scene_detection'],
  ['_vlm_annotation.json', 'vlm_annotation'],
  ['_face_detections.json', 'face_analysis'],
  ['_openface3_analysis.json', 'openface3_faces'],
  ['_openface3_detailed.json', 'openface3_faces'],
  ['complete_results.json', 'complete_results'],
];

async function detectJSONFile(file: File): Promise<DetectedFile> {
  let data: unknown;
  try {
    data = JSON.parse(await file.text());
  } catch {
    return { file, type: 'unknown', confidence: 0 };
  }

  const structure = detectJSONStructure(data);
  if (structure === 'person_id_summary') {
    return { file, type: 'unknown', confidence: 0.9 };
  }
  if (structure) {
    return { file, type: structure, pipeline: PIPELINE_OF[structure] ?? structure, confidence: STRUCTURE_CONFIDENCE[structure] };
  }

  const name = file.name.toLowerCase();
  const bySuffix = FILE_SUFFIXES.find(([suffix]) => name.endsWith(suffix));
  if (bySuffix) {
    const type = bySuffix[1];
    return { file, type, pipeline: PIPELINE_OF[type] ?? type, confidence: 0.5 };
  }
  return { file, type: 'unknown', confidence: 0.2 };
}

export async function detectFileTypes(files: File[]): Promise<DetectedFile[]> {
  return Promise.all(files.map(detectFileType));
}

export function describeFileType(type: DetectedFileType): string {
  switch (type) {
    case 'video': return 'Video File';
    case 'audio': return 'Audio File';
    case 'person_tracking': return 'Person Tracking (COCO)';
    case 'speech_recognition': return 'Speech Recognition (WebVTT)';
    case 'speaker_diarization': return 'Speaker Diarization (RTTM)';
    case 'scene_detection': return 'Scene Detection (JSON)';
    case 'vlm_annotation': return 'VLM Frame Annotation (JSON)';
    case 'elan_ground_truth': return 'ELAN Ground Truth (.eaf)';
    case 'face_analysis': return 'Face Analysis (COCO)';
    case 'openface3_faces': return 'OpenFace3 Analysis (JSON)';
    case 'complete_results': return 'Complete Results (VideoAnnotator)';
    default: return 'Unknown File Type';
  }
}

/** Upload limits per kind of file. JSON results are not capped: OpenFace's run to tens of MB. */
export function validateFileSize(file: File, type: DetectedFileType): { valid: boolean; error?: string } {
  if (type === 'video' && file.size > 500 * 1024 * 1024) {
    return { valid: false, error: 'Video file too large (max 500MB)' };
  }
  if (type === 'audio' && file.size > 100 * 1024 * 1024) {
    return { valid: false, error: 'Audio file too large (max 100MB)' };
  }
  return { valid: true };
}

/** Whether a set of detected files can be opened, and what to warn about. */
export function validateFileSet(detected: DetectedFile[]): { valid: boolean; missing: string[]; warnings: string[] } {
  const types = detected.map((d) => d.type);
  const missing: string[] = [];
  const warnings: string[] = [];

  if (!types.includes('video')) {
    missing.push('Video file (.mp4, .webm, .avi, .mov)');
  }
  if (!types.some(isAnnotationType)) {
    warnings.push('No annotation files detected. Consider adding tracking, speech, or scene data.');
  }
  const unknown = types.filter((t) => t === 'unknown').length;
  if (unknown > 0) {
    warnings.push(`${unknown} file(s) could not be identified. Check file formats.`);
  }
  return { valid: missing.length === 0, missing, warnings };
}
