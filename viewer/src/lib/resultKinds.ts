/**
 * Kinds of result, each with one label, icon and hue, used wherever the kind
 * appears: Videos columns, lanes, Layers, overlays, New run, Pipelines (viewer
 * design P8, "Icons" and "Colour"; UAT #36, #48). Hues keep clear of the state
 * colours (green done, red failed, blue running); tracks within a kind are
 * shades of its hue, never another kind's.
 */
import {
  AudioLines,
  Captions,
  Clapperboard,
  PersonStanding,
  ScanFace,
  Smile,
  Sparkles,
  StickyNote,
  type LucideIcon,
} from 'lucide-react';

export type ResultKindId = 'speech' | 'speakers' | 'people' | 'faces' | 'face_measures' | 'scenes' | 'ask_ai' | 'notes';

export interface ResultKind {
  id: ResultKindId;
  label: string;
  icon: LucideIcon;
  /** null: neutral (notes are the user's, not a pipeline's). */
  hue: number | null;
}

export const RESULT_KINDS: Record<ResultKindId, ResultKind> = {
  speech: { id: 'speech', label: 'Speech', icon: Captions, hue: 45 },
  speakers: { id: 'speakers', label: 'Speakers', icon: AudioLines, hue: 250 },
  people: { id: 'people', label: 'People', icon: PersonStanding, hue: 170 },
  faces: { id: 'faces', label: 'Faces', icon: Smile, hue: 325 },
  face_measures: { id: 'face_measures', label: 'Face measures', icon: ScanFace, hue: 22 },
  scenes: { id: 'scenes', label: 'Scenes', icon: Clapperboard, hue: 200 },
  ask_ai: { id: 'ask_ai', label: 'Ask AI', icon: Sparkles, hue: 290 },
  notes: { id: 'notes', label: 'Notes', icon: StickyNote, hue: null },
};

const BY_PIPELINE: Record<string, ResultKindId> = {
  speech_recognition: 'speech',
  audio_processing: 'speech',
  speaker_diarization: 'speakers',
  person_tracking: 'people',
  face_analysis: 'faces',
  face_openface3_embedding: 'face_measures',
  scene_detection: 'scenes',
  vlm_annotation: 'ask_ai',
};

const BY_LANE: Record<string, ResultKindId> = {
  speech: 'speech',
  speakers: 'speakers',
  people: 'people',
  person_area: 'people',
  faces: 'faces',
  emotion: 'faces',
  action_units: 'face_measures',
  head_pose: 'face_measures',
  gaze: 'face_measures',
  scenes: 'scenes',
  ask_ai: 'ask_ai',
  notes: 'notes',
};

export const kindOfPipeline = (pipeline: string): ResultKind | null =>
  BY_PIPELINE[pipeline] ? RESULT_KINDS[BY_PIPELINE[pipeline]] : null;

/** A lane's kind: by the pipeline that made it where it's known (OpenFace 3's faces are face measures). */
export function kindOfLane(laneKind: string, pipeline: string | null): ResultKind | null {
  if (laneKind === 'faces' || laneKind === 'emotion') {
    const byPipeline = pipeline ? kindOfPipeline(pipeline) : null;
    if (byPipeline) return byPipeline;
  }
  return BY_LANE[laneKind] ? RESULT_KINDS[BY_LANE[laneKind]] : pipeline ? kindOfPipeline(pipeline) : null;
}

/** The kind's colour; `shade` (0, 1, 2…) steps through lighter and darker versions for tracks within it. */
export function kindColour(kind: ResultKind | null, shade = 0): string {
  if (!kind || kind.hue === null) return `hsl(220, 9%, ${shade % 2 ? 62 : 52}%)`;
  const lightness = [58, 44, 70, 36, 78][shade % 5];
  const hue = (kind.hue + [0, 8, -8, 14, -14][Math.floor(shade / 5) % 5] + 360) % 360;
  return `hsl(${hue}, 72%, ${lightness}%)`;
}

/** Each track's colour, in the order the lanes list them, so a box on the video matches its row. */
export function trackColours(kind: ResultKind | null, keys: Iterable<string>): Map<string, string> {
  const distinct = [...new Set(keys)].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return new Map(distinct.map((key, shade) => [key, kindColour(kind, shade)]));
}
