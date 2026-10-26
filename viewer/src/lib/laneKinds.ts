/**
 * What kinds of lane the Video page draws, and from which table rows (spec 028
 * contracts/viewer.md, "Lane kinds"). Order on the page is this table's order;
 * Notes are always last.
 */
import type { Row } from '@/lib/table/rows';

export type LaneForm = 'span' | 'moment' | 'numeric' | 'presence';

export interface LaneKind {
  kind: string;
  label: string;
  /** For the tooltip on the lane's label (SC-002). */
  description: string;
  measures: string[];
  form: LaneForm;
  /** What makes one row of the lane: a track, the row's label, or nothing (one row). */
  rowsBy: 'track' | 'label' | 'none';
  /** Whether this kind draws on the video. */
  overlay: boolean;
  unit?: string;
  order: number;
}

const KINDS: Omit<LaneKind, 'order'>[] = [
  {
    kind: 'speech',
    label: 'Speech',
    description: 'What was said, as transcribed: one bar per utterance',
    measures: ['speech'],
    form: 'span',
    rowsBy: 'none',
    overlay: true,
  },
  {
    kind: 'speakers',
    label: 'Speakers',
    description: 'Who spoke when: one row per speaker the pipeline told apart',
    measures: ['speaker_turn'],
    form: 'span',
    rowsBy: 'track',
    overlay: true,
  },
  {
    kind: 'people',
    label: 'People',
    description: 'When each tracked person was seen, in the sampled frames',
    measures: ['person'],
    form: 'presence',
    rowsBy: 'track',
    overlay: true,
  },
  {
    kind: 'faces',
    label: 'Faces',
    description: 'When faces were found, and how many per sampled frame',
    measures: ['face'],
    form: 'presence',
    rowsBy: 'track',
    overlay: true,
  },
  {
    kind: 'emotion',
    label: 'Emotion',
    description: 'The most likely facial expression per detected face, and valence and arousal where measured',
    measures: ['emotion', 'valence', 'arousal'],
    form: 'moment',
    rowsBy: 'label',
    overlay: true,
  },
  {
    kind: 'scenes',
    label: 'Scenes',
    description: 'Stretches of the video the pipeline judged to be one scene, with its type',
    measures: ['scene'],
    form: 'span',
    rowsBy: 'none',
    overlay: true,
  },
  {
    kind: 'ask_ai',
    label: 'Ask AI',
    description: "A vision model's answer at each sampled moment",
    measures: ['vlm_label'],
    form: 'moment',
    rowsBy: 'none',
    overlay: true,
  },
  {
    kind: 'action_units',
    label: 'Action units',
    description: 'Facial muscle movements (FACS action units), intensity per sampled face',
    measures: ['action_unit'],
    form: 'numeric',
    rowsBy: 'label',
    overlay: false,
  },
  {
    kind: 'head_pose',
    label: 'Head pose',
    description: 'Head rotation (pitch, yaw, roll) per sampled face',
    measures: ['head_pose'],
    form: 'numeric',
    rowsBy: 'label',
    overlay: false,
    unit: '°',
  },
  {
    kind: 'gaze',
    label: 'Gaze',
    description: 'Gaze direction (x, y, z) per sampled face',
    measures: ['gaze'],
    form: 'numeric',
    rowsBy: 'label',
    overlay: false,
  },
  {
    kind: 'person_area',
    label: 'People · box area (not movement)',
    description:
      "The summed area of every person's box per sampled frame. It changes when people come closer or leave, so it is not a measure of movement",
    measures: [],
    form: 'numeric',
    rowsBy: 'none',
    overlay: false,
    unit: 'px²',
  },
];

export const LANE_KINDS: LaneKind[] = KINDS.map((k, order) => ({ ...k, order }));

export const NOTES_KIND: LaneKind = {
  kind: 'notes',
  label: 'Notes (yours)',
  description: 'Notes you and others on this server wrote on this video; not pipeline output',
  measures: ['note'],
  form: 'span',
  rowsBy: 'none',
  overlay: false,
  order: Number.MAX_SAFE_INTEGER,
};

/** In the Data tab and the hover, but not drawn as lanes of their own. */
export const UNDRAWN_MEASURES = new Set([
  'emotion_score',
  'age',
  'gender',
  'vlm_reasoning',
  'vlm_fit',
  'vlm_fit_problem',
  'action_unit_presence',
  'scene_score',
]);

/** A shaped Ask AI answer's fields (spec 033): `vlm.<name>`, in the Data tab with the answer. */
const isAnswerField = (measure: string) => measure.startsWith('vlm.');

const BY_MEASURE = new Map<string, LaneKind>();
for (const k of [...LANE_KINDS, NOTES_KIND]) for (const m of k.measures) BY_MEASURE.set(m, k);
const BY_KIND = new Map([...LANE_KINDS, NOTES_KIND].map((k) => [k.kind, k]));

function otherKind(pipeline: string, measure: string, numeric: boolean): LaneKind {
  return {
    kind: `other:${pipeline}/${measure}`,
    label: `${pipeline} · ${measure}`,
    description: `"${measure}" from ${pipeline}; the viewer has no track of its own for it`,
    measures: [measure],
    form: numeric ? 'numeric' : 'moment',
    rowsBy: numeric ? 'label' : 'track',
    overlay: false,
    order: LANE_KINDS.length,
  };
}

/** The lane a row belongs to; null for measures shown only in the Data tab. */
export function kindOf(row: Pick<Row, 'pipeline' | 'measure' | 'value'>): LaneKind | null {
  if (UNDRAWN_MEASURES.has(row.measure) || isAnswerField(row.measure)) return null;
  return BY_MEASURE.get(row.measure) ?? otherKind(row.pipeline, row.measure, row.value !== null);
}

export function laneKind(kind: string): LaneKind | undefined {
  if (kind.startsWith('other:')) {
    const [pipeline, measure] = kind.slice('other:'.length).split('/');
    return otherKind(pipeline, measure ?? '', false);
  }
  return BY_KIND.get(kind);
}

/** Short names for a pipeline in a lane label: the tool, as the registry's display name says it (D1). */
export const PIPELINE_SHORT_NAMES: Record<string, string> = {
  face_analysis: 'DeepFace',
  face_openface3_embedding: 'OpenFace 3',
  person_tracking: 'YOLO26 pose',
  scene_detection: 'PySceneDetect + CLIP',
  speaker_diarization: 'pyannote',
  speech_recognition: 'Whisper',
  vlm_annotation: 'Ask AI',
};

export const pipelineShortName = (pipeline: string) => PIPELINE_SHORT_NAMES[pipeline] ?? pipeline;

/**
 * The key that switches a lane's overlay on the video. Faces and emotion come
 * from two pipelines that draw separately, so OpenFace 3's have their own.
 */
export function overlayKey(kind: string, pipeline: string | null): string | null {
  if (!laneKind(kind)?.overlay) return null;
  return pipeline === 'face_openface3_embedding' ? `${kind}.openface3` : kind;
}

/** Overlay settings before spec 028, by the lane kind that replaces each. */
export const OLD_OVERLAY_KEYS: Record<string, string> = {
  pose: 'people',
  subtitles: 'speech',
  speakers: 'speakers',
  scenes: 'scenes',
  faces: 'faces',
  emotions: 'emotion',
};

/** Where a measure without a lane of its own sits in the Data tab's lane filter. */
const UNDRAWN_KIND: Record<string, string> = {
  emotion_score: 'emotion',
  age: 'faces',
  gender: 'faces',
  vlm_reasoning: 'ask_ai',
  vlm_fit: 'ask_ai',
  vlm_fit_problem: 'ask_ai',
  action_unit_presence: 'action_units',
  scene_score: 'scenes',
};

/** The lane kind a table row is filed under in the Data tab: every row has one. */
export function tableKindOf(row: Pick<Row, 'pipeline' | 'measure' | 'value'>): LaneKind {
  const drawn = kindOf(row);
  if (drawn) return drawn;
  return laneKind(isAnswerField(row.measure) ? 'ask_ai' : UNDRAWN_KIND[row.measure])!;
}
