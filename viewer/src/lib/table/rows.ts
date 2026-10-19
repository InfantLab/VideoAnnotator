/**
 * The spec 025 results table, built in the browser (spec 028 R5).
 *
 * A twin of `src/videoannotator/exporters/readers.py` and `table.video_rows`:
 * the same files give the same rows, so the Data tab, the lanes and an export
 * can't disagree. It reads the files' own text rather than the viewer's parsed
 * structures, which fill in defaults and drop fields the table keeps (track,
 * frame, absent action units). `tests/unit/test_viewer_table_fixture.py`
 * writes the expected rows from the real exporter; `tableRows.test.ts` checks
 * this file against them. Change both sides together.
 */

export const TABLE_FORMAT_VERSION = 1;
export const NOT_RECORDED = 'not recorded';

export const COLUMNS = [
  'run',
  'video',
  'pipeline',
  'event',
  'track',
  'start',
  'end',
  'frame',
  'measure',
  'label',
  'value',
  'confidence',
  'x',
  'y',
  'width',
  'height',
  'model',
  'videoannotator_version',
  'file',
  'author',
] as const;

export type Column = (typeof COLUMNS)[number];

export interface Row {
  run: string;
  video: string;
  pipeline: string;
  event: number;
  track: string;
  start: number;
  end: number | null;
  frame: number | null;
  measure: string;
  label: string | null;
  value: number | null;
  confidence: number | null;
  x: number | null;
  y: number | null;
  width: number | null;
  height: number | null;
  model: string;
  videoannotator_version: string;
  file: string;
  author: string;
}

export interface SourceFile {
  name: string;
  text: string;
}

export interface SkippedFile {
  file: string;
  reason: string;
}

interface Measure {
  measure: string;
  label: string | null;
  value: number | null;
  confidence: number | null;
}

interface Event {
  start: number;
  measures: Measure[];
  end: number | null;
  frame: number | null;
  track: string;
  box: [number, number, number, number] | null;
}

interface FileContents {
  events: Event[];
  annotatedModel: string | null;
}

class ReaderError extends Error {}

type Json = Record<string, unknown>;

const isDict = (v: unknown): v is Json => !!v && typeof v === 'object' && !Array.isArray(v);

function num(value: unknown): number | null {
  if (typeof value === 'boolean') return Number(value);
  return typeof value === 'number' ? value : null;
}

// Python's str() for the values a JSON file can hold where text is expected.
function pyStr(value: unknown): string {
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  return String(value);
}

function text(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const s = pyStr(value);
  return s ? s : null;
}

const truthy = (v: unknown) =>
  !(v === null || v === undefined || v === false || v === 0 || v === '' ||
    (Array.isArray(v) && v.length === 0) || (isDict(v) && Object.keys(v).length === 0));

const measure = (m: string, label: string | null = null, value: number | null = null, confidence: number | null = null): Measure =>
  ({ measure: m, label, value, confidence });

function loadCoco(source: string): [Json[], Map<unknown, Json>] {
  let data: unknown;
  try {
    data = JSON.parse(source);
  } catch (e) {
    throw new ReaderError(`unreadable: ${e instanceof Error ? e.message : e}`);
  }
  if (!isDict(data) || !Array.isArray(data.annotations)) {
    throw new ReaderError('unreadable: no COCO annotations list');
  }
  const images = new Map<unknown, Json>();
  for (const image of Array.isArray(data.images) ? data.images : []) {
    if (isDict(image)) images.set(image.id, image);
  }
  return [data.annotations.filter(isDict), images];
}

function time(annotation: Json, image: Json, ...keys: string[]): number {
  for (const source of [annotation, image]) {
    for (const key of keys) {
      const value = num(source[key]);
      if (value !== null) return value;
    }
  }
  throw new ReaderError(`annotation ${annotation.id} has no time`);
}

const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);

function frameOf(annotation: Json, image: Json): number | null {
  for (const source of [annotation, image]) {
    if (isInt(source.frame_number)) return source.frame_number;
  }
  return null;
}

function boxOf(annotation: Json): [number, number, number, number] | null {
  const bbox = annotation.bbox;
  if (Array.isArray(bbox) && bbox.length === 4 && bbox.every((v) => num(v) !== null)) {
    return [bbox[0], bbox[1], bbox[2], bbox[3]];
  }
  return null;
}

function scores(name: string, values: unknown): Measure[] {
  if (!isDict(values)) return [];
  return Object.entries(values)
    .filter(([, v]) => num(v) !== null)
    .map(([label, v]) => measure(name, label, num(v)));
}

const event = (start: number, measures: Measure[], rest: Partial<Event> = {}): Event => ({
  start,
  measures,
  end: null,
  frame: null,
  track: '',
  box: null,
  ...rest,
});

function readPerson(source: string): FileContents {
  const [annotations, images] = loadCoco(source);
  const events = annotations.map((a) => {
    const image = images.get(a.image_id) ?? {};
    const track = truthy(a.person_id)
      ? a.person_id
      : a.track_id !== null && a.track_id !== undefined ? pyStr(a.track_id) : '';
    return event(time(a, image, 'timestamp'), [measure('person', text(a.person_label), null, num(a.score))], {
      frame: frameOf(a, image),
      track: pyStr(track),
      box: boxOf(a),
    });
  });
  return { events, annotatedModel: null };
}

function readDeepface(source: string): FileContents {
  const [annotations, images] = loadCoco(source);
  const events = annotations.map((a) => {
    const image = images.get(a.image_id) ?? {};
    const measures = [measure('face', null, null, num(a.score))];
    if (a.emotion !== null && a.emotion !== undefined) {
      measures.push(measure('emotion', text(a.emotion), null, num(a.emotion_confidence)));
    }
    measures.push(...scores('emotion_score', a.emotion_scores));
    if (num(a.age) !== null) measures.push(measure('age', null, num(a.age)));
    if (a.gender !== null && a.gender !== undefined) {
      measures.push(measure('gender', text(a.gender), null, num(a.gender_confidence)));
    }
    // face_id is the face's index within its frame, not a track over time.
    return event(time(a, image, 'timestamp'), measures, { frame: frameOf(a, image), box: boxOf(a) });
  });
  return { events, annotatedModel: null };
}

function readOpenface3(source: string): FileContents {
  const [annotations, images] = loadCoco(source);
  const events = annotations.map((a) => {
    const image = images.get(a.image_id) ?? {};
    const features: Json = isDict(a.openface3) ? a.openface3 : {};
    const start = num(features.timestamp);
    const measures = [measure('face', null, null, num(features.confidence))];

    const units = features.action_units;
    if (isDict(units)) {
      for (const [name, unit] of Object.entries(units)) {
        if (isDict(unit)) measures.push(measure('action_unit', name, num(unit.intensity)));
      }
      for (const [name, unit] of Object.entries(units)) {
        if (isDict(unit) && unit.presence !== null && unit.presence !== undefined) {
          measures.push(measure('action_unit_presence', name, truthy(unit.presence) ? 1 : 0));
        }
      }
    }

    for (const [name, components] of [
      ['head_pose', ['pitch', 'yaw', 'roll']],
      ['gaze', ['direction_x', 'direction_y', 'direction_z']],
    ] as const) {
      const part = features[name];
      if (isDict(part)) {
        const confidence = num(part.confidence);
        for (const c of components) {
          if (num(part[c]) !== null) measures.push(measure(name, c, num(part[c]), confidence));
        }
      }
    }

    const emotion = features.emotion;
    if (isDict(emotion)) {
      const probabilities = emotion.probabilities;
      const dominant = text(emotion.dominant);
      if (dominant !== null) {
        const confidence = isDict(probabilities) ? num(probabilities[dominant]) : null;
        measures.push(measure('emotion', dominant, null, confidence));
      }
      measures.push(...scores('emotion_score', probabilities));
      for (const key of ['valence', 'arousal']) {
        if (num(emotion[key]) !== null) measures.push(measure(key, null, num(emotion[key])));
      }
    }

    const track = features.track_id;
    return event(start !== null ? start : time(a, image, 'timestamp'), measures, {
      frame: frameOf(a, image),
      track: track !== null && track !== undefined ? pyStr(track) : '',
      box: boxOf(a),
    });
  });
  return { events, annotatedModel: null };
}

function readScene(source: string): FileContents {
  const [annotations, images] = loadCoco(source);
  const events = annotations.map((a) => {
    const image = images.get(a.image_id) ?? {};
    const start = num(a.start_time);
    return event(
      start !== null ? start : time(a, image, 'timestamp'),
      [measure('scene', text(a.scene_type), null, num(a.score)), ...scores('scene_score', a.all_scores)],
      { end: num(a.end_time), frame: isInt(a.frame_start) ? a.frame_start : null },
    );
  });
  return { events, annotatedModel: null };
}

function readVlm(source: string): FileContents {
  const [annotations, images] = loadCoco(source);
  const models: string[] = [];
  const events = annotations.map((a) => {
    const image = images.get(a.image_id) ?? {};
    const label = text(a.label);
    const measures = [measure('vlm_label', label)];
    const reasoning = text(a.reasoning);
    if (reasoning !== null && reasoning !== label) measures.push(measure('vlm_reasoning', reasoning));
    if (truthy(a.model) && !models.includes(pyStr(a.model))) models.push(pyStr(a.model));
    return event(time(a, image, 'timestamp_sec', 'timestamp'), measures, { frame: frameOf(a, image) });
  });
  return { events, annotatedModel: models.join(' + ') || null };
}

// Python's round(x, n) on a value that came from adding two parsed decimals.
const round = (x: number, digits: number) => Number(x.toFixed(digits));

function readRttm(source: string): FileContents {
  const events: Event[] = [];
  source.split(/\r\n|\r|\n/).forEach((line, i) => {
    const fields = line.trim().split(/\s+/).filter(Boolean);
    if (!fields.length || fields[0] !== 'SPEAKER') return;
    const start = Number(fields[3]);
    const duration = Number(fields[4]);
    const speaker = fields[7];
    if (fields.length < 8 || Number.isNaN(start) || Number.isNaN(duration) || !fields[3] || !fields[4]) {
      throw new ReaderError(`unreadable: line ${i + 1} is not RTTM`);
    }
    // RTTM gives a duration; keep the sum to RTTM's own precision.
    events.push(event(start, [measure('speaker_turn', speaker)], { end: round(start + duration, 6), track: speaker }));
  });
  return { events, annotatedModel: null };
}

const CUE_TIME = /^\s*((?:\d+:)?\d{1,2}:\d{2}\.\d{3})\s+-->\s+((?:\d+:)?\d{1,2}:\d{2}\.\d{3})/;

function seconds(stamp: string): number {
  let total = 0;
  for (const part of stamp.split(':')) total = total * 60 + Number(part);
  return round(total, 3);
}

function readWebvtt(source: string): FileContents {
  const body = source.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n');
  if (!body.startsWith('WEBVTT')) throw new ReaderError('unreadable: not a WebVTT file');
  const events: Event[] = [];
  for (const block of body.split(/\n\s*\n/)) {
    const lines = block.split('\n');
    if (/^(WEBVTT|NOTE|STYLE|REGION)/.test(lines[0])) continue;
    for (const [i, line] of lines.slice(0, 2).entries()) {
      const match = CUE_TIME.exec(line);
      if (match) {
        const cue = lines.slice(i + 1).join('\n').trim();
        events.push(event(seconds(match[1]), [measure('speech', cue || null)], { end: seconds(match[2]) }));
        break;
      }
    }
  }
  return { events, annotatedModel: null };
}

export interface FileKind {
  suffix: string;
  kind: string;
  defaultPipeline: string;
  read: (source: string) => FileContents;
}

export const FILE_KINDS: FileKind[] = [
  { suffix: 'person_tracking.json', kind: 'person', defaultPipeline: 'person_tracking', read: readPerson },
  { suffix: 'face_detections.json', kind: 'face_deepface', defaultPipeline: 'face_analysis', read: readDeepface },
  { suffix: 'openface3_analysis.json', kind: 'face_openface3', defaultPipeline: 'face_openface3_embedding', read: readOpenface3 },
  { suffix: 'scene_detection.json', kind: 'scene', defaultPipeline: 'scene_detection', read: readScene },
  { suffix: 'vlm_annotation.json', kind: 'vlm', defaultPipeline: 'vlm_annotation', read: readVlm },
  { suffix: 'speaker_diarization.rttm', kind: 'rttm', defaultPipeline: 'speaker_diarization', read: readRttm },
  { suffix: 'speech_recognition.vtt', kind: 'webvtt', defaultPipeline: 'speech_recognition', read: readWebvtt },
];

const RESTATING_SUFFIXES = ['person_tracks.json', 'openface3_detailed.json'];
const PIPELINE_NAMES = new Set([...FILE_KINDS.map((k) => k.defaultPipeline), 'audio_processing']);
const EXPORT_SUFFIXES = ['.csv', '.parquet', '.eaf', '.bib'];
const VIDEO_SUFFIXES = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'];
const IGNORED_NAMES = new Set(['run.json', 'job_metadata.json', 'methods.md']);
export const COMPANION_SUFFIX = '.provenance.json';

/**
 * `{kind, stem}` for a result file; `null` for a file that isn't a result (a
 * video, run.json, an export, a provenance companion); `{kind: null, stem: ''}`
 * for a file no reader knows.
 */
export function classify(name: string): { kind: FileKind | null; stem: string } | null {
  const lower = name.toLowerCase();
  const dot = name.lastIndexOf('.');
  const ext = dot > 0 ? name.slice(dot) : '';
  const stem = dot > 0 ? name.slice(0, dot) : name;
  if (
    name.startsWith('.') ||
    lower.endsWith(COMPANION_SUFFIX) ||
    VIDEO_SUFFIXES.some((s) => lower.endsWith(s)) ||
    EXPORT_SUFFIXES.some((s) => lower.endsWith(s)) ||
    IGNORED_NAMES.has(name) ||
    (ext === '.json' && PIPELINE_NAMES.has(stem)) ||
    RESTATING_SUFFIXES.some((s) => lower.endsWith(`_${s}`))
  ) {
    return null;
  }
  for (const kind of FILE_KINDS) {
    if (name.endsWith(`_${kind.suffix}`)) return { kind, stem: name.slice(0, -kind.suffix.length - 1) };
  }
  return { kind: null, stem: '' };
}

const VTT_MARKER = 'NOTE videoannotator-provenance ';

/** The provenance record stamped in or beside a file, as `provenance.read_record` reads it. */
export function readRecord(file: SourceFile, companion?: SourceFile): Json | null {
  try {
    let record: unknown = null;
    if (companion) {
      record = JSON.parse(companion.text);
    } else if (file.name.toLowerCase().endsWith('.json')) {
      const data = JSON.parse(file.text);
      record = isDict(data) ? data.provenance : null;
    } else if (file.name.toLowerCase().endsWith('.vtt')) {
      const block = file.text.split('\n\n').find((b) => b.startsWith(VTT_MARKER));
      record = block ? JSON.parse(block.slice(VTT_MARKER.length)) : null;
    }
    return isDict(record) ? record : null;
  } catch {
    return null;
  }
}

function pipelineOf(kind: FileKind, provenance: Json | null): string {
  const recorded = provenance?.pipeline;
  const name = isDict(recorded) ? recorded.name : null;
  return typeof name === 'string' && name ? name : kind.defaultPipeline;
}

function modelOf(provenance: Json | null, contents: FileContents): string {
  const models = provenance?.models;
  if (Array.isArray(models) && models.length) {
    const names = models.filter((m) => isDict(m) && truthy(m.name)).map((m) => pyStr((m as Json).name));
    if (names.length) return names.join(' + ');
  }
  if (provenance === null && contents.annotatedModel) return contents.annotatedModel;
  return NOT_RECORDED;
}

function versionOf(provenance: Json | null): string {
  const version = provenance?.videoannotator_version;
  return truthy(version) ? pyStr(version) : NOT_RECORDED;
}

const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

export interface ReadFile {
  name: string;
  pipeline: string;
  provenance: Json | null;
  contents: FileContents;
}

/** Every readable result among `files`, in the order the exporter reads them. */
export function readFiles(files: SourceFile[]): { read: ReadFile[]; skipped: SkippedFile[] } {
  const byName = new Map(files.map((f) => [f.name, f]));
  const results: { file: SourceFile; kind: FileKind }[] = [];
  const skipped: SkippedFile[] = [];
  for (const file of [...files].sort((a, b) => byCodePoint(a.name, b.name))) {
    const found = classify(file.name);
    if (found?.kind) results.push({ file, kind: found.kind });
    else if (found) skipped.push({ file: file.name, reason: 'no reader for this file' });
  }
  results.sort((a, b) => FILE_KINDS.indexOf(a.kind) - FILE_KINDS.indexOf(b.kind));

  const read: ReadFile[] = [];
  for (const { file, kind } of results) {
    const provenance = readRecord(file, byName.get(file.name + COMPANION_SUFFIX));
    try {
      read.push({ name: file.name, pipeline: pipelineOf(kind, provenance), provenance, contents: kind.read(file.text) });
    } catch (e) {
      if (!(e instanceof ReaderError)) throw e;
      skipped.push({ file: file.name, reason: e.message });
    }
  }
  return { read, skipped };
}

/** One video's rows in table order, from what `readFiles` read. */
export function videoRows(read: ReadFile[], video: string, run = ''): Row[] {
  const numbered: { pipeline: string; number: number; event: Event; item: ReadFile }[] = [];
  const counters = new Map<string, number>();
  for (const item of read) {
    for (const e of item.contents.events) {
      const number = (counters.get(item.pipeline) ?? 0) + 1;
      counters.set(item.pipeline, number);
      numbered.push({ pipeline: item.pipeline, number, event: e, item });
    }
  }
  numbered.sort(
    (a, b) => a.event.start - b.event.start || byCodePoint(a.pipeline, b.pipeline) || a.number - b.number,
  );

  const rows: Row[] = [];
  for (const { pipeline, number, event: e, item } of numbered) {
    const model = modelOf(item.provenance, item.contents);
    const version = versionOf(item.provenance);
    for (const m of e.measures) {
      const box = m.measure === 'person' || m.measure === 'face' ? e.box : null;
      rows.push({
        run,
        video,
        pipeline,
        event: number,
        track: e.track,
        start: e.start,
        end: e.end,
        frame: e.frame,
        measure: m.measure,
        label: m.label,
        value: m.value,
        confidence: m.confidence,
        x: box ? box[0] : null,
        y: box ? box[1] : null,
        width: box ? box[2] : null,
        height: box ? box[3] : null,
        model,
        videoannotator_version: version,
        file: item.name,
        author: '',
      });
    }
  }
  return rows;
}

/** The table rows `files` make for one video (the exporter's `video_rows`). */
export function rowsFromFiles(files: SourceFile[], video: string, run = ''): Row[] {
  return videoRows(readFiles(files).read, video, run);
}
