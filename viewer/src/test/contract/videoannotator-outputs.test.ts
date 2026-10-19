/**
 * Contract test: the viewer reads what VideoAnnotator writes.
 *
 * The fixtures are real job outputs (tests/fixtures/viewer_contract/README.md
 * says where they came from). If the server changes an output format, or the
 * viewer changes how it detects or parses one, this fails in CI instead of in
 * a user's results view.
 */
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { parseElanFile } from '@/lib/parsers/elan';
import { detectFileType, mergeAnnotationData } from '@/lib/parsers/merger';

const CURRENT = path.resolve(__dirname, '../../../../tests/fixtures/viewer_contract');
// The same outputs as made before spec 017 added provenance records.
const LEGACY = path.join(CURRENT, 'legacy');

// jsdom's File drops Node Buffers (size 0), so read fixtures as strings.
function fixtureIn(dir: string, name: string): File {
  return new File([readFileSync(path.join(dir, name), 'utf8')], name);
}

const outputsIn = (dir: string) =>
  readdirSync(dir).filter((name) => name.startsWith('demo_clip_')).sort();

const EXPECTED_TYPE: Record<string, string> = {
  'demo_clip_face_detections.json': 'face_analysis',
  'demo_clip_openface3_analysis.json': 'openface3_faces',
  'demo_clip_openface3_detailed.json': 'openface3_faces',
  'demo_clip_person_tracking.json': 'person_tracking',
  'demo_clip_person_tracks.json': 'unknown',
  'demo_clip_scene_detection.json': 'scene_detection',
  'demo_clip_speaker_diarization.rttm': 'speaker_diarization',
  'demo_clip_speech_recognition.vtt': 'speech_recognition',
  'demo_clip_vlm_annotation.json': 'vlm_annotation',
};

// Re-captured with provenance (2026-10-04, faces 2026-10-18); the VLM fixture
// comes from another take and predates it (README).
const RECORDED_TRACKS = [
  'face_analysis',
  'person_tracking',
  'openface3_faces',
  'scene_detection',
  'speech_recognition',
  'speaker_diarization',
] as const;

// As useZipDownloader does: detect each file in archive order, drop unknowns,
// merge. No sorting, so the first file detected as a type is the one shown.
async function loadLikeArtifactsZipFrom(dir: string) {
  const detected = [];
  for (const name of outputsIn(dir)) {
    const result = await detectFileType(fixtureIn(dir, name));
    if (result.type !== 'unknown') detected.push(result);
  }
  detected.push({ file: new File(['video'], 'demo_clip.mp4', { type: 'video/mp4' }), type: 'video' as const, confidence: 1 });
  return mergeAnnotationData(detected);
}

function fakeVideoMetadata() {
  vi.spyOn(console, 'log').mockImplementation(() => {});
  // jsdom can't decode video; report metadata as soon as a source is set.
  const createElement = document.createElement.bind(document);
  vi.spyOn(document, 'createElement').mockImplementation((tag: string, options?: ElementCreationOptions) => {
    const element = createElement(tag, options);
    if (tag === 'video') {
      Object.defineProperties(element, {
        duration: { value: 20 },
        videoWidth: { value: 1280 },
        videoHeight: { value: 720 },
        src: { set: () => queueMicrotask(() => (element as HTMLVideoElement).onloadedmetadata?.(new Event('loadedmetadata'))) },
      });
    }
    return element;
  });
  URL.createObjectURL = vi.fn(() => 'blob:video');
  URL.revokeObjectURL = vi.fn();
}

describe.each([
  ['current', CURRENT],
  ['legacy (no provenance)', LEGACY],
])('VideoAnnotator outputs → viewer: %s', (setName, FIXTURES) => {
  const OUTPUTS = outputsIn(FIXTURES);
  const fixture = (name: string) => fixtureIn(FIXTURES, name);
  const loadLikeArtifactsZip = () => loadLikeArtifactsZipFrom(FIXTURES);
  const isLegacy = FIXTURES === LEGACY;

  beforeEach(fakeVideoMetadata);

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('has a fixture for every expected output', () => {
    expect(OUTPUTS).toEqual(expect.arrayContaining(Object.keys(EXPECTED_TYPE)));
  });

  it.each(Object.entries(EXPECTED_TYPE))('detects %s as %s', async (name, type) => {
    const detected = await detectFileType(fixture(name));
    expect(detected.type).toBe(type);
  });

  it('merges a job\'s artifacts into every annotation track', async () => {
    const { data, metadata } = await loadLikeArtifactsZip();

    expect(metadata.warnings).toEqual([]);
    expect(data.person_tracking?.length).toBeGreaterThan(0);
    // OpenFace's COCO export also has keypoints and bboxes, and sorts first; it
    // must not take person tracking's place.
    const personAnnotations = JSON.parse(readFileSync(path.join(FIXTURES, 'demo_clip_person_tracking.json'), 'utf8')).annotations;
    expect(data.person_tracking).toHaveLength(personAnnotations.length);
    expect(data.face_analysis?.length).toBeGreaterThan(0);
    expect(data.openface3_faces?.length).toBeGreaterThan(0);
    expect(data.scene_detection?.length).toBeGreaterThan(0);
    expect(data.speech_recognition?.length).toBeGreaterThan(0);
    expect(data.speaker_diarization?.length).toBeGreaterThan(0);
    expect(data.vlm_annotations?.length).toBeGreaterThan(0);
  });

  it('parses timestamps and boxes into the shapes the overlays draw', async () => {
    const { data } = await loadLikeArtifactsZip();

    for (const track of [data.person_tracking!, data.face_analysis!]) {
      expect(track[0].bbox).toHaveLength(4);
      expect(typeof track[0].timestamp).toBe('number');
    }
    const face = data.openface3_faces![0];
    expect(face.bbox).toBeDefined();
    expect(typeof face.timestamp).toBe('number');

    const cue = data.speech_recognition![0];
    expect(cue.endTime).toBeGreaterThan(cue.startTime);
    const turn = data.speaker_diarization![0];
    expect(turn.end_time).toBeGreaterThan(turn.start_time);
  });

  it('labels every track with what made it, and never invents it', async () => {
    const { data } = await loadLikeArtifactsZip();
    for (const track of RECORDED_TRACKS) {
      const info = data.provenance?.[track];
      if (isLegacy) {
        expect(info?.kind, track).not.toBe('recorded');
      } else {
        expect(info?.kind, track).toBe('recorded');
        if (info?.kind === 'recorded') {
          expect(info.record.videoannotator_version).toBeTruthy();
          expect(info.record.models?.length, track).toBeGreaterThan(0);
        }
      }
    }
    // Older COCO files carry only VideoAnnotator's version.
    if (isLegacy) {
      expect(data.provenance?.face_analysis).toEqual({ kind: 'partial', videoannotatorVersion: '1.5.0' });
    }
    expect(data.provenance?.vlm_annotations?.kind).toBe('partial');
  });

  it.runIf(!isLegacy)('parses to the same annotations as the legacy fixtures', async () => {
    const strip = (d: Awaited<ReturnType<typeof loadLikeArtifactsZipFrom>>['data']) => {
      const { provenance: _p, metadata: _m, ...rest } = d;
      return rest;
    };
    const { data: current } = await loadLikeArtifactsZipFrom(CURRENT);
    const { data: legacy } = await loadLikeArtifactsZipFrom(LEGACY);
    // Exact where the outputs are the same; the default models changed in
    // spec 030 (README), so for the rest compare their shape.
    expect(current.vlm_annotations).toEqual(legacy.vlm_annotations);
    for (const track of ['person_tracking', 'openface3_faces', 'scene_detection', 'speech_recognition'] as const) {
      expect(current[track]?.length, track).toBe(legacy[track]?.length);
    }
    // Legacy faces came from another take (Haar cascade era), and 3.1 split the
    // mother's last turn in two where community-1 doesn't; both must parse.
    for (const track of ['face_analysis', 'speaker_diarization'] as const) {
      expect(current[track]?.length, track).toBeGreaterThan(0);
      expect(legacy[track]?.length, track).toBeGreaterThan(0);
    }
    expect(Object.keys(strip(current)).sort()).toEqual(Object.keys(strip(legacy)).sort());
  });
});

// Spec 025: VideoAnnotator's ELAN export opens in the viewer, before and after
// a coder has added their own tier in ELAN (FR-022).
const EXPORTED = path.join(CURRENT, 'exported');
const EXPORTED_TIERS = [
  'scene_detection scenes',
  'speech_recognition speech',
  'speaker_diarization SPEAKER_00',
  'person_tracking person_demo_clip_001 (parent)',
  'person_tracking person_demo_clip_002 (parent)',
  'face_analysis faces',
  'face_openface3_embedding faces',
  'vlm_annotation labels',
];

describe('VideoAnnotator ELAN export → viewer', () => {
  it.each([
    ['as exported', 'demo_clip.eaf', EXPORTED_TIERS],
    ['after editing in ELAN', 'demo_clip_edited_in_elan.eaf', [...EXPORTED_TIERS, 'Infant gaze']],
  ])('opens %s and shows every tier', async (_label, name, tiers) => {
    const file = fixtureIn(EXPORTED, name);
    expect((await detectFileType(file)).type).toBe('elan_ground_truth');
    // The parser the merger uses for an .eaf (merger.ts, elan_ground_truth).
    const annotations = await parseElanFile(file);
    expect([...new Set(annotations.map((a) => a.tier))].sort()).toEqual([...tiers].sort());
    const speech = annotations.filter((a) => a.tier === 'speech_recognition speech');
    expect(speech[0]).toMatchObject({ value: 'Ready, baby girl?', startSec: 0, endSec: 1.34 });
  });
});

// Spec 027: a run from Ask AI, stopped after 4 of 7 samples, writes the same
// annotations with additive fields (frame_times, status, vlm_progress, vlm_skipped).
describe('a stopped Ask AI run → viewer', () => {
  const ASK_AI = path.join(CURRENT, 'ask_ai');
  beforeEach(fakeVideoMetadata);
  afterEach(() => vi.restoreAllMocks());

  it('reads the same annotations as a run without the new fields', async () => {
    const stopped = fixtureIn(ASK_AI, 'demo_clip_vlm_annotation_stopped.json');
    const detected = await detectFileType(stopped);
    expect(detected.type).toBe('vlm_annotation');
    const video = { file: new File(['video'], 'demo_clip.mp4', { type: 'video/mp4' }), type: 'video' as const, confidence: 1 };
    const { data, metadata } = await mergeAnnotationData([detected, video]);
    expect(metadata.warnings).toEqual([]);
    expect(data.vlm_annotations).toHaveLength(3);

    const plain = await mergeAnnotationData([await detectFileType(fixtureIn(CURRENT, 'demo_clip_vlm_annotation.json')), video]);
    expect(data.vlm_annotations?.slice(0, 2)).toEqual(plain.data.vlm_annotations?.slice(0, 2));
  });
});

// Spec 028: every measure VideoAnnotator writes has a lane of its own on the
// Video page; a generic `other:` lane is only for outputs the viewer doesn't know.
describe('VideoAnnotator outputs → Video page lanes', () => {
  it.each([
    ['current', CURRENT],
    ['legacy', LEGACY],
    ['stopped Ask AI run', path.join(CURRENT, 'ask_ai')],
  ])('builds lanes from the %s fixtures with no generic lane', async (_name, dir) => {
    const { rowsFromFiles } = await import('@/lib/table/rows');
    const { buildLanes } = await import('@/lib/lanes');
    const files = readdirSync(dir)
      .filter((name) => name.startsWith('demo_clip_'))
      // Named as a run writes it (the fixture's name marks it as the stopped one).
      .map((name) => ({ name: name.replace('_stopped', ''), text: readFileSync(path.join(dir, name), 'utf8') }));
    const lanes = buildLanes(rowsFromFiles(files, 'demo_clip'), []);
    expect(lanes.length).toBeGreaterThan(0);
    expect(lanes.filter((l) => l.kind.startsWith('other:')).map((l) => l.kind)).toEqual([]);
  });
});
