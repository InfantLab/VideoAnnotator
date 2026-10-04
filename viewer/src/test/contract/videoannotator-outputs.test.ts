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

// Re-captured with provenance (2026-10-04); the face-detection and VLM fixtures
// come from another take and predate it (README).
const RECORDED_TRACKS = [
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

describe.each([
  ['current', CURRENT],
  ['legacy (no provenance)', LEGACY],
])('VideoAnnotator outputs → viewer: %s', (setName, FIXTURES) => {
  const OUTPUTS = outputsIn(FIXTURES);
  const fixture = (name: string) => fixtureIn(FIXTURES, name);
  const loadLikeArtifactsZip = () => loadLikeArtifactsZipFrom(FIXTURES);
  const isLegacy = FIXTURES === LEGACY;

  beforeEach(() => {
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
  });

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
    expect(data.provenance?.face_analysis).toEqual({ kind: 'partial', videoannotatorVersion: '1.5.0' });
    expect(data.provenance?.vlm_annotations?.kind).toBe('partial');
  });

  it.runIf(!isLegacy)('parses to the same annotations as the legacy fixtures', async () => {
    const strip = (d: Awaited<ReturnType<typeof loadLikeArtifactsZipFrom>>['data']) => {
      const { provenance: _p, metadata: _m, ...rest } = d;
      return rest;
    };
    const { data: current } = await loadLikeArtifactsZipFrom(CURRENT);
    const { data: legacy } = await loadLikeArtifactsZipFrom(LEGACY);
    // Exact where the outputs are; GPU-sampled tracks moved slightly with the
    // 2026-10-04 torch-settings fix (README), so for those compare their shape.
    for (const track of ['speech_recognition', 'speaker_diarization', 'face_analysis', 'vlm_annotations'] as const) {
      expect(current[track], track).toEqual(legacy[track]);
    }
    for (const track of ['person_tracking', 'openface3_faces', 'scene_detection'] as const) {
      expect(current[track]?.length, track).toBe(legacy[track]?.length);
    }
    expect(Object.keys(strip(current)).sort()).toEqual(Object.keys(strip(legacy)).sort());
  });
});
