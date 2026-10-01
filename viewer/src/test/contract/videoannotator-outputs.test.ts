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

const FIXTURES = path.resolve(__dirname, '../../../../tests/fixtures/viewer_contract');

// jsdom's File drops Node Buffers (size 0), so read fixtures as strings.
function fixture(name: string): File {
  return new File([readFileSync(path.join(FIXTURES, name), 'utf8')], name);
}

const OUTPUTS = readdirSync(FIXTURES).filter((name) => name.startsWith('demo_clip_')).sort();

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

// As useZipDownloader does: detect each file in archive order, drop unknowns,
// merge. No sorting, so the first file detected as a type is the one shown.
async function loadLikeArtifactsZip() {
  const detected = [];
  for (const name of OUTPUTS) {
    const result = await detectFileType(fixture(name));
    if (result.type !== 'unknown') detected.push(result);
  }
  detected.push({ file: new File(['video'], 'demo_clip.mp4', { type: 'video/mp4' }), type: 'video' as const, confidence: 1 });
  return mergeAnnotationData(detected);
}

describe('VideoAnnotator outputs → viewer', () => {
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
});
