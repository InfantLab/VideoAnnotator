import { describe, expect, it } from 'vitest';
import {
  describeFileType,
  detectFileType,
  detectFileTypes,
  validateFileSet,
  validateFileSize,
} from '@/lib/fileDetection';

const json = (data: unknown, name = 'data.json') =>
  new File([JSON.stringify(data)], name, { type: 'application/json' });

describe('detectFileType', () => {
  it.each([
    ['clip.mp4', 'video/mp4', 'video'],
    ['clip.mkv', '', 'video'],
    ['clip.wav', 'audio/wav', 'audio'],
    ['speech.rttm', '', 'speaker_diarization'],
    ['notes.xyz', '', 'unknown'],
  ])('%s is %s by extension', async (name, type, expected) => {
    expect((await detectFileType(new File(['SPEAKER x'], name, { type }))).type).toBe(expected);
  });

  it('checks a .vtt file is WebVTT', async () => {
    const good = await detectFileType(new File(['WEBVTT\n\n'], 's.vtt'));
    const bad = await detectFileType(new File(['hello'], 's.vtt'));
    expect(good).toMatchObject({ type: 'speech_recognition', confidence: 0.9 });
    expect(bad.confidence).toBeLessThan(0.5);
  });

  it.each([
    [{ annotations: [{ keypoints: [1, 2, 2], bbox: [0, 0, 1, 1] }] }, 'person_tracking'],
    [{ annotations: [{ openface3: {}, keypoints: [], bbox: [0, 0, 1, 1] }] }, 'openface3_faces'],
    [{ metadata: { pipeline: 'openface3' }, faces: [] }, 'openface3_faces'],
    [{ annotations: [{ reasoning: '', sampling_mode: 'single_frame' }] }, 'vlm_annotation'],
    [{ annotations: [{ scene_type: 'nursery' }] }, 'scene_detection'],
    [[{ start_time: 0, end_time: 5 }], 'scene_detection'],
    [{ scenes: [] }, 'scene_detection'],
    [{ results: [{ face_id: 0, bbox: [0, 0, 1, 1] }] }, 'face_analysis'],
    [{ video_path: 'v.mp4', pipeline_results: {}, config: {} }, 'complete_results'],
  ])('classifies JSON by its fields (%#)', async (data, expected) => {
    expect((await detectFileType(json(data))).type).toBe(expected);
  });

  it('does not take a person ID summary for frames to draw', async () => {
    expect((await detectFileType(json({ person_tracks: [] }))).type).toBe('unknown');
  });

  it('falls back to the VideoAnnotator file name when the JSON says nothing', async () => {
    const empty = { info: {}, images: [], annotations: [] };
    expect(await detectFileType(json(empty, 'clip_person_tracking.json'))).toMatchObject({
      type: 'person_tracking',
      confidence: 0.5,
    });
    expect((await detectFileType(json(empty, 'clip.json'))).type).toBe('unknown');
  });

  it('a large file is classified from all of it, not a prefix', async () => {
    const big = {
      info: { description: 'VideoAnnotator COCO Export' },
      images: Array.from({ length: 2000 }, (_, i) => ({ id: i, file_name: `frame_${i}.jpg` })),
      annotations: [{ openface3: { action_units: {} }, keypoints: [], bbox: [0, 0, 1, 1] }],
    };
    expect((await detectFileType(json(big))).type).toBe('openface3_faces');
  });

  it.each(['{ "broken": ', '', '\u0000\u0001'])('invalid JSON is unknown (%#)', async (text) => {
    const result = await detectFileType(new File([text], 'x.json', { type: 'application/json' }));
    expect(result).toMatchObject({ type: 'unknown', confidence: 0 });
  });
});

describe('validateFileSet', () => {
  it('needs a video', async () => {
    const result = validateFileSet(await detectFileTypes([json({})]));
    expect(result.valid).toBe(false);
  });

  it('counts JSON results as annotations (it used to warn "no annotation files")', async () => {
    const result = validateFileSet(
      await detectFileTypes([
        new File([''], 'clip.mp4', { type: 'video/mp4' }),
        json({ annotations: [{ keypoints: [], bbox: [0, 0, 1, 1] }] }),
      ]),
    );
    expect(result).toEqual({ valid: true, missing: [], warnings: [] });
  });

  it('counts ELAN ground truth as annotations', async () => {
    const result = validateFileSet([
      { file: new File([''], 'clip.mp4'), type: 'video', confidence: 1 },
      { file: new File([''], 'truth.eaf'), type: 'elan_ground_truth', confidence: 1 },
    ]);
    expect(result.warnings).toEqual([]);
  });

  it('warns about files it cannot identify', async () => {
    const result = validateFileSet(
      await detectFileTypes([new File([''], 'clip.mp4', { type: 'video/mp4' }), new File([''], 'x.xyz')]),
    );
    expect(result.warnings).toContain('1 file(s) could not be identified. Check file formats.');
  });
});

describe('validateFileSize', () => {
  const sized = (name: string, size: number) => {
    const file = new File([''], name);
    Object.defineProperty(file, 'size', { value: size });
    return file;
  };

  it('caps video and audio', () => {
    expect(validateFileSize(sized('a.mp4', 600 * 1024 * 1024), 'video').valid).toBe(false);
    expect(validateFileSize(sized('a.mp4', 100 * 1024 * 1024), 'video').valid).toBe(true);
    expect(validateFileSize(sized('a.wav', 200 * 1024 * 1024), 'audio').valid).toBe(false);
  });

  it('does not cap JSON results (OpenFace files run to tens of MB)', () => {
    expect(validateFileSize(sized('a.json', 50 * 1024 * 1024), 'openface3_faces').valid).toBe(true);
  });
});

it('describes each type', () => {
  expect(describeFileType('openface3_faces')).toBe('OpenFace3 Analysis (JSON)');
  expect(describeFileType('unknown')).toBe('Unknown File Type');
});
