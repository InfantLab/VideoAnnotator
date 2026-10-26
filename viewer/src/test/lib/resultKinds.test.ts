/** One hue per kind of result; tracks are its shades (viewer design P8, UAT #48). */
import { describe, expect, it } from 'vitest';

import { RESULT_KINDS, kindColour, kindOfLane, kindOfPipeline, trackColours } from '@/lib/resultKinds';

const hueOf = (hsl: string) => Number(/hsl\((\d+)/.exec(hsl)?.[1]);

describe('result kinds', () => {
  it('gives every pipeline and lane a kind, telling DeepFace and OpenFace 3 apart', () => {
    expect(kindOfPipeline('speaker_diarization')?.id).toBe('speakers');
    expect(kindOfLane('faces', 'face_analysis')?.id).toBe('faces');
    expect(kindOfLane('faces', 'face_openface3_embedding')?.id).toBe('face_measures');
    expect(kindOfLane('action_units', null)?.id).toBe('face_measures');
    expect(kindOfLane('person_area', 'person_tracking')?.id).toBe('people');
  });

  it('keeps the tracks of a kind near its hue, and no kind on the state colours', () => {
    const speakers = trackColours(RESULT_KINDS.speakers, ['SPEAKER_01', 'SPEAKER_00', 'SPEAKER_10', 'SPEAKER_2']);
    expect([...speakers.keys()]).toEqual(['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_2', 'SPEAKER_10']);
    for (const colour of speakers.values()) expect(Math.abs(hueOf(colour) - 250)).toBeLessThanOrEqual(14);
    const hues = Object.values(RESULT_KINDS).flatMap((k) => (k.hue === null ? [] : [hueOf(kindColour(k))]));
    expect(new Set(hues).size).toBe(hues.length);
    // Red (failed), green (done) and blue (running) mean state only.
    for (const h of hues) for (const state of [0, 120, 220]) expect(Math.min(Math.abs(h - state), 360 - Math.abs(h - state))).toBeGreaterThanOrEqual(18);
  });
});
