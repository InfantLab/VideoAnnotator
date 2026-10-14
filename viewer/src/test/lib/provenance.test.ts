import { describe, expect, it } from 'vitest';
import {
  attributionLabel,
  provenanceFromCompanion,
  provenanceFromJSON,
  provenanceFromWebVTT,
} from '@/lib/provenance';

const record = {
  schema_version: 1,
  pipeline: { name: 'speech_recognition', sub_pipeline: null },
  videoannotator_version: '1.6.0',
  models: [{ name: 'whisper base', source: 'openai-whisper', revision: 'ed3a', revision_kind: 'sha256' }],
  created_at: '2026-10-04T12:00:00+00:00',
  settings: { prompt: 'a --> b' },
};

describe('provenance', () => {
  it('reads a JSON output’s top-level record', () => {
    expect(provenanceFromJSON({ annotations: [], provenance: record })).toEqual({ kind: 'recorded', record });
  });

  it('an older COCO file says only its VideoAnnotator version', () => {
    expect(provenanceFromJSON({ info: { version: '1.5.0' }, annotations: [] })).toEqual({
      kind: 'partial',
      videoannotatorVersion: '1.5.0',
    });
  });

  it.each([{}, [], null, { provenance: { pipeline: 'x' } }])('anything else is not recorded (%#)', (data) => {
    expect(provenanceFromJSON(data)).toEqual({ kind: 'none' });
  });

  it('reads a WebVTT NOTE block, with --> escaped inside the JSON', () => {
    const json = JSON.stringify(record).replace('-->', '--\\u003e');
    const vtt = `WEBVTT\n\nNOTE videoannotator-provenance ${json}\n\n1\n00:00:00.000 --> 00:00:01.000\nHi\n`;
    const info = provenanceFromWebVTT(vtt);
    expect(info.kind).toBe('recorded');
    expect(info.kind === 'recorded' && info.record.settings).toEqual({ prompt: 'a --> b' });
    expect(provenanceFromWebVTT('WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHi\n')).toEqual({ kind: 'none' });
  });

  it('reads a companion file, and rejects a broken one', () => {
    expect(provenanceFromCompanion(JSON.stringify(record)).kind).toBe('recorded');
    expect(provenanceFromCompanion('{ nope')).toEqual({ kind: 'none' });
  });

  it.each([
    [{ kind: 'recorded', record }, 'speech_recognition · VideoAnnotator 1.6.0'],
    [{ kind: 'partial', videoannotatorVersion: '1.5.0' }, 'person_tracking · VideoAnnotator 1.5.0 · rest not recorded'],
    [{ kind: 'none' }, 'person_tracking · version not recorded'],
    [undefined, 'person_tracking · version not recorded'],
    [{ kind: 'ground_truth', fileName: 'truth.eaf' }, 'Ground truth · truth.eaf'],
  ] as const)('labels %j', (info, label) => {
    expect(attributionLabel(info as Parameters<typeof attributionLabel>[0], 'person_tracking')).toBe(label);
  });
});
