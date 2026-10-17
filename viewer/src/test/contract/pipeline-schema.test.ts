// The schema route's spec 026 fields, as the server sends them today
// (fixture regenerated from the API; tests/api/test_pipeline_guide_api.py
// fails when it goes stale), parse into the viewer's types.
import { afterEach, describe, expect, it, vi } from 'vitest';

import { APIClient, apiClient } from '@/api/client';
import fixture from './pipeline-schema-face_analysis.json';

describe('pipeline schema contract (spec 026)', () => {
  afterEach(() => vi.restoreAllMocks());

  it('maps groups, labels, units and conditions, with nulls as undefined', async () => {
    vi.spyOn(APIClient.prototype, 'getServerInfo').mockResolvedValue({} as never);
    vi.spyOn(APIClient.prototype as unknown as { request: () => Promise<unknown> }, 'request').mockResolvedValue(
      fixture
    );
    const { parameters } = await apiClient.getPipelineSchema('face_analysis');
    const byName = Object.fromEntries(parameters.map((p) => [p.name, p]));

    expect(byName.max_faces).toMatchObject({ group: 'behavioural', label: 'Most faces per frame', unit: 'faces', min: 1 });
    expect(byName.max_faces.max).toBeUndefined();
    expect(byName.scale_factor.appliesWhen).toEqual({ detection_backend: ['opencv'] });
    expect(byName.detect_age.unit).toBeUndefined();
    expect(parameters.every((p) => p.group === 'behavioural' || p.group === 'technical')).toBe(true);
  });
});

describe('pipeline catalog guide fields (spec 026)', () => {
  it('maps the guide and turns nulls into undefined', () => {
    const client = new APIClient();
    const map = (client as unknown as { mapLegacyPipelineResponse: (p: unknown[]) => { pipelines: unknown[] } })
      .mapLegacyPipelineResponse;
    const [withGuide, without] = map.call(client, [
      {
        name: 'face_analysis',
        display_name: 'Face Analysis (DeepFace)',
        description: 'd',
        title: 'Faces: expression, age, gender',
        summary: 'Finds faces.',
        homepage: 'https://github.com/serengil/deepface',
        benchmarks: [{ name: 'WIDER FACE (hard)', measure: 'AP', figure: '91.4%', source: 'https://arxiv.org/abs/1905.00641', note: null }],
        quality_note: null,
        citations: [{ key: 'k', bibtex: '@x{k,}' }],
      },
      { name: 'old', display_name: 'Old', description: 'd', benchmarks: null, title: null },
    ]).pipelines as Record<string, unknown>[];
    expect(withGuide).toMatchObject({ title: 'Faces: expression, age, gender', summary: 'Finds faces.', homepage: 'https://github.com/serengil/deepface' });
    expect((withGuide.benchmarks as { note?: string }[])[0].note).toBeUndefined();
    expect(withGuide.qualityNote).toBeUndefined();
    expect(without.title).toBeUndefined();
    expect(without.benchmarks).toBeUndefined();
  });
});
