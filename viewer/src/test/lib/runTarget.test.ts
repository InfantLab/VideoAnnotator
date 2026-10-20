import { describe, expect, it } from 'vitest';

import { projectOf, runStartOf, runState, targetLabel } from '@/lib/runTarget';
import { wizardState } from '@/lib/wizardStart';

describe('runStartOf', () => {
  it('reads a target passed as router state', () => {
    const start = runState({ target: { kind: 'project', id: 'd1', name: 'QS' }, workflowId: 'w1' });
    expect(runStartOf(start)).toEqual({ target: { kind: 'project', id: 'd1', name: 'QS' }, settings: undefined, workflowId: 'w1' });
  });

  it('reads the old wizard shapes', () => {
    expect(runStartOf(wizardState({ mode: 'rerun', jobId: 'j1', label: 'clip.mp4', selectedPipelines: ['a'], config: { a: {} } }))).toEqual({
      target: { kind: 'rerun', runId: 'j1', label: 'clip.mp4', single: true },
      settings: { selectedPipelines: ['a'], config: { a: {} } },
    });
    expect(runStartOf(wizardState({ mode: 'rerunBatch', batchId: 'b1', label: 'Pilot', videoCount: 3 })).target).toEqual({
      kind: 'rerun',
      runId: 'b1',
      label: 'Pilot',
      single: false,
      videoCount: 3,
    });
    expect(runStartOf(wizardState({ mode: 'dataset', datasetId: 'd1', name: 'QS' })).target).toEqual({ kind: 'project', id: 'd1', name: 'QS' });
    expect(runStartOf({ retryJobId: 'j9', retryJobVideoFilename: 'x.mp4' }).target).toMatchObject({ kind: 'rerun', runId: 'j9' });
    const ask = runStartOf(wizardState({ mode: 'settings', label: 'Ask AI (qwen)', selectedPipelines: ['vlm_annotation'] }));
    expect(ask.target).toBeNull();
    expect(ask.settings?.selectedPipelines).toEqual(['vlm_annotation']);
  });

  it('is empty for nothing', () => {
    expect(runStartOf(null)).toEqual({ target: null });
    expect(runStartOf({})).toEqual({ target: null });
  });
});

describe('a target', () => {
  it('names its project and itself', () => {
    expect(projectOf({ kind: 'project', id: 'd1', name: 'QS' })).toBe('d1');
    expect(projectOf({ kind: 'videos', paths: ['/a'], label: '1 video', projectId: 'd2' })).toBe('d2');
    expect(projectOf({ kind: 'folder', path: '/data', recursive: false })).toBeNull();
    expect(targetLabel({ kind: 'folder', path: '/data/pk', recursive: false })).toBe('/data/pk');
  });
});
