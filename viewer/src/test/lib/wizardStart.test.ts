import { describe, expect, it } from 'vitest';
import { wizardStartOf, wizardState, type WizardStart } from '@/lib/wizardStart';

describe('wizardStartOf', () => {
  it('reads the typed state', () => {
    const start: WizardStart = { mode: 'settings', label: 'clip.mp4', selectedPipelines: ['a'] };
    expect(wizardStartOf(wizardState(start))).toEqual(start);
  });

  it('maps the older retry state to editing and running again', () => {
    expect(
      wizardStartOf({ retryJobId: 'j1', retryJobPipelines: ['a'], retryJobConfig: { a: {} }, retryJobVideoFilename: 'v.mp4' }),
    ).toEqual({ mode: 'rerun', jobId: 'j1', label: 'v.mp4', selectedPipelines: ['a'], config: { a: {} } });
  });

  it('maps the Datasets page start', () => {
    expect(wizardStartOf({ startFromDataset: { id: 'd', name: 'Pilot' } })).toEqual({ mode: 'dataset', datasetId: 'd', name: 'Pilot' });
  });

  it.each([null, undefined, {}, 'x', { usr: 1 }])('anything else is a blank start (%#)', (state) => {
    expect(wizardStartOf(state)).toBeNull();
  });
});
