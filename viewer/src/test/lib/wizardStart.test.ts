import { describe, expect, it } from 'vitest';
import { askWizardStart, wizardStartOf, wizardState, type WizardStart } from '@/lib/wizardStart';
import { samplingFromConfig, samplingToConfig } from '@/lib/sampling';

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

describe('askWizardStart', () => {
  it("starts run setup with Ask AI's question, model and sampling", () => {
    const config = samplingToConfig({ interval: 2, frames: 3, spacing: 0.5 });
    expect(askWizardStart('Touch?', 'qwen3.5:9b', config)).toEqual({
      mode: 'settings',
      label: 'Ask AI (qwen3.5:9b)',
      selectedPipelines: ['vlm_annotation'],
      config: {
        vlm_annotation: {
          frame_interval_sec: 2,
          start_sec: 0,
          end_sec: null,
          moments_sec: null,
          sampling_mode: 'frame_burst',
          burst_offsets: [-1, 0, 1],
          burst_spacing_sec: 0.5,
          prompt: 'Touch?',
          model: 'qwen3.5:9b',
        },
      },
    });
  });

  it('reads the same sampling back from those settings', () => {
    const sampling = { interval: 2, start: 10, end: 20, moments: null, frames: 3, spacing: 0.5 };
    expect(samplingFromConfig(samplingToConfig(sampling))).toEqual(sampling);
    expect(samplingFromConfig({ sampling_mode: 'frame_burst', frame_interval_sec: 4 })).toMatchObject({ frames: 5, spacing: 4 });
  });
});
