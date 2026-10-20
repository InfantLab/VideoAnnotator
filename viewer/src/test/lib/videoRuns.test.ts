import { describe, expect, it } from 'vitest';

import { buildLanes } from '@/lib/lanes';
import { MISSING_RESULTS, defaultSelection, failedPipelines } from '@/lib/videoRuns';
import type { VideoRun } from '@/types/video';

const result = (name: string) => ({ name, kind: 'result', download_url: `/files/${name}` });

const run = (pipelines: VideoRun['pipelines']): VideoRun =>
  ({
    job_id: 'j1',
    status: 'completed',
    created_at: '2026-10-18T10:00:00Z',
    completed_at: '2026-10-18T10:05:00Z',
    batch_id: null,
    dataset_id: null,
    run_name: null,
    pipelines,
  }) as VideoRun;

describe('pipelines with nothing to show', () => {
  // A run whose results folder was deleted (or lost with a container) still lists its
  // pipelines as completed; they used to vanish from the page without a word.
  it('names a completed pipeline whose result files are gone', () => {
    const runs = [
      run([
        { name: 'scene_detection', status: 'completed', error: null, files: [result('clip_scenes.json')] },
        { name: 'person_tracking', status: 'completed', error: null, files: [] },
        { name: 'vlm_annotation', status: 'failed', error: 'model not found', files: [] },
      ]),
    ];
    const selection = defaultSelection(runs);
    expect(selection).toEqual({ scene_detection: 'j1' });
    expect(failedPipelines(runs, selection)).toEqual([
      { run: 'j1', pipeline: 'person_tracking', error: MISSING_RESULTS, missing: true },
      { run: 'j1', pipeline: 'vlm_annotation', error: 'model not found' },
    ]);
  });

  it('gives each its own track, saying which failed and which lost its files', () => {
    const runs = [
      run([
        { name: 'person_tracking', status: 'completed', error: null, files: [] },
        { name: 'vlm_annotation', status: 'failed', error: 'model not found', files: [] },
      ]),
    ];
    const lanes = buildLanes([], [{ id: 'j1', label: 'run 18 Oct', date: null }], failedPipelines(runs, {}));
    expect(lanes.map((l) => [l.pipeline, l.status])).toEqual(
      expect.arrayContaining([
        ['person_tracking', 'missing'],
        ['vlm_annotation', 'failed'],
      ]),
    );
  });
});
