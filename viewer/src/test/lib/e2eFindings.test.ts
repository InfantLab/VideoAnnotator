// Helpers behind the spec 011 end-to-end fixes: not-ready pipelines kept out of
// a job, first-run setup kept out of time-remaining, failed pipelines named.
import { describe, expect, it } from 'vitest';
import { notReadyReason, partitionSelection } from '@/lib/pipelineExtras';
import { throughputSecondsRemaining, totalDownloadLabel, weightsNotesFor } from '@/lib/runSetup';
import { failedPipelinesOf, isCompletedWithErrors, jobErrorSummary } from '@/lib/jobOutcome';
import type { PipelineDescriptor } from '@/types/pipelines';

const pipeline = (id: string, name: string, readiness?: Partial<PipelineDescriptor['readiness']>): PipelineDescriptor =>
  ({
    id,
    name,
    readiness: readiness
      ? {
          state: 'ready',
          nextAction: 'none',
          extrasGroup: null,
          installJobId: null,
          blockers: [],
          notes: [],
          ...readiness,
        }
      : undefined,
  }) as PipelineDescriptor;

const vlmDown = pipeline('vlm_annotation', 'VLM Frame Annotation', {
  state: 'needs_setup',
  blockers: [{ kind: 'service', name: 'ollama', message: "Can't reach the Ollama server." }],
});
const face = pipeline('face_analysis', 'Face Analysis', {
  notes: [{ kind: 'weights_not_cached', name: 'deepface', message: 'Downloads DeepFace weights on first run.', approxMb: 1100 }],
});

describe('partitionSelection', () => {
  it('leaves out pipelines that are not ready, and says why', () => {
    const { kept, leftOut } = partitionSelection(['face_analysis', 'vlm_annotation'], [face, vlmDown]);
    expect(kept).toEqual(['face_analysis']);
    expect(leftOut).toEqual(["VLM Frame Annotation left out: Can't reach the Ollama server."]);
  });

  it('keeps ids the catalog does not know (the server validates them)', () => {
    expect(partitionSelection(['mystery'], [face]).kept).toEqual(['mystery']);
  });

  it('falls back to the card state when there is no blocker', () => {
    const notInstalled = pipeline('scene_detection', 'Scene', { state: 'not_installed' });
    expect(notReadyReason(notInstalled)).toBe("it isn't installed");
    expect(notReadyReason(face)).toBe('');
  });
});

describe('first-run setup', () => {
  it('collects weights notes for the selected pipelines only', () => {
    const notes = weightsNotesFor([face, vlmDown], ['face_analysis']);
    expect(notes).toHaveLength(1);
    expect(totalDownloadLabel(notes)).toBe('about 1.1 GB');
    expect(weightsNotesFor([face], [])).toEqual([]);
  });

  it('counts weights two pipelines share once', () => {
    const pyannote = { kind: 'weights_not_cached', name: 'pyannote/speaker-diarization-3.1', message: 'x', helpUrl: null, approxMb: 300 };
    const audio = pipeline('audio_processing', 'Audio', { notes: [pyannote] });
    const diar = pipeline('speaker_diarization', 'Diarization', { notes: [pyannote] });
    expect(totalDownloadLabel(weightsNotesFor([audio, diar], ['audio_processing', 'speaker_diarization']))).toBe('about 300 MB');
  });

  it('estimates from videos after the first, ignoring its setup time', () => {
    const t0 = Date.parse('2026-09-25T10:00:00Z');
    const at = (minutes: number) => new Date(t0 + minutes * 60_000).toISOString();
    // First video took 10 min (setup); the next two took 1 min each.
    const jobs = [
      { status: 'completed', completed_at: at(10) },
      { status: 'completed', completed_at: at(11) },
      { status: 'completed', completed_at: at(12) },
      { status: 'running', completed_at: null },
    ];
    expect(throughputSecondsRemaining(jobs, 2)).toBe(120);
  });

  it('has no estimate from the first video alone', () => {
    expect(throughputSecondsRemaining([{ status: 'completed', completed_at: '2026-09-25T10:00:00Z' }], 4)).toBeNull();
  });
});

describe('job outcome', () => {
  it('treats completed with an error message as completed with errors', () => {
    expect(isCompletedWithErrors({ status: 'completed', error_message: 'Completed with errors. Failed pipelines: person_tracking' })).toBe(true);
    expect(isCompletedWithErrors({ status: 'completed', error_message: null })).toBe(false);
    expect(isCompletedWithErrors({ status: 'failed', error_message: 'boom' })).toBe(false);
  });

  it('names failed pipelines with their reasons', () => {
    expect(
      failedPipelinesOf({
        job_id: 'j',
        status: 'completed',
        pipeline_results: {
          face_analysis: { status: 'completed' },
          person_tracking: { status: 'failed', error_message: 'No module named torch' },
        },
      })
    ).toEqual({ person_tracking: 'No module named torch' });
  });

  it('gives a batch row the reasons, without repeating what the badge says', () => {
    expect(
      jobErrorSummary({ status: 'completed', error_message: 'Completed with errors. person_tracking: No module named torch' })
    ).toBe('person_tracking: No module named torch');
    expect(
      jobErrorSummary({ status: 'failed', error_message: 'All pipelines failed. face_analysis: CUDA out of memory' })
    ).toBe('face_analysis: CUDA out of memory');
    expect(jobErrorSummary({ status: 'failed', error_message: 'Video file not found' })).toBe('Video file not found');
    expect(jobErrorSummary({ status: 'completed', error_message: null })).toBeNull();
    expect(jobErrorSummary({ status: 'cancelled', error_message: 'Job cancelled by user request' })).toBeNull();
  });
});
