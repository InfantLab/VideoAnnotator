import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { ProvenanceAttribution } from '@/components/ProvenanceAttribution';

const record = {
  schema_version: 1,
  pipeline: { name: 'person_tracking', sub_pipeline: null },
  videoannotator_version: '1.6.0',
  models: [
    { name: 'yolo11n-pose.pt', source: 'ultralytics', revision: '869e83fc', revision_kind: 'sha256' },
    { name: 'other', source: 'file', revision: null, revision_kind: 'unknown', revision_note: 'weights file not found' },
  ],
  created_at: '2026-10-04T12:00:00+00:00',
  job_id: 'job-1',
  input: { name: 'clip.mp4', sha256: 'abc' },
  settings: { conf_threshold: 0.4 },
};

describe('ProvenanceAttribution', () => {
  it('names the pipeline and version, and shows the record verbatim on demand', () => {
    const { container } = render(<ProvenanceAttribution info={{ kind: 'recorded', record }} pipeline="person_tracking" />);
    // Split into spans so it wraps after underscores; the text reads whole.
    expect(container.textContent).toContain('person_tracking · VideoAnnotator 1.6.0');
    fireEvent.click(screen.getByRole('button', { name: /Provenance/ }));
    expect(screen.getByText('yolo11n-pose.pt (ultralytics) 869e83fc')).toBeInTheDocument();
    expect(screen.getByText('other (file) revision not recorded: weights file not found')).toBeInTheDocument();
    expect(screen.getByText('2026-10-04T12:00:00+00:00')).toBeInTheDocument();
    expect(screen.getByText('{"conf_threshold":0.4}')).toBeInTheDocument();
  });

  it('says plainly when a file records nothing', () => {
    const { container } = render(<ProvenanceAttribution info={{ kind: 'none' }} pipeline="scene_detection" />);
    expect(container.textContent).toContain('scene_detection · version not recorded');
    fireEvent.click(screen.getByRole('button', { name: /Provenance/ }));
    expect(screen.getByText(/doesn't record what made it/)).toBeInTheDocument();
  });
});
