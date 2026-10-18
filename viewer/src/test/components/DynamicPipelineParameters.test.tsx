import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it } from 'vitest';

import { DynamicPipelineParameters } from '@/components/DynamicPipelineParameters';
import type { PipelineDescriptor, PipelineParameterSchema } from '@/types/pipelines';

const parameters: PipelineParameterSchema[] = [
  { name: 'max_faces', type: 'integer', label: 'Most faces per frame', default: 10, group: 'behavioural', unit: 'faces' },
  { name: 'detect_age', type: 'boolean', label: 'Age', default: true, group: 'behavioural' },
  { name: 'confidence_threshold', type: 'number', label: 'Detection confidence', default: 0.7, group: 'technical' },
  { name: 'detection_backend', type: 'string', label: 'Face detector', default: 'deepface', group: 'technical' },
  {
    name: 'scale_factor',
    type: 'number',
    label: 'Haar cascade scale factor',
    default: 1.1,
    group: 'technical',
    appliesWhen: { detection_backend: ['opencv'] },
  },
];

const pipeline = (params: PipelineParameterSchema[]): PipelineDescriptor => ({
  id: 'face_analysis',
  name: 'Face Analysis (DeepFace)',
  title: 'Faces: expression, age, gender',
  summary: 'Finds faces.',
  parameters: params,
});

let lastConfig: Record<string, unknown> = {};

const Harness = ({ params = parameters, initial = {} }: { params?: PipelineParameterSchema[]; initial?: Record<string, unknown> }) => {
  const [config, setConfig] = useState<Record<string, unknown>>(initial);
  lastConfig = config;
  return (
    <DynamicPipelineParameters
      pipelines={[pipeline(params)]}
      selectedPipelineIds={['face_analysis']}
      config={config}
      onConfigChange={(updater) => setConfig((prev) => updater(prev))}
    />
  );
};

const openTechnical = () => fireEvent.click(screen.getByText(/Technical settings/));

describe('DynamicPipelineParameters grouping (spec 026)', () => {
  it('shows the title, behavioural settings open and technical collapsed with a count', () => {
    render(<Harness />);
    expect(screen.getByText('Faces: expression, age, gender')).toBeInTheDocument();
    expect(screen.getByText('What to measure')).toBeInTheDocument();
    expect(screen.getByLabelText('Most faces per frame')).toBeInTheDocument();
    expect(screen.getByText('Technical settings (2 · all at defaults)')).toBeInTheDocument();
    expect(screen.queryByLabelText('Detection confidence')).not.toBeInTheDocument();
  });

  it('counts technical settings changed from their defaults', () => {
    render(<Harness />);
    openTechnical();
    fireEvent.change(screen.getByLabelText('Detection confidence'), { target: { value: '0.5' } });
    expect(screen.getByText('Technical settings (2 · 1 changed)')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Most faces per frame'), { target: { value: '3' } });
    expect(screen.getByText('Technical settings (2 · 1 changed)')).toBeInTheDocument();
  });

  it('shows a conditional setting only when it applies', () => {
    render(<Harness />);
    openTechnical();
    expect(screen.queryByLabelText('Haar cascade scale factor')).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Face detector'), { target: { value: 'opencv' } });
    expect(screen.getByLabelText('Haar cascade scale factor')).toBeInTheDocument();
    expect(screen.getByText('Technical settings (3 · 1 changed)')).toBeInTheDocument();
  });

  it('keeps a changed setting visible, marked, when it no longer applies', () => {
    render(<Harness initial={{ face_analysis: { scale_factor: 1.2 } }} />);
    openTechnical();
    expect(screen.getByLabelText('Haar cascade scale factor')).toBeInTheDocument();
    expect(screen.getByText('Not used with these settings')).toBeInTheDocument();
  });

  it('produces the same config object as before', () => {
    render(<Harness />);
    fireEvent.change(screen.getByLabelText('Most faces per frame'), { target: { value: '3' } });
    expect(lastConfig).toEqual({ face_analysis: { max_faces: 3 } });
  });

  it('shows every setting, ungrouped, for servers that send no groups', () => {
    const legacy = parameters.map(({ group: _group, appliesWhen: _when, ...rest }) => rest);
    render(<Harness params={legacy} />);
    expect(screen.queryByText(/Technical settings/)).not.toBeInTheDocument();
    expect(screen.queryByText('What to measure')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Detection confidence')).toBeInTheDocument();
    expect(screen.getByLabelText('Haar cascade scale factor')).toBeInTheDocument();
  });
});
