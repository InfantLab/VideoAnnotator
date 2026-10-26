import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { PipelineAbout, PipelineTitle } from '@/components/PipelineGuide';
import { pipelineSummary } from '@/lib/pipelineGuide';
import type { PipelineDescriptor } from '@/types/pipelines';

const base: PipelineDescriptor = {
  id: 'face_analysis',
  name: 'Face Analysis (DeepFace)',
  description: 'Standards-only face detection.',
};

describe('PipelineTitle', () => {
  it('names what runs first, then what it finds (D1)', () => {
    render(<PipelineTitle pipeline={{ ...base, title: 'Faces: expression, age, gender' }} />);
    expect(screen.getByText('Face Analysis (DeepFace)')).toBeInTheDocument();
    expect(screen.getByText('expression, age, gender')).toBeInTheDocument();
    expect(screen.queryByText(/^Faces:/)).toBeNull();
  });

  it('falls back to the display name alone on older servers', () => {
    render(<PipelineTitle pipeline={base} />);
    expect(screen.getAllByText('Face Analysis (DeepFace)')).toHaveLength(1);
  });
});

describe('pipelineSummary', () => {
  it('prefers the summary, then the description', () => {
    expect(pipelineSummary({ ...base, summary: 'Finds faces.' })).toBe('Finds faces.');
    expect(pipelineSummary(base)).toBe('Standards-only face detection.');
    expect(pipelineSummary({ id: 'x', name: 'X' })).toBe('No description provided.');
  });
});

describe('PipelineAbout', () => {
  const full: PipelineDescriptor = {
    ...base,
    homepage: 'https://github.com/serengil/deepface',
    benchmarks: [
      { name: 'WIDER FACE (hard)', measure: 'AP', figure: '91.4%', source: 'https://arxiv.org/abs/1905.00641', note: 'RetinaFace' },
    ],
    qualityNote: 'None of these figures is for infants.',
    citations: [
      { key: 'a2026', bibtex: '@article{a2026,}' },
      { key: 'b2020', bibtex: '@inproceedings{b2020,}' },
    ],
  };

  it('shows homepage, benchmarks with their source, the note and citations', () => {
    render(<PipelineAbout pipeline={full} defaultOpen />);
    expect(screen.getByRole('link', { name: 'Homepage' })).toHaveAttribute('href', full.homepage);
    expect(screen.getByRole('link', { name: 'WIDER FACE (hard)' })).toHaveAttribute('href', 'https://arxiv.org/abs/1905.00641');
    expect(screen.getByRole('listitem')).toHaveTextContent('AP 91.4%');
    // The measure says what it means (UAT #19).
    expect(screen.getByText('AP')).toHaveAttribute('title', expect.stringContaining('Average precision'));
    expect(screen.getByText('None of these figures is for infants.')).toBeInTheDocument();
    expect(screen.getByText('Cite: a2026, b2020')).toBeInTheDocument();
  });

  it('says so when no benchmark is published, and is collapsed by default', () => {
    render(<PipelineAbout pipeline={{ ...base, benchmarks: [] }} />);
    expect(screen.queryByText('No published benchmark')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('About'));
    expect(screen.getByText('No published benchmark')).toBeInTheDocument();
  });

  it('renders nothing for servers that send no guide', () => {
    const { container } = render(<PipelineAbout pipeline={base} defaultOpen />);
    expect(container).toBeEmptyDOMElement();
  });

  it('copies every BibTeX entry, joined as in the methods paragraph', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    render(<PipelineAbout pipeline={full} defaultOpen />);
    fireEvent.click(screen.getByRole('button', { name: 'Copy BibTeX' }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('@article{a2026,}\n\n@inproceedings{b2020,}'));
    expect(await screen.findByRole('button', { name: 'Copied' })).toBeInTheDocument();
  });
});
