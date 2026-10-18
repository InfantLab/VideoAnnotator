// Spec 025: the methods paragraph, to copy into a paper.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MethodsDialog from '@/components/MethodsDialog';
import { methodsMarkdown } from '@/lib/exports';
import { apiClient, type MethodsDescription } from '@/api/client';

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client');
  return { ...actual, apiClient: { getMethods: vi.fn() } };
});

const METHODS: MethodsDescription = {
  prose: 'All 2 videos were annotated automatically with VideoAnnotator 1.6.0 [@addyman2026videoannotator].',
  bibtex: '@software{addyman2026videoannotator,\n  title = {VideoAnnotator}\n}',
  revisions: ['whisper base (openai-whisper): sha256 ed3a0b6b1c0e'],
  prompts: [{ sha256: 'f'.repeat(64), text: '<b>Classify</b> as TOUCH' }],
  summary: { videos: 2, groups: 1, not_recorded: [] },
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.getMethods).mockResolvedValue(METHODS);
});

describe('MethodsDialog', () => {
  it('shows prose, revisions, BibTeX and prompts as text', async () => {
    render(<MethodsDialog scope={{ kind: 'batch', id: 'b1' }} open onOpenChange={() => {}} />);
    expect(await screen.findByTestId('methods-prose')).toHaveTextContent('All 2 videos');
    expect(screen.getByTestId('methods-bibtex')).toHaveTextContent('@software{addyman2026videoannotator,');
    expect(screen.getByText(/whisper base/)).toBeInTheDocument();
    // A prompt is shown literally, never as markup.
    expect(screen.getByText('<b>Classify</b> as TOUCH')).toBeInTheDocument();
    expect(apiClient.getMethods).toHaveBeenCalledWith({ kind: 'batch', id: 'b1' });
  });

  it('copies the paragraph', async () => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, 'writeText').mockResolvedValue();
    render(<MethodsDialog scope={{ kind: 'job', id: 'j1' }} open onOpenChange={() => {}} />);
    await screen.findByTestId('methods-prose');
    await user.click(screen.getByRole('button', { name: 'Copy paragraph' }));
    expect(writeText).toHaveBeenCalledWith(METHODS.prose);
  });

  it('says when the methods could not be written', async () => {
    vi.mocked(apiClient.getMethods).mockRejectedValue(new Error('This run has no results to export'));
    render(<MethodsDialog scope={{ kind: 'job', id: 'j1' }} open onOpenChange={() => {}} />);
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('no results to export'));
  });

  it('builds the same Markdown as the CLI', () => {
    const markdown = methodsMarkdown(METHODS);
    expect(markdown.startsWith('# Methods\n\nAll 2 videos')).toBe(true);
    expect(markdown).toContain('## Model revisions\n\n- whisper base');
    expect(markdown).toContain('```bibtex\n@software{addyman2026videoannotator,');
    expect(markdown).toContain(`## Prompt ${'f'.repeat(12)}`);
  });
});

describe('methodsMarkdown', () => {
  it('is byte for byte what MethodsDescription.to_markdown() writes', () => {
    // From videoannotator.exporters.methods with the same fields.
    const fromPython =
      '# Methods\n\nP\n\n## Model revisions\n\n- r\n\n## References (BibTeX)\n\n```bibtex\nB\n```\n\n' +
      `## Prompt ffffffffffff\n\nsha256 \`${'f'.repeat(64)}\`\n\n\`\`\`text\nT\n\`\`\`\n`;
    expect(
      methodsMarkdown({
        prose: 'P',
        bibtex: 'B',
        revisions: ['r'],
        prompts: [{ sha256: 'f'.repeat(64), text: 'T' }],
        summary: { videos: 1, groups: 1, not_recorded: [] },
      })
    ).toBe(fromPython);
  });
});
