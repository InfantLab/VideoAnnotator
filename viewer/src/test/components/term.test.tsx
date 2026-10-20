import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import { PageHelp } from '@/components/PageHelp';
import { Term } from '@/components/Term';
import { TooltipProvider } from '@/components/ui/tooltip';
import { GLOSSARY } from '@/lib/glossary';

describe('Term', () => {
  it('explains the word on focus', async () => {
    render(
      <TooltipProvider delayDuration={0}>
        <p>
          Put videos in a <Term word="project" />.
        </p>
      </TooltipProvider>,
    );

    await userEvent.tab();

    expect(screen.getByText('project')).toHaveFocus();
    expect((await screen.findAllByText(GLOSSARY.project)).length).toBeGreaterThan(0);
  });

  it('has a sentence for every word, none in the API words', () => {
    for (const sentence of Object.values(GLOSSARY)) {
      expect(sentence).not.toMatch(/\b(jobs?|batch(es)?|datasets?|presets?)\b/i);
    }
    expect(Object.keys(GLOSSARY)).toEqual(
      expect.arrayContaining(['video', 'folder', 'run', 'project', 'workflow', 'track']),
    );
  });
});

describe('PageHelp', () => {
  it('links into the docs in a new tab', () => {
    render(<PageHelp page="projects">Group videos for a study.</PageHelp>);

    const link = screen.getByRole('link', { name: /More in the docs/ });
    expect(link).toHaveAttribute('href', 'https://infantlab.github.io/VideoAnnotator/usage/videos_and_projects/#projects');
    expect(link).toHaveAttribute('target', '_blank');
  });
});
