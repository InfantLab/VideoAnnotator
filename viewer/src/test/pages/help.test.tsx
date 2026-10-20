/** Help (spec 029 FR-027, T053). */
import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import Help from '@/pages/Help';
import { GLOSSARY } from '@/lib/glossary';
import { renderAt } from '@/test/fixtures/renderPage';

describe('Help', () => {
  it('has the words, the start points, the demos and the docs', () => {
    renderAt('/help', [{ path: '/help', element: <Help /> }]);
    const glossary = screen.getByTestId('glossary');
    for (const meaning of Object.values(GLOSSARY)) expect(within(glossary).getByText(meaning)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Run pipelines on videos/ })).toHaveAttribute('href', '/run');
    expect(screen.getByRole('link', { name: 'Peekaboo' })).toHaveAttribute('href', '/viewer?demo=peekaboo-rep3-v1.1.1');
    const docs = screen.getAllByRole('link').filter((a) => a.getAttribute('href')?.startsWith('https://infantlab.github.io/'));
    expect(docs.length).toBeGreaterThanOrEqual(8);
    for (const link of docs) expect(link).toHaveAttribute('target', '_blank');
  });
});
