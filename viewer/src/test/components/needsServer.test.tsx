import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import { NeedsServer } from '@/components/NeedsServer';

describe('NeedsServer', () => {
  it('says what needs a server, how to connect, and what works without one', () => {
    render(
      <MemoryRouter>
        <NeedsServer what="Videos lists the videos on a server." />
      </MemoryRouter>,
    );

    expect(screen.getByText('Videos lists the videos on a server.')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Connect' })).toHaveAttribute('href', '/settings');
    expect(screen.getByRole('link', { name: /Open files from this computer/ })).toHaveAttribute('href', '/viewer');
    expect(screen.getByRole('link', { name: /Demos/ }).getAttribute('href')).toMatch(/^\/viewer\?demo=/);
  });
});
