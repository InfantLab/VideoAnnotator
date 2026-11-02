/** The footer: on every page, each link once (UAT #69). */
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import { Footer } from '@/components/Footer';
import { serverDiffers } from '@/utils/version';

const server = { version: '1.7.0' as string | null };
vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => ({ capabilities: server.version ? { version: server.version } : null }),
}));

const show = () =>
  render(
    <MemoryRouter>
      <Footer />
    </MemoryRouter>,
  );

describe('Footer', () => {
  it('links to each place once, and the repository only as Source', () => {
    show();
    const links = screen.getAllByRole('link').map((a) => [a.textContent, a.getAttribute('href')]);
    expect(links).toEqual([
      ['Help', '/help'],
      ['Docs ', 'https://infantlab.github.io/VideoAnnotator/'],
      ['Source ', 'https://github.com/InfantLab/VideoAnnotator'],
      ['How to cite ', 'https://doi.org/10.5281/zenodo.16961751'],
    ]);
    expect(screen.queryByText(/Powered by/)).toBeNull();
    expect(screen.getByRole('button', { name: 'Report a problem' })).toBeInTheDocument();
  });

  it("names the server's version, in amber with a reason when it isn't this release", () => {
    server.version = '1.7.0';
    const { unmount } = show();
    const shown = screen.getByText(/server 1\.7\.0/);
    expect(shown).toHaveTextContent(/^· server 1\.7\.0$/);
    expect(shown).not.toHaveClass('text-amber-600');
    unmount();
    server.version = '1.6.0';
    show();
    expect(screen.getByText(/server 1\.6\.0: older than this viewer/)).toHaveClass('text-amber-600');
  });

  it('leaves the server out when not connected', () => {
    server.version = null;
    show();
    expect(screen.getByTestId('footer')).not.toHaveTextContent('server');
  });

  it('opens Report a problem', async () => {
    show();
    await userEvent.click(screen.getByRole('button', { name: 'Report a problem' }));
    expect(screen.getByRole('dialog', { name: 'Report a problem' })).toBeInTheDocument();
  });

  it('compares releases, not builds', () => {
    expect(serverDiffers('1.7.0.dev0', '1.7.0-dev.0')).toBeNull();
    expect(serverDiffers('1.6.2', '1.7.0')).toBe('older');
    expect(serverDiffers('1.8.0', '1.7.0')).toBe('newer');
  });
});
