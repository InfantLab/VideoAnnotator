/** The Layers panel says a shared run once and keeps its boxes in their columns (UAT #31). */
import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { LayersMenu } from '@/components/lanes/LayersMenu';
import type { Lane } from '@/lib/lanes';

const lane = (id: string, label: string, run: string): Lane =>
  ({ id, kind: id, pipeline: null, label, description: '', run: { id: run, label: run }, form: 'segments', rows: [], status: 'ready', overlay: id, source: [] }) as unknown as Lane;

const toggles = { collapsed: () => false, overlayOn: () => true, toggleCollapsed: vi.fn(), setOverlay: vi.fn() };

describe('LayersMenu', () => {
  it('names the shared run once, and a row’s run only where it differs', async () => {
    render(
      <LayersMenu
        lanes={[lane('speech', 'Speech', 'peekaboo · 26 Oct'), lane('scenes', 'Scenes', 'peekaboo · 26 Oct'), lane('speakers', 'Speakers', 'run 26 Sep')]}
        toggles={toggles}
        showLanes={vi.fn()}
        setOverlays={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /Layers/ }));
    const menu = await screen.findByTestId('layers-menu');
    expect(within(menu).getAllByText(/peekaboo · 26 Oct/)).toHaveLength(1);
    expect(within(menu).getByText('From peekaboo · 26 Oct')).toBeInTheDocument();
    const rows = within(menu).getAllByRole('row').slice(2);
    expect(rows.map((r) => r.textContent)).toEqual(['Speech', 'Scenes', 'Speakers · run 26 Sep']);
    rows.forEach((r) => expect(within(r).getAllByRole('cell')).toHaveLength(3));
  });
});
