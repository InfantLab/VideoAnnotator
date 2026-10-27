/** Spec 028 US3: the values beside the video. */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { DataTable } from '@/components/video/DataTable';
import { rowsFromFiles, type Row } from '@/lib/table/rows';
import { FIXTURES, lanesLoaded, mockVideoApi, renderVideoPage, wideLayout } from '@/test/videoPageHarness';

const demoRows = () =>
  rowsFromFiles(
    readdirSync(FIXTURES)
      .filter((n) => statSync(path.join(FIXTURES, n)).isFile())
      .map((name) => ({ name, text: readFileSync(path.join(FIXTURES, name), 'utf8') })),
    'demo_clip',
    'job-1',
  ).sort((a, b) => a.start - b.start);

function table(rows: Row[], over: Partial<Parameters<typeof DataTable>[0]> & { summary?: boolean } = {}) {
  const props = {
    rows,
    currentTime: 0,
    selectedRow: null,
    onRowClick: vi.fn(),
    filter: 'all',
    onFilterChange: vi.fn(),
    runLabels: {},
    ...over,
  };
  delete (props as { summary?: boolean }).summary;
  const view = render(<DataTable {...props} />);
  // Most of these are about every row; the summary has its own tests.
  if (!over.summary) fireEvent.click(screen.getByRole('radio', { name: 'All rows' }));
  return { props, ...view };
}

const highlighted = () => screen.getAllByTestId('data-row').filter((r) => r.getAttribute('aria-selected') === 'true');

afterEach(() => vi.restoreAllMocks());

describe('the Data tab', () => {
  it('lists the same rows, in the same order, as the table builder', () => {
    const rows = demoRows();
    table(rows);
    const shown = screen.getAllByTestId('data-row').map((r) => Number(r.dataset.row));
    expect(shown).toEqual(shown.map((_, i) => shown[0] + i));
    expect(screen.getByTestId('row-count')).toHaveTextContent(`${rows.length} rows`);
  });

  it('highlights the row at or before the playhead as it moves', () => {
    const rows = demoRows();
    const { rerender, props } = table(rows, { currentTime: 0 });
    for (const t of [1.0, 9.5]) {
      rerender(<DataTable {...props} currentTime={t} />);
      const expected = rows.map((r) => r.start <= t).lastIndexOf(true);
      expect(highlighted().map((r) => Number(r.dataset.row))).toEqual([expected]);
    }
  });

  it('stops following when the reader scrolls, until "Follow playback"', () => {
    table(demoRows());
    expect(screen.queryByRole('button', { name: /Follow playback/ })).toBeNull();
    fireEvent.scroll(screen.getByRole('table'), { target: { scrollTop: 300 } });
    fireEvent.click(screen.getByRole('button', { name: /Follow playback/ }));
    expect(screen.queryByRole('button', { name: /Follow playback/ })).toBeNull();
  });

  it('filters by lane and searches, saying how many of how many', () => {
    const rows = demoRows();
    const { rerender, props } = table(rows, { filter: 'speakers' });
    expect(screen.getByTestId('row-count')).toHaveTextContent(`3 of ${rows.length} rows`);
    expect(screen.getAllByTestId('data-row').every((r) => within(r).getByText('Speakers'))).toBe(true);
    rerender(<DataTable {...props} filter="all" />);
    fireEvent.change(screen.getByRole('textbox', { name: 'Search the rows' }), { target: { value: 'BABY' } });
    expect(screen.getByTestId('row-count')).toHaveTextContent(`1 of ${rows.length} rows`);
  });

  it('renders only the rows in view, even with 50,000', () => {
    const [one] = demoRows();
    const rows = Array.from({ length: 50_000 }, (_, i) => ({ ...one, start: i * 0.07 }));
    table(rows);
    expect(screen.getAllByTestId('data-row').length).toBeLessThan(200);
    expect(screen.getByTestId('row-count')).toHaveTextContent('50,000 rows');
  });
});

describe('the Data tab and the timeline (SC-005)', () => {
  const lane = (kind: string) => screen.getAllByRole('group').filter((g) => g.dataset.kind === kind);

  it('seeks to a row\'s start when it is clicked, and selects its lane', async () => {
    mockVideoApi();
    wideLayout();
    renderVideoPage();
    await lanesLoaded();
    const row = screen.getAllByTestId('data-row').find((r) => within(r).queryByText('Speakers'))!;
    fireEvent.click(row);
    expect(screen.getByTestId('readout')).toHaveTextContent(/^0:00\.6 /);
    expect(within(lane('speakers')[0]).getAllByTestId('row-label')[0]).toHaveAttribute('aria-pressed', 'true');
  });

  it('picks out an event\'s row when the event is clicked', async () => {
    mockVideoApi();
    wideLayout();
    renderVideoPage();
    await lanesLoaded();
    fireEvent.click(within(lane('speakers')[0]).getByTestId(/^lane-body-/), { clientX: 50, clientY: 5 });
    const picked = screen.getAllByTestId('data-row').filter((r) => r.className.includes('outline'));
    expect(picked).toHaveLength(1);
    expect(within(picked[0]).getByText('Speakers')).toBeInTheDocument();
    expect(within(picked[0]).getAllByText('SPEAKER_00').length).toBeGreaterThan(0);
  });
});

describe('the Data tab summary (D6, UAT #46)', () => {
  it('shows one row per event, without per-class scores or repeats, and answers as plain text', () => {
    const rows = demoRows();
    const answer = { ...rows[0], pipeline: 'vlm_annotation', measure: 'vlm_label', label: 'There are **4** people', track: '', value: null };
    const repeated = rows.find((r) => r.measure === 'speech')!;
    table([...rows, answer, { ...repeated }].sort((a, b) => a.start - b.start), { summary: true });
    expect(screen.getByRole('radio', { name: 'Summary' })).toHaveAttribute('aria-checked', 'true');
    const count = Number((screen.getByTestId('row-count').textContent ?? '').split(' ')[0]);
    expect(count).toBeLessThan(rows.length / 2);
    const texts = screen.getAllByTestId('data-row').map((r) => r.textContent ?? '');
    expect(texts.some((t) => t.includes('There are 4 people'))).toBe(true);
    expect(texts.filter((t) => t.includes(repeated.label!) && t.includes('Speech'))).toHaveLength(1);
    fireEvent.click(screen.getByRole('radio', { name: 'All rows' }));
    expect(screen.getByTestId('row-count')).toHaveTextContent(`${rows.length + 2} rows`);
  });
});
