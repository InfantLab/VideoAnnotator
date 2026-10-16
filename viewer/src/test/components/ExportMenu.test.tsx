// Spec 025: results out of a job, run or dataset page.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ExportMenu } from '@/components/ExportMenu';
import { summaryLines } from '@/lib/exports';
import { apiClient, filenameFromDisposition, parseExportSummary, type ExportSummary } from '@/api/client';
import { APIError } from '@/api/handleError';

let isAdmin: boolean | 'unknown' = false;
vi.mock('@/hooks/useCurrentUser', () => ({ useCurrentUser: () => ({ isAdmin }) }));

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client');
  return {
    ...actual,
    apiClient: {
      exportResults: vi.fn(),
      getMethods: vi.fn(),
      installPipelineExtras: vi.fn(),
    },
  };
});

const SUMMARY: ExportSummary = {
  table_format_version: 1,
  rows: 10,
  videos: 2,
  pipelines: ['speech_recognition'],
  skipped: [{ run: 'R', video: 'clip3', file: 'notes.txt', reason: 'no reader for this file' }],
  missing: [{ run: 'R', video: 'clip7', reason: 'failed: boom' }],
  notes: [],
  left_out: [],
  truncated: false,
};

beforeEach(() => {
  vi.clearAllMocks();
  isAdmin = false;
  globalThis.URL.createObjectURL = vi.fn(() => 'blob:x');
  globalThis.URL.revokeObjectURL = vi.fn();
  // jsdom can't follow a download link.
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
});

describe('ExportMenu', () => {
  it('opens to the four exports for a run', async () => {
    render(<ExportMenu scope={{ kind: 'batch', id: 'b1' }} />);
    await userEvent.click(screen.getByRole('button', { name: /export/i }));
    for (const name of ['Table (CSV)', 'Table (Parquet)', 'ELAN files (.eaf)', 'Methods paragraph']) {
      expect(screen.getByRole('button', { name })).toBeInTheDocument();
    }
  });

  it('offers only the table for a dataset', async () => {
    render(<ExportMenu scope={{ kind: 'dataset', id: 'd1' }} />);
    await userEvent.click(screen.getByRole('button', { name: /export/i }));
    expect(screen.getByRole('button', { name: 'Table (CSV)' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'ELAN files (.eaf)' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Methods paragraph' })).not.toBeInTheDocument();
  });

  it('downloads the table and says what was left out', async () => {
    vi.mocked(apiClient.exportResults).mockResolvedValue({
      blob: new Blob(['a,b']),
      filename: 'Wave 2.csv',
      summary: SUMMARY,
    });
    render(<ExportMenu scope={{ kind: 'batch', id: 'b1' }} />);
    await userEvent.click(screen.getByRole('button', { name: /export/i }));
    await userEvent.click(screen.getByRole('button', { name: 'Table (CSV)' }));
    expect(apiClient.exportResults).toHaveBeenCalledWith({ kind: 'batch', id: 'b1' }, 'table', 'csv');
    expect(await screen.findByText('No results for clip7: failed: boom')).toBeInTheDocument();
    expect(screen.getByText(/Skipped notes.txt \(clip3\)/)).toBeInTheDocument();
  });

  it('asks for ELAN files of a job', async () => {
    vi.mocked(apiClient.exportResults).mockResolvedValue({ blob: new Blob(['x']), filename: null, summary: null });
    render(<ExportMenu scope={{ kind: 'job', id: 'j1' }} />);
    await userEvent.click(screen.getByRole('button', { name: /export/i }));
    await userEvent.click(screen.getByRole('button', { name: 'ELAN files (.eaf)' }));
    expect(apiClient.exportResults).toHaveBeenCalledWith({ kind: 'job', id: 'j1' }, 'elan', 'csv');
  });

  it('explains how to get Parquet, and lets an admin install it', async () => {
    isAdmin = true;
    vi.mocked(apiClient.exportResults).mockRejectedValue(
      new APIError('Parquet export needs the optional component', 422, undefined, {
        error: { code: 'PARQUET_NOT_INSTALLED', hint: 'pip install videoannotator[parquet]', details: { extra: 'parquet' } },
      })
    );
    vi.mocked(apiClient.installPipelineExtras).mockResolvedValue({ jobId: 'x', extraName: 'parquet', status: 'pending' });
    render(<ExportMenu scope={{ kind: 'job', id: 'j1' }} />);
    await userEvent.click(screen.getByRole('button', { name: /export/i }));
    await userEvent.click(screen.getByRole('button', { name: 'Table (Parquet)' }));
    expect(await screen.findByText('pip install videoannotator[parquet]')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Install it now' }));
    expect(apiClient.installPipelineExtras).toHaveBeenCalledWith('parquet');
    expect(await screen.findByText(/try Parquet again/)).toBeInTheDocument();
  });

  it('does not offer the install to non-admins', async () => {
    vi.mocked(apiClient.exportResults).mockRejectedValue(
      new APIError('x', 422, undefined, { error: { code: 'PARQUET_NOT_INSTALLED', hint: 'pip install videoannotator[parquet]' } })
    );
    render(<ExportMenu scope={{ kind: 'job', id: 'j1' }} />);
    await userEvent.click(screen.getByRole('button', { name: /export/i }));
    await userEvent.click(screen.getByRole('button', { name: 'Table (Parquet)' }));
    await screen.findByText('pip install videoannotator[parquet]');
    expect(screen.queryByRole('button', { name: 'Install it now' })).not.toBeInTheDocument();
  });

  it('shows other errors', async () => {
    vi.mocked(apiClient.exportResults).mockRejectedValue(new APIError('Job j1 has no results to export', 409));
    render(<ExportMenu scope={{ kind: 'job', id: 'j1' }} />);
    await userEvent.click(screen.getByRole('button', { name: /export/i }));
    await userEvent.click(screen.getByRole('button', { name: 'Table (CSV)' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('no results to export');
  });

  it('opens the methods paragraph', async () => {
    vi.mocked(apiClient.getMethods).mockResolvedValue({
      prose: 'The video was annotated automatically with VideoAnnotator 1.6.0 [@addyman2026videoannotator].',
      bibtex: '@software{addyman2026videoannotator,\n}',
      revisions: [],
      prompts: [],
      summary: { videos: 1, groups: 1, not_recorded: [] },
    });
    render(<ExportMenu scope={{ kind: 'job', id: 'j1' }} />);
    await userEvent.click(screen.getByRole('button', { name: /export/i }));
    await userEvent.click(screen.getByRole('button', { name: 'Methods paragraph' }));
    await waitFor(() => expect(screen.getByTestId('methods-prose')).toHaveTextContent('annotated automatically'));
  });
});

describe('export helpers', () => {
  it('reads the UTF-8 file name first', () => {
    expect(
      filenameFromDisposition(`attachment; filename="Wave ?.csv"; filename*=UTF-8''Wave%20%C3%A9.csv`)
    ).toBe('Wave é.csv');
    expect(filenameFromDisposition('attachment; filename="a.eaf"')).toBe('a.eaf');
    expect(filenameFromDisposition(null)).toBeNull();
  });

  it('parses the summary header, tolerating junk', () => {
    expect(parseExportSummary(JSON.stringify(SUMMARY))?.rows).toBe(10);
    expect(parseExportSummary('{nope')).toBeNull();
  });

  it('says nothing when nothing was left out', () => {
    expect(summaryLines({ ...SUMMARY, skipped: [], missing: [] })).toEqual([]);
    expect(summaryLines({ ...SUMMARY, skipped: [], missing: [], truncated: true })).toHaveLength(1);
  });
});
