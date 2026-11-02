/** Report a problem (UAT #68): nothing leaves until the researcher sends it, and they see all of it. */
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ReportProblemDialog } from '@/components/ReportProblem';
import { clearErrors, recordError } from '@/lib/errorLog';
import { DEMO_DATA_SETS } from '@/utils/debugUtils';

vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
  useServerCapabilitiesContext: () => ({ capabilities: { version: '1.7.0' } }),
}));

const DEMO = DEMO_DATA_SETS['peekaboo-rep3-v1.1.1'].video.split('/').pop()!;
const show = (video: string | null = null) =>
  render(<ReportProblemDialog open onOpenChange={() => {}} video={video} />);

afterEach(() => {
  clearErrors();
  vi.restoreAllMocks();
});

describe('Report a problem', () => {
  it('sends nothing until something is written', async () => {
    show();
    for (const name of ['Copy', 'Email it', 'Open a GitHub issue']) {
      expect(screen.getByRole('button', { name })).toBeDisabled();
    }
    await userEvent.type(screen.getByLabelText('What happened?'), 'The lane is empty');
    expect(screen.getByRole('button', { name: 'Email it' })).toBeEnabled();
  });

  it('shows every detail that goes with it, and never the key or paths', async () => {
    recordError(new Error("Can't read /home/ada/Studies/p01_speech_recognition.vtt"));
    show();
    const list = screen.getByTestId('report-details-list');
    expect(list).toHaveTextContent('Server 1.7.0');
    expect(list).toHaveTextContent("Can't read …/p01_speech_recognition.vtt");
    expect(list).not.toHaveTextContent('/home/ada');
    expect(list).toHaveTextContent('Never added: your key, folder paths, results.');
  });

  it("adds a study video's name only if ticked; a demo clip's starts ticked", async () => {
    const { unmount } = show('p01_session2.mp4');
    expect(screen.getByLabelText(/The video's name: p01_session2.mp4/)).not.toBeChecked();
    unmount();
    show(DEMO);
    expect(screen.getByLabelText(new RegExp(`The video's name: ${DEMO.replace(/\./g, '\\.')}`))).toBeChecked();
  });

  it('opens a GitHub issue with the form filled in', async () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    show('p01_session2.mp4');
    await userEvent.type(screen.getByLabelText('What happened?'), 'The lane is empty');
    await userEvent.click(screen.getByRole('button', { name: 'Open a GitHub issue' }));
    const url = new URL(String(open.mock.calls[0][0]));
    expect(url.searchParams.get('what')).toBe('The lane is empty');
    expect(url.searchParams.get('details')).toContain('Viewer');
    expect(url.searchParams.get('details')).not.toContain('p01_session2');
  });

  it('copies the report as text', async () => {
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    show();
    await userEvent.type(screen.getByLabelText('What happened?'), 'The lane is empty');
    await userEvent.click(screen.getByLabelText('Add details that help us find it'));
    await userEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(writeText).toHaveBeenCalledWith(expect.stringContaining('What happened:\nThe lane is empty'));
    expect(writeText.mock.calls[0][0]).not.toContain('Details');
    expect(await screen.findByRole('button', { name: 'Copied' })).toBeInTheDocument();
  });
});
