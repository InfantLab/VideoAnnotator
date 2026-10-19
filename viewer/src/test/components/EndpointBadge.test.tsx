import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { EndpointBadge } from '@/components/ask/EndpointBadge';
import { NOTE_DISMISSED_KEY } from '@/lib/endpointWords';

describe('EndpointBadge', () => {
  beforeEach(() => vi.mocked(localStorage.getItem).mockReturnValue(null));

  it('says the model is on this computer, and says it once more in words', async () => {
    const { unmount } = render(<EndpointBadge endpoint={{ host: '127.0.0.1', class: 'local', confirmed: true }} model="qwen3.5:9b" />);
    expect(screen.getByTestId('endpoint-badge')).toHaveTextContent('Local · qwen3.5:9b on this computer');
    expect(screen.getByTestId('endpoint-note')).toHaveTextContent('Frames from your video are sent only to this computer.');
    await userEvent.click(screen.getByRole('button', { name: 'Got it' }));
    expect(localStorage.setItem).toHaveBeenCalledWith(NOTE_DISMISSED_KEY, '1');
    unmount();
    vi.mocked(localStorage.getItem).mockImplementation((key) => (key === NOTE_DISMISSED_KEY ? '1' : null));
    render(<EndpointBadge endpoint={{ host: 'localhost', class: 'local', confirmed: true }} model="m" />);
    expect(screen.queryByTestId('endpoint-note')).not.toBeInTheDocument();
  });

  it('names a lab server on the local network', () => {
    render(<EndpointBadge endpoint={{ host: 'lab-gpu', class: 'local', confirmed: true }} model="qwen3.5:9b" />);
    expect(screen.getByTestId('endpoint-badge')).toHaveTextContent('Local · qwen3.5:9b on lab-gpu');
  });

  it('warns that frames go to a remote server', () => {
    render(<EndpointBadge endpoint={{ host: 'api.example.com', class: 'remote', confirmed: false }} model="m" />);
    const badge = screen.getByTestId('endpoint-badge');
    expect(badge).toHaveTextContent('Remote · frames are sent to api.example.com');
    expect(badge.className).toContain('amber');
    expect(screen.queryByTestId('endpoint-note')).not.toBeInTheDocument();
  });

  it("takes the server's word for what is local", () => {
    render(<EndpointBadge endpoint={{ host: 'localhost', class: 'remote', confirmed: false }} model="m" />);
    expect(screen.getByTestId('endpoint-badge').dataset.class).toBe('remote');
  });
});
