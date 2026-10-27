import { fireEvent, render as rtlRender, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { APIClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { HuggingFaceTokenCard } from '@/components/HuggingFaceTokenSetup';
import { ReadinessDetails } from '@/components/LockedPipelineCard';
import { API_TOKEN_STORAGE_KEY } from '@/lib/apiConnection';
import type { ReadinessItem } from '@/types/pipelines';

const render = (ui: React.ReactElement) => rtlRender(ui, { wrapper: MemoryRouter });

const tokenBlocker: ReadinessItem = {
  kind: 'secret',
  name: 'HUGGINGFACE_TOKEN',
  message: "A Hugging Face access token isn't set.",
  helpUrl: 'https://huggingface.co/settings/tokens',
};
const licenceNote: ReadinessItem = {
  kind: 'licence',
  name: 'pyannote/speaker-diarization-3.1',
  message: 'Accept the licence.',
  helpUrl: 'https://huggingface.co/pyannote/speaker-diarization-3.1',
};
const TOKEN = 'hf_SECRETsecret';

const renderPanel = (isAdmin: boolean | 'unknown' = true, onCheckAgain = vi.fn()) => {
  render(<ReadinessDetails blockers={[tokenBlocker]} notes={[licenceNote]} isAdmin={isAdmin} onCheckAgain={onCheckAgain} />);
  return onCheckAgain;
};

const paste = (value: string) => {
  fireEvent.change(screen.getByLabelText('Paste the token here'), { target: { value } });
  fireEvent.click(screen.getByRole('button', { name: 'Save and check' }));
};

// The test setup stubs localStorage; say a key is configured, as connecting the viewer would.
const connect = () =>
  vi.spyOn(localStorage, 'getItem').mockImplementation((key) => (key === API_TOKEN_STORAGE_KEY ? 'va_test' : null));

describe('Hugging Face token setup (spec 031)', () => {
  beforeEach(connect);
  afterEach(() => vi.restoreAllMocks());

  it('shows the three steps with the token page and the licence linked', () => {
    renderPanel();
    expect(screen.getByText(/needs a free Hugging Face token/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open huggingface.co' })).toHaveAttribute('href', tokenBlocker.helpUrl);
    expect(screen.getByRole('link', { name: 'pyannote/speaker-diarization-3.1' })).toHaveAttribute('href', licenceNote.helpUrl);
    expect(screen.getByLabelText('Paste the token here')).toHaveAttribute('type', 'password');
    // The licence is in the panel, not repeated as a separate note.
    expect(screen.queryByText('Accept the licence.')).not.toBeInTheDocument();
  });

  it('tells a non-admin who to ask, with no field', () => {
    renderPanel(false);
    expect(screen.getByText(/Ask whoever runs this VideoAnnotator/)).toBeInTheDocument();
    expect(screen.queryByLabelText('Paste the token here')).not.toBeInTheDocument();
  });

  it('saves, names the account, re-checks, and clears the field', async () => {
    const save = vi
      .spyOn(APIClient.prototype, 'saveHuggingFaceToken')
      .mockResolvedValue({ set: true, source: 'saved', envVar: null, account: 'caspar', check: 'ok' });
    const onCheckAgain = renderPanel();
    paste(`  ${TOKEN} `);
    expect(await screen.findByRole('status')).toHaveTextContent('Token saved and works (account: caspar).');
    expect(save).toHaveBeenCalledWith(TOKEN);
    expect(onCheckAgain).toHaveBeenCalled();
    expect(screen.getByLabelText('Paste the token here')).toHaveValue('');
    expect(document.body.textContent).not.toContain(TOKEN);
  });

  it('says a rejected token was not saved', async () => {
    vi.spyOn(APIClient.prototype, 'saveHuggingFaceToken').mockRejectedValue(
      new APIError('rejected', 400, undefined, { error: { code: 'HF_TOKEN_REJECTED' } })
    );
    renderPanel();
    paste(TOKEN);
    expect(await screen.findByRole('status')).toHaveTextContent('Hugging Face rejected that token');
  });

  it('says when the environment token is used instead', async () => {
    vi.spyOn(APIClient.prototype, 'saveHuggingFaceToken').mockResolvedValue({
      set: true,
      source: 'environment',
      envVar: 'HF_TOKEN',
      account: 'x',
      check: 'ok',
    });
    renderPanel();
    paste(TOKEN);
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent("the server's environment sets HF_TOKEN"));
  });

  it('without an API key, says to connect first and offers the terminal, with no field', () => {
    vi.spyOn(localStorage, 'getItem').mockReturnValue(null);
    renderPanel('unknown');
    expect(screen.queryByLabelText('Paste the token here')).not.toBeInTheDocument();
    expect(screen.getByText(/once this viewer is connected to the server with an admin key/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'the steps in Settings' })).toHaveAttribute('href', '/settings');
    expect(screen.getByText('videoannotator settings huggingface-token')).toBeInTheDocument();
  });

  it('turns a 401 into what to do, not the raw API message', async () => {
    vi.spyOn(APIClient.prototype, 'saveHuggingFaceToken').mockRejectedValue(
      new APIError('API key required for this endpoint', 401)
    );
    renderPanel('unknown');
    paste(TOKEN);
    const status = await screen.findByRole('status');
    expect(status).toHaveTextContent("isn't connected with an API key");
    expect(status).toHaveTextContent('videoannotator settings huggingface-token');
    expect(status).not.toHaveTextContent('API key required for this endpoint');
  });

  it('turns a 403 into what to do', async () => {
    vi.spyOn(APIClient.prototype, 'saveHuggingFaceToken').mockRejectedValue(new APIError('Admin required', 403));
    renderPanel('unknown');
    paste(TOKEN);
    expect(await screen.findByRole('status')).toHaveTextContent("isn't an admin key");
  });

  it('marks licences known not to be accepted', () => {
    render(
      <ReadinessDetails
        blockers={[tokenBlocker, { ...licenceNote, message: 'not accepted' }]}
        notes={[]}
        isAdmin
      />
    );
    expect(screen.getByText('not accepted yet')).toBeInTheDocument();
  });
});

describe('Settings: Hugging Face token card (spec 031)', () => {
  beforeEach(connect);
  afterEach(() => vi.restoreAllMocks());

  const renderCard = (isAdmin: boolean | 'unknown' = true) => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <HuggingFaceTokenCard isAdmin={isAdmin} />
      </QueryClientProvider>
    );
  };

  it('shows a saved token by its account and offers change and remove', async () => {
    vi.spyOn(APIClient.prototype, 'getHuggingFaceToken').mockResolvedValue({
      set: true, source: 'saved', envVar: null, account: 'caspar', check: 'ok',
    });
    const remove = vi.spyOn(APIClient.prototype, 'removeHuggingFaceToken').mockResolvedValue({
      set: false, source: null, envVar: null, account: null, check: null,
    });
    renderCard();
    expect(await screen.findByTestId('hf-token-state')).toHaveTextContent('Token working · account caspar');
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }));
    await waitFor(() => expect(screen.getByTestId('hf-token-state')).toHaveTextContent('Not set'));
    expect(remove).toHaveBeenCalled();
  });

  it('says the environment token is used and offers no change', async () => {
    vi.spyOn(APIClient.prototype, 'getHuggingFaceToken').mockResolvedValue({
      set: true, source: 'environment', envVar: 'HUGGINGFACE_TOKEN', account: null, check: 'ok',
    });
    renderCard();
    expect(await screen.findByText(/the server was started with \(HUGGINGFACE_TOKEN\)/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Change' })).not.toBeInTheDocument();
  });

  it('lets an admin add one, and hides the controls from a non-admin', async () => {
    vi.spyOn(APIClient.prototype, 'getHuggingFaceToken').mockResolvedValue({
      set: false, source: null, envVar: null, account: null, check: null,
    });
    renderCard(false);
    await waitFor(() => expect(screen.getByTestId('hf-token-state')).toHaveTextContent('Not set'));
    expect(screen.getByText(/Only an administrator of this server can add it/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Add a token' })).not.toBeInTheDocument();
  });

  it('without an API key, explains how to connect instead of offering buttons', async () => {
    vi.spyOn(localStorage, 'getItem').mockReturnValue(null);
    vi.spyOn(APIClient.prototype, 'getHuggingFaceToken').mockResolvedValue({
      set: false, source: null, envVar: null, account: null, check: null,
    });
    renderCard('unknown');
    await waitFor(() => expect(screen.getByTestId('hf-token-state')).toHaveTextContent('Not set'));
    expect(screen.getByRole('link', { name: 'accept the pyannote licence' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Add a token' })).not.toBeInTheDocument();
    expect(screen.getByText(/To add or change it from here/)).toBeInTheDocument();
  });
});
