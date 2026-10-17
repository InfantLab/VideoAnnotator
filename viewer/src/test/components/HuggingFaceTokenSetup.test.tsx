import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { APIClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { HuggingFaceTokenCard } from '@/components/HuggingFaceTokenSetup';
import { ReadinessDetails } from '@/components/LockedPipelineCard';
import type { ReadinessItem } from '@/types/pipelines';

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

describe('Hugging Face token setup (spec 031)', () => {
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
    expect(await screen.findByText('Set (account caspar, saved on this server).')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }));
    expect(await screen.findByText(/^Not set\./)).toBeInTheDocument();
    expect(remove).toHaveBeenCalled();
  });

  it('says the environment token is used and offers no change', async () => {
    vi.spyOn(APIClient.prototype, 'getHuggingFaceToken').mockResolvedValue({
      set: true, source: 'environment', envVar: 'HUGGINGFACE_TOKEN', account: null, check: 'ok',
    });
    renderCard();
    expect(await screen.findByText(/from HUGGINGFACE_TOKEN in the server's environment/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Change' })).not.toBeInTheDocument();
  });

  it('lets an admin add one, and hides the controls from a non-admin', async () => {
    vi.spyOn(APIClient.prototype, 'getHuggingFaceToken').mockResolvedValue({
      set: false, source: null, envVar: null, account: null, check: null,
    });
    renderCard(false);
    expect(await screen.findByText(/^Not set\./)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Add a token' })).not.toBeInTheDocument();
  });
});
