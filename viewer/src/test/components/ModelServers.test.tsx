import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { ModelServers } from '@/components/settings/ModelServers';
import {
  DEFAULT_SERVER,
  LAB_SERVER,
  OLD_REMOTE_SERVER,
  REMOTE_WARNING,
  serversResponse,
  stubPointerCapture,
} from '@/test/fixtures/modelServers';
import type { ModelServerCheck } from '@/types/ask';

vi.mock('@/api/client', () => ({
  apiClient: {
    listModelServers: vi.fn(),
    addModelServer: vi.fn(),
    removeModelServer: vi.fn(),
    checkModelServer: vi.fn(),
    setRemoteModelServers: vi.fn(),
  },
}));

const KEY = 'sk-typed-into-the-form-123456';

const check = (over: Partial<ModelServerCheck> = {}): ModelServerCheck => ({
  outcome: 'models',
  message: 'Reachable. 1 model.',
  hint: null,
  kind_matches: true,
  suggest_kind: null,
  suggest_base_url: null,
  models: ['gemma-3-4b'],
  endpoint: { host: '127.0.0.1', class: 'local', confirmed: true, sendable: true },
  video: null,
  ...over,
});

function renderCard(isAdmin: boolean | 'unknown' = true) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ModelServers isAdmin={isAdmin} />
    </QueryClientProvider>,
  );
}

async function fillAdd(address = 'http://127.0.0.1:8080/v1') {
  await userEvent.click(await screen.findByRole('button', { name: /Add a model server/ }));
  await userEvent.type(screen.getByLabelText('Name'), 'workstation');
  await userEvent.type(screen.getByLabelText('Address'), address);
}

describe('Settings → Model servers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    stubPointerCapture();
    vi.mocked(apiClient.listModelServers).mockResolvedValue(serversResponse());
  });

  it('lists the servers, the default first and not removable', async () => {
    renderCard();
    const rows = await screen.findAllByTestId('model-server');
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent(DEFAULT_SERVER.name);
    expect(within(rows[0]).queryByRole('button', { name: /Remove/ })).not.toBeInTheDocument();
    expect(rows[1]).toHaveTextContent('lab-gpu');
    expect(rows[1]).toHaveTextContent('OpenAI-compatible');
    expect(rows[1]).toHaveTextContent('key set');
    expect(within(rows[1]).getByTestId('endpoint-badge')).toHaveTextContent('Local · lab-gpu on 10.0.4.12');
    expect(screen.getByText('/data/model_server_keys.json')).toBeInTheDocument();
  });

  it('marks a server confirmed before the switch existed', async () => {
    vi.mocked(apiClient.listModelServers).mockResolvedValue(serversResponse([DEFAULT_SERVER, OLD_REMOTE_SERVER]));
    renderCard();
    const rows = await screen.findAllByTestId('model-server');
    expect(rows[1]).toHaveTextContent('Confirmed before remote servers needed switching on');
    expect(rows[1]).toHaveTextContent("no key ($FARM_KEY isn't set)");
    expect(within(rows[1]).getByTestId('endpoint-badge')).toHaveAttribute('data-class', 'remote');
  });

  it('adds a server only after a check passes, and never shows the key', async () => {
    vi.mocked(apiClient.checkModelServer).mockResolvedValue(check());
    vi.mocked(apiClient.addModelServer).mockResolvedValue(LAB_SERVER);
    renderCard();
    await fillAdd();
    await userEvent.click(screen.getByRole('radio', { name: 'A key' }));
    const keyInput = screen.getByLabelText('Key');
    expect(keyInput).toHaveAttribute('type', 'password');
    await userEvent.type(keyInput, KEY);
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Check' }));
    expect(await screen.findByText('Reachable. 1 model.')).toBeInTheDocument();
    expect(apiClient.checkModelServer).toHaveBeenCalledWith({
      kind: 'openai_compatible',
      base_url: 'http://127.0.0.1:8080/v1',
      api_key: KEY,
    });
    await userEvent.click(screen.getByRole('button', { name: 'Add' }));
    await waitFor(() =>
      expect(apiClient.addModelServer).toHaveBeenCalledWith({
        name: 'workstation',
        kind: 'openai_compatible',
        base_url: 'http://127.0.0.1:8080/v1',
        api_key: KEY,
        api_key_env: null,
      }),
    );
    await waitFor(() => expect(screen.queryByTestId('add-model-server')).not.toBeInTheDocument());
    expect(document.body.textContent).not.toContain(KEY);
    for (const input of document.querySelectorAll('input')) expect(input.value).not.toContain(KEY);
  });

  it('offers the address and kind the check suggests', async () => {
    vi.mocked(apiClient.checkModelServer).mockResolvedValue(
      check({
        outcome: 'wrong_kind',
        message: 'Something answers at http://127.0.0.1:8080, but not as an OpenAI-compatible server.',
        hint: 'Try http://127.0.0.1:8080/v1.',
        suggest_base_url: 'http://127.0.0.1:8080/v1',
        suggest_kind: 'ollama',
        models: [],
      }),
    );
    renderCard();
    await fillAdd('http://127.0.0.1:8080');
    await userEvent.click(screen.getByRole('button', { name: 'Check' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Use http://127.0.0.1:8080/v1' }));
    expect(screen.getByLabelText('Address')).toHaveValue('http://127.0.0.1:8080/v1');
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled();
  });

  it('can add anyway when the check fails', async () => {
    vi.mocked(apiClient.checkModelServer).mockResolvedValue(
      check({ outcome: 'unreachable', message: "Can't reach http://127.0.0.1:8080/v1.", models: [] }),
    );
    vi.mocked(apiClient.addModelServer).mockResolvedValue(LAB_SERVER);
    renderCard();
    await fillAdd();
    await userEvent.click(screen.getByRole('button', { name: 'Check' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Add anyway' }));
    await userEvent.click(screen.getByRole('button', { name: 'Add' }));
    await waitFor(() => expect(apiClient.addModelServer).toHaveBeenCalled());
  });

  it('says why a remote server is refused and how to switch them on', async () => {
    vi.mocked(apiClient.checkModelServer).mockResolvedValue(check());
    vi.mocked(apiClient.addModelServer).mockRejectedValue(
      new APIError('Remote model servers are switched off on this server.', 403, undefined, {
        error: {
          code: 'REMOTE_NOT_ALLOWED',
          message: 'Remote model servers are switched off on this server.',
          hint: 'An administrator can switch them on in Settings → Model servers.',
        },
      }),
    );
    renderCard();
    await fillAdd('https://api.openai.com/v1');
    await userEvent.click(screen.getByRole('button', { name: 'Check' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Add' }));
    expect(await screen.findByText(/switched off on this server\. An administrator can switch them on/)).toBeInTheDocument();
  });

  it('removes a server after a second click', async () => {
    vi.mocked(apiClient.removeModelServer).mockResolvedValue(undefined);
    renderCard();
    const rows = await screen.findAllByTestId('model-server');
    await userEvent.click(within(rows[1]).getByRole('button', { name: /Remove/ }));
    expect(apiClient.removeModelServer).not.toHaveBeenCalled();
    await userEvent.click(within(rows[1]).getByRole('button', { name: 'Remove lab-gpu' }));
    expect(apiClient.removeModelServer).toHaveBeenCalledWith('lab');
  });

  describe('the remote switch', () => {
    it('an admin switches it on after reading the warning', async () => {
      vi.mocked(apiClient.setRemoteModelServers).mockResolvedValue({
        remote_model_servers: true,
        fixed_by_environment: false,
        warning: REMOTE_WARNING,
      });
      renderCard(true);
      const toggle = await screen.findByRole('switch', { name: 'Remote model servers' });
      expect(screen.getByTestId('remote-switch')).toHaveTextContent('Remote model servers are off');
      await userEvent.click(toggle);
      expect(await screen.findByText(REMOTE_WARNING)).toBeInTheDocument();
      expect(apiClient.setRemoteModelServers).not.toHaveBeenCalled();
      await userEvent.click(screen.getByRole('button', { name: 'Switch on' }));
      expect(apiClient.setRemoteModelServers).toHaveBeenCalledWith(true);
    });

    it('is fixed when the configuration sets it', async () => {
      vi.mocked(apiClient.listModelServers).mockResolvedValue(serversResponse(undefined, { fixed_by_environment: true }));
      renderCard(true);
      expect(await screen.findByRole('switch', { name: 'Remote model servers' })).toBeDisabled();
      expect(screen.getByText("Set by this server's configuration.")).toBeInTheDocument();
    });

    it("isn't offered to someone who isn't an admin", async () => {
      renderCard(false);
      await screen.findByTestId('remote-switch');
      expect(screen.queryByRole('switch')).not.toBeInTheDocument();
    });
  });
});
