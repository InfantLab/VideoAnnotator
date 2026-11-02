/** The Pipelines page (UAT #71): what a pipeline is, your defaults, Ask AI set apart. */
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import Pipelines from '@/pages/Pipelines';
import { mockVideosApi } from '@/test/fixtures/videos';
import { landed, renderAt } from '@/test/fixtures/renderPage';

const client = () => Object.getPrototypeOf(getApiClient());
const show = () => renderAt('/pipelines', [{ path: '/pipelines', element: <Pipelines /> }]);

function mockPipelinesApi(yours?: Record<string, Record<string, unknown>>) {
  const api = mockVideosApi({ yours });
  vi.spyOn(client(), 'getCurrentUser').mockResolvedValue({ id: 'u1', is_admin: false });
  vi.spyOn(client(), 'getExtrasGroups').mockResolvedValue([]);
  vi.spyOn(client(), 'getVlmModels').mockResolvedValue({ baseUrl: 'http://127.0.0.1:11434', models: ['qwen3.5:9b'], kind: 'ollama' });
  vi.spyOn(client(), 'getPipelineSchema').mockImplementation(async (id: string) => ({
    pipeline: { id, name: id },
    parameters:
      id === 'speech_recognition'
        ? [{ name: 'language', label: 'Language', type: 'string', default: '', description: 'The language spoken.' }]
        : [],
  }));
  return api;
}

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.restoreAllMocks());

describe('Pipelines', () => {
  it('says what a pipeline is, and sets Ask AI apart', async () => {
    mockPipelinesApi();
    show();
    expect(await screen.findByText(/A pipeline is a published research tool/)).toBeInTheDocument();
    const entries = await screen.findAllByTestId('pipeline-entry');
    expect(entries.map((e) => e.textContent)).not.toContainEqual(expect.stringContaining('Ask AI'));
    const panel = screen.getByTestId('ask-ai-panel');
    expect(panel).toHaveTextContent('you write the question, and you choose the model');
    expect(await within(panel).findByTestId('ask-model-line')).toHaveTextContent('Ollama at http://127.0.0.1:11434 answers, with qwen3.5:9b');
    expect(screen.getByTestId('another-tool')).toHaveTextContent('from VideoAnnotator 1.8, a pipeline can be added as a plugin');
  });

  it("keeps your defaults, explained, and runs a pipeline with them", async () => {
    const api = mockPipelinesApi({ speech_recognition: { language: 'en' } });
    const view = show();
    const whisper = (await screen.findAllByTestId('pipeline-entry')).find((e) => e.textContent?.includes('Whisper'))!;
    await waitFor(() => expect(within(whisper).getByTestId('your-defaults-cue')).toHaveTextContent('Yours: Language: en'));
    await userEvent.click(within(whisper).getAllByRole('button')[0]);
    expect(within(whisper).getByText('The language spoken.')).toBeInTheDocument();
    const field = within(whisper).getByLabelText('Language');
    fireEvent.change(field, { target: { value: 'nl' } });
    await waitFor(() => expect(api.putYourDefaults).toHaveBeenCalledWith({ speech_recognition: { language: 'nl' } }), { timeout: 2000 });

    await userEvent.click(within(whisper).getByRole('button', { name: /Run on videos/ }));
    expect(landed(view.container)).toEqual({
      address: '/run',
      state: {
        runStart: {
          target: null,
          settings: {
            selectedPipelines: ['speech_recognition'],
            config: { speech_recognition: { language: 'nl' } },
            label: 'your defaults for Whisper large-v3-turbo',
          },
        },
      },
    });
  });
});
