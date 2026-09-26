// Load and save presets from the job wizard (VideoAnnotator spec 007, handoff item 4).
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { PresetBar } from '@/components/PresetBar';
import { apiClient } from '@/api/client';
import type { Preset } from '@/types/presets';

vi.mock('@/api/client', () => ({
  apiClient: { listPresets: vi.fn(), createPreset: vi.fn() },
}));

const preset: Preset = {
  id: 'p1',
  name: 'Faces only',
  owner_user_id: 'u',
  selected_pipelines: ['face_analysis'],
  config: { face_analysis: { confidence_threshold: 0.5 } },
  created_at: '2026-09-26T10:00:00Z',
};

const renderBar = (props: Partial<React.ComponentProps<typeof PresetBar>> = {}) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <PresetBar selectedPipelines={['face_analysis', 'scene_detection']} config={{ face_analysis: { a: 1 }, other: { b: 2 } }} onApply={vi.fn()} {...props} />
    </QueryClientProvider>
  );

describe('PresetBar', () => {
  beforeEach(() => vi.clearAllMocks());

  it('applies the chosen preset', async () => {
    vi.mocked(apiClient.listPresets).mockResolvedValue({ presets: [preset], total: 1 });
    const onApply = vi.fn();
    renderBar({ onApply });
    fireEvent.change(await screen.findByLabelText('Preset'), { target: { value: 'p1' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    expect(onApply).toHaveBeenCalledWith(preset);
  });

  it('saves the selection with only the selected pipelines settings', async () => {
    vi.mocked(apiClient.listPresets).mockResolvedValue({ presets: [], total: 0 });
    vi.mocked(apiClient.createPreset).mockResolvedValue({ ...preset, id: 'p2', name: 'Mine' });
    renderBar();
    fireEvent.click(await screen.findByRole('button', { name: /Save selection as preset/ }));
    fireEvent.change(screen.getByLabelText('Preset name'), { target: { value: 'Mine' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() =>
      expect(apiClient.createPreset).toHaveBeenCalledWith({
        name: 'Mine',
        selected_pipelines: ['face_analysis', 'scene_detection'],
        config: { face_analysis: { a: 1 } },
      })
    );
  });

  it('hides itself on servers without presets', async () => {
    vi.mocked(apiClient.listPresets).mockRejectedValue(new Error('404'));
    const { container } = renderBar();
    await waitFor(() => expect(container).toBeEmptyDOMElement());
  });
});
