// Component tests for VlmModelPicker (VideoAnnotator spec 009)
// Covers: live dropdown when reachable, free-text fallback when unreachable
// or when reachable-but-empty -- viewer-handoff #1/#3: "falling back to the
// existing free-text input when the endpoint reports the server
// unreachable. Don't hard-block configuration just because Ollama isn't
// running yet."

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

const mockUseVlmModels = vi.fn();

vi.mock('@/hooks/useVlmModels', () => ({
  useVlmModels: () => mockUseVlmModels()
}));

import { VlmModelPicker } from '@/components/VlmModelPicker';

describe('VlmModelPicker', () => {
  it('shows a live dropdown of models when the server is reachable', () => {
    mockUseVlmModels.mockReturnValue({
      data: { baseUrl: 'http://127.0.0.1:11434', models: ['qwen3.5:9b', 'gemma4:12b'] },
      isLoading: false,
      isError: false,
      error: null
    });

    render(<VlmModelPicker value="qwen3.5:9b" onChange={vi.fn()} fieldId="model" />);

    expect(screen.getByText(/2 models available/i)).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toBeInTheDocument();
    // The free-text fallback input must not also be present.
    expect(screen.queryByPlaceholderText(/e.g. qwen3.5:9b/i)).not.toBeInTheDocument();
  });

  it('falls back to free text and flags unreachability when the server cannot be reached', () => {
    mockUseVlmModels.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error('Cannot reach Ollama server')
    });

    render(<VlmModelPicker value="my-typed-model" onChange={vi.fn()} fieldId="model" />);

    expect(screen.getByText(/Ollama unreachable/i)).toBeInTheDocument();
    const input = screen.getByDisplayValue('my-typed-model');
    expect(input).toBeInTheDocument();
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
  });

  it('falls back to free text when reachable but zero models are pulled', () => {
    mockUseVlmModels.mockReturnValue({
      data: { baseUrl: 'http://127.0.0.1:11434', models: [] },
      isLoading: false,
      isError: false,
      error: null
    });

    render(<VlmModelPicker value="" onChange={vi.fn()} fieldId="model" />);

    expect(screen.getByText(/No models pulled/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/e.g. qwen3.5:9b/i)).toBeInTheDocument();
  });

  it('calls onChange as the user types in the free-text fallback', async () => {
    mockUseVlmModels.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error('unreachable')
    });
    const onChange = vi.fn();
    const user = userEvent.setup();

    render(<VlmModelPicker value="" onChange={onChange} fieldId="model" />);
    await user.type(screen.getByRole('textbox'), 'x');

    expect(onChange).toHaveBeenCalledWith('x');
  });

  it('shows a loading state while the models query is in flight', () => {
    mockUseVlmModels.mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null
    });

    render(<VlmModelPicker value="" onChange={vi.fn()} fieldId="model" />);

    expect(screen.getByText(/Checking Ollama/i)).toBeInTheDocument();
  });
});
