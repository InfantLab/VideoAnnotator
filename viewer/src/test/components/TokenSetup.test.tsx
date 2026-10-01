import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';

import { TokenSetup } from '@/components/TokenSetup';

describe('TokenSetup', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('says how to get a key: the first-run link, generate-token, someone else, auth off', () => {
    render(<TokenSetup />);
    const help = screen.getByRole('heading', { name: 'How to get a key' }).parentElement!;
    expect(help).toHaveTextContent('[API KEY] VIDEOANNOTATOR API KEY GENERATED');
    expect(help).toHaveTextContent('/viewer-connect?token=');
    expect(help).toHaveTextContent('videoannotator generate-token');
    expect(help).toHaveTextContent('--user <your email>');
    expect(help).toHaveTextContent('AUTH_REQUIRED=false');
    expect(screen.queryByText(/dev-token/)).toBeNull();
  });

  it('defaults to the origin that served the page, not a fixed host', () => {
    render(<TokenSetup />);
    expect(screen.getByLabelText('Server URL')).toHaveValue('');
    expect(screen.getByRole('button', { name: 'Save Configuration' })).toBeEnabled();
  });

  it('catches a wrongly pasted key before it is saved', () => {
    render(<TokenSetup />);
    fireEvent.change(screen.getByLabelText(/API Token/), { target: { value: `Bearer va_${'a'.repeat(43)}` } });
    expect(screen.getByText(/without "Bearer "/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save Configuration' })).toBeDisabled();
  });
});
