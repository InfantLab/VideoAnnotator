import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { TokenSetup } from '@/components/TokenSetup';
import { checkConnection } from '@/lib/apiConnection';

const KEY = `va_${'a'.repeat(43)}`;

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
    expect(screen.getByLabelText('Server address')).toHaveValue('');
    expect(screen.getByRole('button', { name: 'Connect' })).toBeEnabled();
  });

  it('catches a wrongly pasted key before it is saved', () => {
    render(<TokenSetup />);
    fireEvent.change(screen.getByLabelText('Key'), { target: { value: `Bearer ${KEY}` } });
    expect(screen.getByText(/without "Bearer "/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Connect' })).toBeDisabled();
  });

  it('shows whose key it is and whether it is an administrator, without a test step (UAT #63)', () => {
    localStorage.setItem('videoannotator_api_token', KEY);
    render(<TokenSetup connection={{ state: 'connected', user: { name: 'caspar', isAdmin: false } }} />);
    expect(screen.getByTestId('key-state')).toHaveTextContent('Connected as caspar · not an administrator');
    expect(screen.queryByText(/Test Connection/)).toBeNull();
    expect(screen.queryByRole('button', { name: 'Connect' })).toBeNull();
    fireEvent.change(screen.getByLabelText('Key'), { target: { value: `va_${'b'.repeat(43)}` } });
    expect(screen.getByRole('button', { name: 'Connect' })).toBeEnabled();
  });
});

describe('checkConnection', () => {
  afterEach(() => vi.unstubAllGlobals());

  const serve = (routes: Record<string, { status: number; body?: unknown }>) => {
    const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
      const path = Object.keys(routes).find((p) => url.endsWith(p));
      if (!path) throw new TypeError('Failed to fetch');
      const { status, body } = routes[path];
      return new Response(JSON.stringify(body ?? {}), { status });
    });
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  };

  it('checks the key that was typed, and says whose it is', async () => {
    const fetchMock = serve({
      '/api/v1/system/health': { status: 200, body: { security: { auth_required: true } } },
      '/api/v1/auth/me': { status: 200, body: { username: 'caspar', is_admin: true } },
    });
    expect(await checkConnection('http://127.0.0.1:18011', KEY)).toEqual({
      ok: true,
      user: { name: 'caspar', isAdmin: true },
      authRequired: true,
    });
    expect(fetchMock.mock.calls[1][1]).toEqual({ headers: { Authorization: `Bearer ${KEY}` } });
  });

  it('tells a refused key from a missing server and a missing key', async () => {
    serve({
      '/api/v1/system/health': { status: 200, body: { security: { auth_required: true } } },
      '/api/v1/auth/me': { status: 401 },
    });
    expect(await checkConnection('', KEY)).toMatchObject({ ok: false, problem: 'key-refused' });
    expect(await checkConnection('', '')).toMatchObject({ ok: false, problem: 'needs-key' });
    serve({});
    expect(await checkConnection('', KEY)).toMatchObject({ ok: false, problem: 'no-server' });
  });
});
