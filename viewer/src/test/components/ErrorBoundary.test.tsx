// The app-wide error screen: a tab older than the current build gets told to
// reload (re-rendering can't fetch a file the server no longer has), and
// "Go Home" goes to the viewer's base, not the server root.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { isStaleBuildError, reloadForNewBuild } from '@/lib/staleBuild';

function Boom({ message }: { message: string }): React.ReactElement {
  throw new Error(message);
}

const STALE =
  'Failed to fetch dynamically imported module: http://127.0.0.1:18011/viewer/assets/Datasets-D0oG2TDN.js';

describe('staleBuild', () => {
  beforeEach(() => sessionStorage.clear());

  it('recognises each browser\'s failed-chunk message', () => {
    expect(isStaleBuildError(new Error(STALE))).toBe(true);
    expect(isStaleBuildError(new Error('Importing a module script failed.'))).toBe(true);
    expect(isStaleBuildError(new Error('error loading dynamically imported module'))).toBe(true);
    expect(isStaleBuildError(new Error('Cannot read properties of undefined'))).toBe(false);
  });

  it('reloads once, not in a loop', () => {
    const reload = vi.fn();
    vi.stubGlobal('location', { ...window.location, reload });

    expect(reloadForNewBuild()).toBe(true);
    expect(reloadForNewBuild()).toBe(false);
    expect(reload).toHaveBeenCalledTimes(1);

    vi.unstubAllGlobals();
  });
});

describe('ErrorBoundary', () => {
  let location: { href: string; reload: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    location = { href: 'http://127.0.0.1:18011/viewer/datasets', reload: vi.fn() };
    vi.stubGlobal('location', location);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('tells a stale tab to reload, and reloads', async () => {
    render(
      <ErrorBoundary>
        <Boom message={STALE} />
      </ErrorBoundary>
    );

    expect(screen.getByText(/updated since this page was opened/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /reload/i }));
    expect(location.reload).toHaveBeenCalled();
  });

  it('keeps Try Again for other errors', () => {
    render(
      <ErrorBoundary>
        <Boom message="Cannot read properties of undefined" />
      </ErrorBoundary>
    );

    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('Go Home goes to the viewer base, not the server root', async () => {
    render(
      <ErrorBoundary>
        <Boom message="anything" />
      </ErrorBoundary>
    );

    await userEvent.click(screen.getByRole('button', { name: /go home/i }));

    expect(location.href).toBe(import.meta.env.BASE_URL);
  });
});
