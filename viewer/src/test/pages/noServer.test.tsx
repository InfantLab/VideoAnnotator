/** Pages that need a server say so, and what works without one (spec 029 FR-026, T056). */
import { screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { getApiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import NewRun from '@/pages/NewRun';
import Pipelines from '@/pages/Pipelines';
import Videos from '@/pages/Videos';
import { renderAt } from '@/test/fixtures/renderPage';

const offline = () => new APIError('Network error: Failed to fetch', 0);

afterEach(() => vi.restoreAllMocks());

function noServer() {
  vi.spyOn(console, 'log').mockImplementation(() => {});
  const client = Object.getPrototypeOf(getApiClient());
  for (const name of ['listVideos', 'getPipelineCatalog', 'getActivity', 'listPresets', 'getCurrentUser', 'getExtrasGroups', 'getIngestAccess', 'listDatasets']) {
    vi.spyOn(client, name).mockRejectedValue(offline());
  }
}

describe.each([
  ['/videos', <Videos key="v" />],
  ['/run', <NewRun key="r" />],
  ['/pipelines', <Pipelines key="p" />],
])('%s without a server', (path, element) => {
  it('says it needs one, how to connect, and offers files and demos', async () => {
    noServer();
    renderAt(path, [{ path, element }]);
    const card = await screen.findByTestId('needs-server');
    expect(within(card).getByRole('link', { name: 'Connect' })).toHaveAttribute('href', '/settings');
    expect(within(card).getByRole('link', { name: /Open files from this computer/ })).toBeInTheDocument();
    expect(within(card).getByRole('link', { name: /Demos/ })).toBeInTheDocument();
  });
});
