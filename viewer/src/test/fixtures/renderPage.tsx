// A test helper, never hot-reloaded.
/* eslint-disable react-refresh/only-export-components */
/**
 * A page at an address, with the viewer's providers and a stand-in for the
 * pages it links to that shows where it went and the router state it carried.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import type { ReactElement } from 'react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';

import { NotifyNavigator } from '@/components/NotifyNavigator';
import { TooltipProvider } from '@/components/ui/tooltip';

export function Landing() {
  const location = useLocation();
  return (
    <div data-testid="landed" data-state={JSON.stringify(location.state ?? null)}>
      {location.pathname + location.search}
    </div>
  );
}

export function renderAt(address: string, routes: { path: string; element: ReactElement }[], state?: unknown) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <TooltipProvider>
        <MemoryRouter initialEntries={[state === undefined ? address : { pathname: address.split('?')[0], search: address.includes('?') ? `?${address.split('?')[1]}` : '', state }]}>
          <NotifyNavigator />
          <Routes>
            {routes.map((r) => (
              <Route key={r.path} path={r.path} element={r.element} />
            ))}
            <Route path="*" element={<Landing />} />
          </Routes>
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  );
}

/** Where the last navigation landed, and its router state. */
export function landed(container: HTMLElement): { address: string; state: unknown } | null {
  const node = container.querySelector('[data-testid="landed"]');
  return node ? { address: node.textContent ?? '', state: JSON.parse(node.getAttribute('data-state') ?? 'null') } : null;
}
