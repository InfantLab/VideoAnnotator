import { render } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { act } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const toast = vi.hoisted(() => Object.assign(vi.fn(() => 1), { success: vi.fn(() => 2), error: vi.fn(() => 3) }));
vi.mock('sonner', () => ({ toast }));

import { NotifyNavigator } from '@/components/NotifyNavigator';
import { notify } from '@/lib/notify';

function Where() {
  return <div data-testid="where">{useLocation().pathname + useLocation().search}</div>;
}

describe('notify', () => {
  beforeEach(() => vi.clearAllMocks());

  it('shows a title, a line and a next step', () => {
    notify({ title: 'Run started on 3 videos', body: 'You can close this tab.', action: { label: 'Show run', to: '/videos?run=b1' } });

    expect(toast).toHaveBeenCalledWith(
      'Run started on 3 videos',
      expect.objectContaining({ description: 'You can close this tab.', action: expect.objectContaining({ label: 'Show run' }) }),
    );
  });

  it('uses the tone', () => {
    notify({ title: 'Saved', tone: 'success' });
    notify({ title: 'Failed', tone: 'error' });

    expect(toast.success).toHaveBeenCalledWith('Saved', expect.anything());
    expect(toast.error).toHaveBeenCalledWith('Failed', expect.objectContaining({ duration: 10_000 }));
  });

  it("goes to the action's address within the viewer", () => {
    const view = render(
      <MemoryRouter initialEntries={['/home']}>
        <NotifyNavigator />
        <Routes>
          <Route path="*" element={<Where />} />
        </Routes>
      </MemoryRouter>,
    );
    notify({ title: 'Done', action: { label: 'Results table', to: '/videos?run=b1' } });
    const options = (toast.mock.calls.at(-1) as unknown as [string, { action: { onClick: () => void } }])[1];

    act(() => options.action.onClick());

    expect(view.getByTestId('where').textContent).toBe('/videos?run=b1');
  });
});
