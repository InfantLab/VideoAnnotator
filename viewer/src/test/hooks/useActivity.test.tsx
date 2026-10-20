import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { useActivity } from '@/hooks/useActivity';
import { activityTransitions, progressLine } from '@/lib/activity';
import { activityRun, mockVideosApi } from '@/test/fixtures/videos';

const running = activityRun({ id: 'b2' });
const finished = (over = {}) => activityRun({ id: 'b2', done: 47, failed: 1, running: 0, waiting: 0, finished_at: '2026-10-20T10:00:00Z', ...over });

describe('activityTransitions', () => {
  it('says a run started', () => {
    expect(activityTransitions({ active: [], recent: [] }, { active: [running], recent: [] }, new Set())).toEqual([
      { kind: 'started', run: running },
    ]);
  });

  it('says a run finished, failed or was stopped', () => {
    const before = { active: [running], recent: [] };
    expect(activityTransitions(before, { active: [], recent: [finished()] }, new Set())[0].kind).toBe('finished');
    expect(activityTransitions(before, { active: [], recent: [finished({ done: 0, failed: 48 })] }, new Set())[0].kind).toBe('failed');
    expect(activityTransitions(before, { active: [], recent: [finished({ stopped: true, cancelled: 10 })] }, new Set())[0].kind).toBe('stopped');
  });

  it('reports a run that finished while the viewer was closed, once', () => {
    const next = { active: [], recent: [finished()] };
    expect(activityTransitions(null, next, new Set())).toHaveLength(1);
    expect(activityTransitions(null, next, new Set(['b2']))).toHaveLength(0);
  });

  it('sums progress', () => {
    expect(progressLine([running, activityRun({ id: 'b3', total: 2, done: 1, running: 1, waiting: 0 })])).toBe('2 runs · 32/50');
  });
});

describe('useActivity', () => {
  beforeEach(() => vi.mocked(localStorage.setItem).mockClear());

  it('polls and exposes the transitions', async () => {
    const api = mockVideosApi({ activity: { active: [], recent: [finished()] } });
    const client = new QueryClient();
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;

    const { result } = renderHook(() => useActivity(), { wrapper });

    await waitFor(() => expect(result.current.transitions).toHaveLength(1));
    expect(result.current.transitions[0].kind).toBe('finished');
    expect(api.getActivity).toHaveBeenCalled();
    expect(localStorage.setItem).toHaveBeenCalledWith('videoannotator_announced_runs', '["b2"]');
  });
});
