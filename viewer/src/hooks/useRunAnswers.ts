import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/api/client';
import type { AskAnswer } from '@/types/ask';

export interface RunAnswersState {
  answers: AskAnswer[];
  planned: number;
  done: number;
  status: 'running' | 'done' | 'stopped' | 'failed';
  stoppedEarly: boolean;
}

export const RUN_POLL_MS = 2000;

/**
 * The answers of each run in `jobIds`, as they come (spec 027 SC-004: polled
 * every 2 s, asking only for answers after those already here).
 */
export function useRunAnswers(jobIds: string[]): Record<string, RunAnswersState> {
  const [state, setState] = useState<Record<string, RunAnswersState>>({});
  const latest = useRef(state);
  latest.current = state;
  const key = jobIds.join(',');

  useEffect(() => {
    if (!key) return;
    const ids = key.split(',');
    let cancelled = false;
    const poll = async () => {
      for (const id of ids) {
        const known = latest.current[id];
        if (known && known.status !== 'running') continue;
        try {
          const page = await apiClient.getRunAnswers(id, known?.answers.length ?? 0);
          if (cancelled) return;
          setState((prev) => {
            const before = prev[id]?.answers ?? [];
            return {
              ...prev,
              [id]: {
                answers: [...before, ...page.answers],
                planned: page.planned,
                done: page.done,
                status: page.status as RunAnswersState['status'],
                stoppedEarly: page.stopped_early,
              },
            };
          });
        } catch {
          // Next poll tries again; a run's answers are safe on the server.
        }
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), RUN_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [key]);

  return state;
}
