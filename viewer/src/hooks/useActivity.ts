import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import { activityTransitions, finishedBeforeLooking, loadAnnounced, saveAnnounced, type Transition } from '@/lib/activity';
import type { Activity } from '@/types/videos';

export const ACTIVE_POLL_MS = 5_000;
export const IDLE_POLL_MS = 30_000;

/**
 * The runs going on, polled every 5 s while any is active and every 30 s
 * otherwise (spec 029 R17), and what changed at the last poll.
 */
export function useActivity(enabled = true) {
  const query = useQuery<Activity>({
    queryKey: ['videos', 'activity'],
    queryFn: () => apiClient.getActivity(),
    enabled,
    refetchInterval: (q) => ((q.state.data?.active.length ?? 0) > 0 ? ACTIVE_POLL_MS : IDLE_POLL_MS),
    retry: false,
  });
  const previous = useRef<Activity | null>(null);
  const announced = useRef<Set<string>>(loadAnnounced());
  const [transitions, setTransitions] = useState<Transition[]>([]);

  const data = query.data;
  useEffect(() => {
    if (!data) return;
    if (!previous.current) for (const id of finishedBeforeLooking(data)) announced.current.add(id);
    const changes = activityTransitions(previous.current, data, announced.current);
    previous.current = data;
    for (const change of changes) {
      if (change.kind !== 'started') announced.current.add(change.run.id);
    }
    saveAnnounced(announced.current);
    if (changes.length) setTransitions(changes);
  }, [data]);

  return {
    active: data?.active ?? [],
    recent: data?.recent ?? [],
    transitions,
    isLoading: query.isLoading,
    error: query.error,
  };
}
