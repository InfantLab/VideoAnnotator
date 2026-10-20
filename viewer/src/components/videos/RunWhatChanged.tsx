import { WhatChanged } from '@/components/videos/WhatChanged';
import { useVideos } from '@/hooks/useVideos';
import type { RunSummary } from '@/types/videos';

/** What changed? from a run's header: against its project's other runs. */
export function RunWhatChanged({ run }: { run: RunSummary }) {
  const { data } = useVideos({ project: run.project?.id }, !!run.project);
  const runs = data?.runs ?? [];
  const other = runs.find((r) => r.id !== run.id);
  if (!run.project || !other) return null;
  return <WhatChanged runs={runs} initial={[run.id, other.id]} />;
}
