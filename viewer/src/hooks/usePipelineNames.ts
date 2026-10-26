import { useCallback, useMemo } from 'react';

import { usePipelineCatalog } from '@/hooks/usePipelineCatalog';

const humanise = (id: string) => id.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());

/** "who spoke when" from the guide's "Speaker turns: who spoke when" (spec 026). */
export const taskOfTitle = (title: string | null | undefined): string => {
  if (!title) return '';
  const at = title.indexOf(':');
  return at < 0 ? '' : title.slice(at + 1).trim();
};

/**
 * Each pipeline named by what runs, then what it finds (viewer design D1, UAT #9):
 * `nameOf` is the registry's display name ("pyannote community-1"), `taskOf` the
 * task ("who spoke when"). `order` is the catalog's.
 */
export function usePipelineNames(enabled = true) {
  const { data } = usePipelineCatalog({ enabled });
  const pipelines = data?.catalog.pipelines;
  const byId = useMemo(() => new Map((pipelines ?? []).map((p) => [p.id, p])), [pipelines]);
  const order = useMemo(() => (pipelines ?? []).map((p) => p.id), [pipelines]);
  const nameOf = useCallback((id: string) => byId.get(id)?.name || humanise(id), [byId]);
  const taskOf = useCallback((id: string) => taskOfTitle(byId.get(id)?.title), [byId]);
  return { nameOf, taskOf, order };
}
