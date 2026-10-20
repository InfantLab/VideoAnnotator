import { useCallback, useMemo } from 'react';

import { usePipelineCatalog } from '@/hooks/usePipelineCatalog';

const humanise = (id: string) => id.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());

/** "Faces" from the guide's "Faces: where they are and their expression" (spec 026). */
const shortTitle = (title: string) => title.split(':')[0].trim();

/** Each pipeline's short guide title, in the catalog's order. */
export function usePipelineTitles(enabled = true) {
  const { data } = usePipelineCatalog({ enabled });
  const pipelines = data?.catalog.pipelines;
  const titles = useMemo(
    () => new Map((pipelines ?? []).map((p) => [p.id, p.title ? shortTitle(p.title) : p.name || humanise(p.id)])),
    [pipelines],
  );
  const order = useMemo(() => (pipelines ?? []).map((p) => p.id), [pipelines]);
  const titleOf = useCallback((id: string) => titles.get(id) ?? humanise(id), [titles]);
  return { titleOf, order };
}
