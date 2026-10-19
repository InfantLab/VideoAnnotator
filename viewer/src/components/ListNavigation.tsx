import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { useListContext } from '@/hooks/useListContext';
import { neighbourLink, positionIn, type ListContext, type ListItem } from '@/lib/listNavigation';

interface ListNavigationProps {
  context: ListContext;
  path: string;
  run?: string | null;
}

/**
 * "Peekaboo study · 3 of 48" with ‹ › through the list the video was opened
 * from: a run, a project or a folder (spec 028 US7). Lane settings carry over:
 * they're the researcher's, not the video's.
 */
export function ListNavigation({ context, path, run }: ListNavigationProps) {
  const navigate = useNavigate();
  const { data: list, error } = useListContext(context);
  const [opening, setOpening] = useState(false);
  const position = list ? positionIn(list.items, path, run) : null;

  const open = async (item: ListItem | null) => {
    if (!item) return;
    setOpening(true);
    try {
      const target = item.path ?? (await item.resolve?.()) ?? null;
      if (target) navigate(neighbourLink(item, target, context));
    } finally {
      setOpening(false);
    }
  };

  if (error) return null;
  const name = list?.name ?? '…';
  return (
    <nav aria-label="Videos in this list" className="flex min-w-0 items-center text-xs" data-testid="list-navigation">
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        disabled={!position?.previous || opening}
        onClick={() => void open(position?.previous ?? null)}
        aria-label="Previous video"
        title={position?.previous ? `Previous: ${position.previous.name}` : 'No earlier video to open'}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
      <span className="min-w-0 truncate" data-testid="list-position">
        {list?.href ? (
          <Link to={list.href} className="hover:underline">
            {name}
          </Link>
        ) : (
          name
        )}
        {position && (
          <span className="text-muted-foreground">
            {' '}
            · {position.position} of {position.total}
          </span>
        )}
      </span>
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        disabled={!position?.next || opening}
        onClick={() => void open(position?.next ?? null)}
        aria-label="Next video"
        title={position?.next ? `Next: ${position.next.name}` : 'No later video to open'}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </nav>
  );
}
