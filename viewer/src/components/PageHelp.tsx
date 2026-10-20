import type { ReactNode } from 'react';
import { ExternalLink } from 'lucide-react';

import { docsUrl, type DocsPage } from '@/lib/docsLinks';

/** A page's one line of help and its link into the docs (spec 029 FR-023). */
export function PageHelp({ page, children }: { page: DocsPage; children: ReactNode }) {
  return (
    <p className="text-sm text-muted-foreground" data-testid="page-help">
      {children}{' '}
      <a
        href={docsUrl(page)}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-1 whitespace-nowrap text-primary underline-offset-4 hover:underline"
      >
        More in the docs
        <ExternalLink className="h-3 w-3" aria-hidden="true" />
      </a>
    </p>
  );
}
