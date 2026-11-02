import { Link } from 'react-router-dom';
import { ExternalLink } from 'lucide-react';

import { ReportProblemButton } from '@/components/ReportProblem';
import { useServerVersionShown } from '@/hooks/useServerVersionShown';
import { DOCS_BASE } from '@/lib/docsLinks';
import { cn } from '@/lib/utils';
import { GITHUB_URL, VERSION, serverDiffers } from '@/utils/version';

// The concept DOI: it always resolves to the latest version.
export const CITE_URL = 'https://doi.org/10.5281/zenodo.16961751';

const linkClass = 'inline-flex items-center gap-1 hover:text-foreground hover:underline';

/** One line on every page, each link once (UAT #69). */
export const Footer = () => {
  const server = useServerVersionShown();
  const differs = server ? serverDiffers(server) : null;
  return (
    <footer className="shrink-0 border-t bg-background/95 px-4 py-1.5 text-xs text-muted-foreground" data-testid="footer">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <span>
          VideoAnnotator {VERSION}
          {server && (
            <span className={cn(differs && 'text-amber-600 dark:text-amber-400')} title={differs ? undefined : 'The server this viewer uses'}>
              {' · '}server {server}
              {differs && `: ${differs} than this viewer`}
            </span>
          )}
        </span>
        <nav aria-label="About VideoAnnotator" className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <Link to="/help" className={linkClass}>
            Help
          </Link>
          <a href={DOCS_BASE} target="_blank" rel="noopener noreferrer" className={linkClass}>
            Docs <ExternalLink className="h-2.5 w-2.5" aria-hidden="true" />
          </a>
          <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className={linkClass}>
            Source <ExternalLink className="h-2.5 w-2.5" aria-hidden="true" />
          </a>
          <a href={CITE_URL} target="_blank" rel="noopener noreferrer" className={linkClass} title="VideoAnnotator's DOI on Zenodo">
            How to cite <ExternalLink className="h-2.5 w-2.5" aria-hidden="true" />
          </a>
          <ReportProblemButton className="h-auto p-0 text-xs font-normal text-muted-foreground hover:text-foreground" />
        </nav>
      </div>
    </footer>
  );
};
