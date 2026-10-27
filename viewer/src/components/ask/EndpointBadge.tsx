import { useState } from 'react';
import { HardDrive, Globe, X } from 'lucide-react';
import { badgeText, localNote, NOTE_DISMISSED_KEY } from '@/lib/endpointWords';
import type { ModelEndpoint } from '@/types/ask';

interface EndpointBadgeProps {
  endpoint: ModelEndpoint | undefined;
  model: string;
  /** The one-time note under a local badge; off where many badges are listed. */
  showNote?: boolean;
  /** Beside the model picker, left-aligned (UAT #25). */
  inline?: boolean;
}

const noteSeen = () => {
  try {
    return localStorage.getItem(NOTE_DISMISSED_KEY) === '1';
  } catch {
    return false;
  }
};

/** Where the model runs, always in view (FR-014); a remote one as a warning (FR-015). */
export function EndpointBadge({ endpoint, model, showNote: noteAllowed = true, inline = false }: EndpointBadgeProps) {
  const [showNote, setShowNote] = useState(() => !noteSeen());
  if (!endpoint) return null;
  const remote = endpoint.class === 'remote';

  const dismiss = () => {
    setShowNote(false);
    try {
      localStorage.setItem(NOTE_DISMISSED_KEY, '1');
    } catch {
      // Private window: shown again next time, which is harmless.
    }
  };

  return (
    <div className={inline ? 'flex flex-col items-start gap-1' : 'flex flex-col items-end gap-1'}>
      <span
        data-testid="endpoint-badge"
        data-class={endpoint.class}
        className={
          remote
            ? 'inline-flex items-center gap-1.5 rounded-full border border-amber-500 bg-amber-100 px-3 py-1 text-sm font-medium text-amber-900 dark:bg-amber-950 dark:text-amber-200'
            : 'inline-flex items-center gap-1.5 rounded-full border border-emerald-600/40 bg-emerald-50 px-3 py-1 text-sm text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200'
        }
      >
        {remote ? <Globe className="h-4 w-4" /> : <HardDrive className="h-4 w-4" />}
        {badgeText(endpoint, model)}
      </span>
      {!remote && noteAllowed && showNote && (
        <span className="inline-flex items-center gap-1 text-xs text-muted-foreground" data-testid="endpoint-note">
          {localNote(endpoint)}
          <button type="button" aria-label="Got it" onClick={dismiss} className="rounded p-0.5 hover:bg-muted">
            <X className="h-3 w-3" />
          </button>
        </span>
      )}
    </div>
  );
}
