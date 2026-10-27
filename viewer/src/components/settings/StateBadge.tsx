// A state shown as a state, not a word in a sentence (UAT #14, #63): the same
// badge for the key, the Hugging Face token and each Ask AI server.
import type { ReactNode } from 'react';
import { CheckCircle2, CircleDashed, Loader2, TriangleAlert, XCircle } from 'lucide-react';

export type StateTone = 'ok' | 'warn' | 'bad' | 'off' | 'checking';

const TONES: Record<StateTone, { className: string; Icon: typeof CheckCircle2 }> = {
  ok: { className: 'border-green-600/40 bg-green-50 text-green-800 dark:bg-green-950/40 dark:text-green-300', Icon: CheckCircle2 },
  warn: { className: 'border-amber-500/50 bg-amber-50 text-amber-900 dark:bg-amber-950/40 dark:text-amber-200', Icon: TriangleAlert },
  bad: { className: 'border-red-500/50 bg-red-50 text-red-800 dark:bg-red-950/40 dark:text-red-300', Icon: XCircle },
  off: { className: 'border-border bg-muted text-muted-foreground', Icon: CircleDashed },
  checking: { className: 'border-border bg-muted text-muted-foreground', Icon: Loader2 },
};

export function StateBadge({ tone, children, testId }: { tone: StateTone; children: ReactNode; testId?: string }) {
  const { className, Icon } = TONES[tone];
  return (
    <span
      data-testid={testId}
      data-tone={tone}
      className={`inline-flex max-w-full items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-sm font-medium ${className}`}
    >
      <Icon className={`h-4 w-4 shrink-0 ${tone === 'checking' ? 'animate-spin' : ''}`} aria-hidden="true" />
      <span className="truncate">{children}</span>
    </span>
  );
}
