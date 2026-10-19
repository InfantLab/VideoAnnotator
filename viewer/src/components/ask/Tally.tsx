import { Lightbulb } from 'lucide-react';
import { ALLOWED_ANSWERS_HINT, MANY_ANSWERS, tally } from '@/lib/ask';
import type { AskAnswer } from '@/types/ask';

interface TallyProps {
  answers: AskAnswer[];
}

/** Distinct answers with counts; errors and skipped samples are not answers (FR-009). */
export function Tally({ answers }: TallyProps) {
  const t = tally(answers);
  if (!answers.length) return null;
  return (
    <div className="space-y-2" data-testid="tally">
      <ul className="flex flex-wrap gap-1.5 text-sm">
        {t.answers.map(([label, count]) => (
          <li key={label} className="rounded border px-2 py-0.5" data-testid="tally-item">
            <span className="font-medium">{label || '(empty)'}</span> <span className="text-muted-foreground">{count}</span>
          </li>
        ))}
        {t.errors > 0 && (
          <li className="rounded border border-destructive/50 px-2 py-0.5 text-destructive" data-testid="tally-errors">
            no answer {t.errors}
          </li>
        )}
        {t.skipped > 0 && (
          <li className="rounded border border-dashed px-2 py-0.5 text-muted-foreground" data-testid="tally-skipped">
            skipped {t.skipped}
          </li>
        )}
      </ul>
      {t.answers.length > MANY_ANSWERS && (
        <p className="flex items-start gap-1 text-xs text-amber-700 dark:text-amber-400" data-testid="tally-hint">
          <Lightbulb className="mt-0.5 h-3 w-3 shrink-0" /> {t.answers.length} different answers. {ALLOWED_ANSWERS_HINT}
        </p>
      )}
    </div>
  );
}
