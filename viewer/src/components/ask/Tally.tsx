import { Lightbulb } from 'lucide-react';
import { ALLOWED_ANSWERS_HINT, MANY_ANSWERS, tally } from '@/lib/ask';
import { answerText, readJson } from '@/lib/answerColours';
import { cn } from '@/lib/utils';
import type { AskAnswer } from '@/types/ask';

interface TallyProps {
  answers: AskAnswer[];
  /** Each answer's colour, as on the timeline (UAT #23). */
  colour?: (answer: Pick<AskAnswer, 'label' | 'fields'>) => string;
  /** The answer at the selected moment: its line is marked (UAT #24). */
  selected?: AskAnswer | null;
}

/** Distinct answers with counts; errors and skipped samples are not answers (FR-009). */
/** Answers in fields, counted field by field: "Adults 1 ×4 · Child smiling no ×4" (UAT #24). */
function fieldCounts(answers: AskAnswer[]): [string, [string, number][]][] | null {
  const counts = new Map<string, Map<string, number>>();
  for (const a of answers) {
    if (a.status !== 'answered' || a.fit === false) continue;
    const fields = a.fields ?? readJson(a.label);
    if (!fields) return null;
    for (const [name, value] of Object.entries(fields)) {
      const values = counts.get(name) ?? new Map<string, number>();
      values.set(String(value), (values.get(String(value)) ?? 0) + 1);
      counts.set(name, values);
    }
  }
  if (counts.size < 2) return null;
  return [...counts].map(([name, values]) => [name, [...values].sort((x, y) => y[1] - x[1] || x[0].localeCompare(y[0]))]);
}

export function Tally({ answers, colour, selected }: TallyProps) {
  const t = tally(answers);
  if (!answers.length) return null;
  const byField = fieldCounts(answers);
  const selectedFields = selected ? selected.fields ?? readJson(selected.label) : null;
  return (
    <div className="space-y-2" data-testid="tally">
      {byField?.map(([name, values], i) => (
        <div key={name} className="space-y-1" data-testid="tally-field">
          <p className="text-xs text-muted-foreground">{name}</p>
          <ul className="flex flex-wrap gap-1.5 text-sm">
            {values.map(([value, count]) => {
              const isSelected = selectedFields != null && String(selectedFields[name]) === value;
              return (
                <li
                  key={value}
                  className={cn('inline-flex items-center gap-1 rounded border px-2 py-0.5', isSelected && 'border-foreground bg-muted')}
                  data-testid="tally-item"
                  data-selected={isSelected ? 'true' : 'false'}
                >
                  {colour && i === 0 && (
                    <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: colour({ label: value, fields: null }) }} aria-hidden="true" />
                  )}
                  <span className="font-medium">{value || '(empty)'}</span> <span className="text-muted-foreground">{count}</span>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
      <ul className="flex flex-wrap gap-1.5 text-sm">
        {!byField && t.answers.map(([label, count]) => {
          const isSelected = !!selected && selected.status === 'answered' && answerText(selected) === label;
          return (
            <li
              key={label}
              className={cn('inline-flex items-center gap-1 rounded border px-2 py-0.5', isSelected && 'border-foreground bg-muted')}
              data-testid="tally-item"
              data-selected={isSelected ? 'true' : 'false'}
            >
              {colour && <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: colour({ label, fields: null }) }} aria-hidden="true" />}
              <span className="font-medium">{label || '(empty)'}</span> <span className="text-muted-foreground">{count}</span>
            </li>
          );
        })}
        {t.errors > 0 && (
          <li className="rounded border border-dashed px-2 py-0.5 text-muted-foreground" data-testid="tally-errors">
            no answer {t.errors}
          </li>
        )}
        {t.misfits > 0 && (
          <li className="rounded border border-amber-500/60 px-2 py-0.5 text-amber-700 dark:text-amber-400" data-testid="tally-misfits">
            didn't fit {t.misfits}
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
