import { AlertTriangle, Shuffle } from 'lucide-react';
import { alignRows, framesSentNote, repeatsDiffer } from '@/lib/ask';
import { formatClock } from '@/lib/lanes';
import { cn } from '@/lib/utils';
import type { AskAnswer } from '@/types/ask';

export interface AnswerColumn {
  key: string;
  title: string;
  subtitle: string;
  answers: AskAnswer[];
  /** Frames per sample asked for, to say when fewer were sent. */
  frames: number;
}

interface AnswersTableProps {
  columns: AnswerColumn[];
  highlight: number | null;
  onSeek: (t: number) => void;
}

function AnswerCell({ answers, frames }: { answers: AskAnswer[]; frames: number }) {
  if (!answers.length) return <span className="text-muted-foreground">·</span>;
  const last = answers[answers.length - 1];
  const differs = repeatsDiffer(answers);
  const note = framesSentNote(last, frames);
  return (
    <span className="inline-flex items-start gap-1">
      {last.status === 'error' ? (
        <span className="inline-flex items-center gap-1 text-destructive" title={last.error ?? undefined}>
          <AlertTriangle className="h-3 w-3" /> no answer
        </span>
      ) : last.status === 'skipped' ? (
        <span className="text-muted-foreground italic" title={last.error ?? undefined}>
          skipped
        </span>
      ) : last.fit === false ? (
        <span className="text-amber-700 dark:text-amber-400" data-testid="misfit">
          Didn't fit: {last.fit_problem}
          {last.raw_response && (
            <details className="text-xs text-muted-foreground" onClick={(e) => e.stopPropagation()}>
              <summary className="cursor-pointer">the reply</summary>
              <span className="whitespace-pre-wrap">{last.raw_response}</span>
            </details>
          )}
        </span>
      ) : last.fields ? (
        <span className="flex flex-col" data-testid="answer-fields">
          {Object.entries(last.fields).map(([name, value]) => (
            <span key={name} className="text-xs">
              <span className="text-muted-foreground">{name}</span> <span className="font-medium">{String(value)}</span>
            </span>
          ))}
        </span>
      ) : (
        <span className="font-medium">{last.label}</span>
      )}
      {note && <span className="text-xs text-muted-foreground">({note})</span>}
      {differs && (
        <span
          data-testid="repeat-differs"
          className="text-amber-600"
          title={`Tried ${answers.length} times: ${answers.map((a) => (a.status === 'answered' ? a.label : a.status)).join(', ')}`}
        >
          <Shuffle className="h-3 w-3" />
        </span>
      )}
    </span>
  );
}

/** One row per sample time; one answer column per shown entry (FR-007). */
export function AnswersTable({ columns, highlight, onSeek }: AnswersTableProps) {
  const rows = alignRows(columns.map((c) => c.answers));
  const single = columns.length === 1;
  if (!rows.length) {
    return <p className="p-4 text-sm text-muted-foreground">No answers yet. Try the question on a frame.</p>;
  }
  return (
    <div className="max-h-80 overflow-auto rounded border">
      <table className="w-full text-sm" data-testid="answers-table">
        <thead className="sticky top-0 bg-muted/80 text-left text-xs">
          <tr>
            <th className="px-2 py-1 font-medium">Time</th>
            <th className="px-2 py-1 font-medium">Frames</th>
            {columns.map((c) => (
              <th key={c.key} className="px-2 py-1 font-medium" title={c.subtitle}>
                {single ? 'Answer' : c.title}
              </th>
            ))}
            {single && <th className="px-2 py-1 font-medium">Reasoning</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const first = row.cells.find((c) => c.length)?.[0];
            const selected = highlight !== null && Math.abs(row.t - highlight) < 1e-3;
            return (
              <tr
                key={row.t}
                data-testid="answer-row"
                data-selected={selected ? 'true' : 'false'}
                className={cn('cursor-pointer border-t hover:bg-muted/50', selected && 'bg-primary/10')}
                onClick={() => onSeek(row.t)}
              >
                <td className="whitespace-nowrap px-2 py-1 tabular-nums">{formatClock(row.t)}</td>
                <td className="whitespace-nowrap px-2 py-1 text-xs text-muted-foreground tabular-nums">
                  {first?.frame_numbers?.join(', ')}
                </td>
                {row.cells.map((answers, i) => (
                  <td key={columns[i].key} className="px-2 py-1 align-top" title={answers[answers.length - 1]?.reasoning ?? undefined}>
                    <AnswerCell answers={answers} frames={columns[i].frames} />
                  </td>
                ))}
                {single && (
                  <td className="max-w-md px-2 py-1 text-xs text-muted-foreground">
                    <span className="line-clamp-2 whitespace-pre-wrap">
                      {row.cells[0][row.cells[0].length - 1]?.reasoning ?? ''}
                    </span>
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
