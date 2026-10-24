import { useState } from 'react';
import { ArrowDown, ArrowUp, Loader2, Plus, Sparkles, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { KIND_WORDS, KINDS, shapeProblem } from '@/lib/answerShape';
import type { AnswerShape, AnswerShapeField, SuggestShapeResponse } from '@/types/ask';

interface AnswerShapeEditorProps {
  /** null: a free-text answer. */
  value: AnswerShape | null;
  onChange: (shape: AnswerShape | null) => void;
  /** Ask for a suggestion; absent where there's no model to ask (the run settings form). */
  onSuggest?: () => Promise<SuggestShapeResponse>;
  /** Set when the chosen server is known not to enforce shapes: its name. */
  cantEnforce?: string | null;
  fieldId?: string;
}

const blankField = (shape: AnswerShape): AnswerShapeField => {
  const used = new Set(shape.map((f) => f.name));
  let n = shape.length + 1;
  while (used.has(n === 1 ? 'answer' : `answer_${n}`)) n += 1;
  return { name: n === 1 ? 'answer' : `answer_${n}`, kind: 'yes_no' };
};

function Values({ field, onChange, label }: { field: AnswerShapeField; onChange: (values: string[]) => void; label: string }) {
  const [text, setText] = useState('');
  const values = field.values ?? [];
  const add = () => {
    const parts = text
      .split(',')
      .map((v) => v.trim())
      .filter((v) => v && !values.includes(v));
    if (parts.length) onChange([...values, ...parts]);
    setText('');
  };
  return (
    <div className="flex flex-wrap items-center gap-1">
      {values.map((v) => (
        <span key={v} className="inline-flex items-center gap-0.5 rounded-full border bg-muted px-2 py-0.5 text-xs">
          {v}
          <button type="button" aria-label={`Remove ${v}`} onClick={() => onChange(values.filter((x) => x !== v))}>
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}
      <Input
        aria-label={label}
        className="h-7 w-36 text-xs"
        placeholder="add a value"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={add}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            add();
          }
        }}
      />
    </div>
  );
}

const asNumber = (text: string): number | null => (text.trim() === '' || !Number.isFinite(Number(text)) ? null : Number(text));

/** The answer's shape: free text, or named fields (spec 033 FR-018). */
export function AnswerShapeEditor({ value, onChange, onSuggest, cantEnforce, fieldId = 'answer-shape' }: AnswerShapeEditorProps) {
  const [proposal, setProposal] = useState<SuggestShapeResponse | null>(null);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestError, setSuggestError] = useState<string | null>(null);
  const shape = value ?? [];
  const problem = value ? shapeProblem(value) : null;

  const update = (index: number, change: Partial<AnswerShapeField>) =>
    onChange(shape.map((f, i) => (i === index ? { ...f, ...change } : f)));
  const move = (index: number, by: number) => {
    const next = [...shape];
    const [field] = next.splice(index, 1);
    next.splice(index + by, 0, field);
    onChange(next);
  };

  const suggest = async () => {
    if (!onSuggest) return;
    setSuggesting(true);
    setSuggestError(null);
    try {
      setProposal(await onSuggest());
    } catch (e) {
      setSuggestError(e instanceof Error ? e.message : String(e));
    } finally {
      setSuggesting(false);
    }
  };

  return (
    <div className="space-y-2" data-testid="answer-shape">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Label id={`${fieldId}-label`}>Answer shape</Label>
        <div className="flex gap-1" role="group" aria-labelledby={`${fieldId}-label`}>
          <Button type="button" size="sm" variant={value ? 'outline' : 'secondary'} aria-pressed={!value} onClick={() => onChange(null)}>
            Free text
          </Button>
          <Button
            type="button"
            size="sm"
            variant={value ? 'secondary' : 'outline'}
            aria-pressed={Boolean(value)}
            onClick={() => !value && onChange([blankField([])])}
          >
            Fields
          </Button>
          {onSuggest && (
            <Button type="button" size="sm" variant="ghost" className="gap-1" disabled={suggesting} onClick={() => void suggest()}>
              {suggesting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />} Suggest
            </Button>
          )}
        </div>
      </div>

      {proposal && (
        <div className="space-y-1 rounded border border-dashed p-2 text-sm" data-testid="shape-proposal">
          {proposal.shape ? (
            <>
              <p className="text-xs text-muted-foreground">
                {proposal.made_by === 'rule' ? 'Suggested from your question' : 'Suggested by the model'}:
              </p>
              <ul className="text-xs">
                {(proposal.shape as unknown as AnswerShape).map((f) => (
                  <li key={f.name}>
                    <span className="font-mono">{f.name}</span> · {KIND_WORDS[f.kind]}
                    {f.values?.length ? `: ${f.values.join(', ')}` : ''}
                    {f.kind === 'number' && (f.min != null || f.max != null) ? ` ${f.min ?? ''}–${f.max ?? ''}` : ''}
                  </li>
                ))}
              </ul>
              <div className="flex gap-1">
                <Button
                  type="button"
                  size="sm"
                  onClick={() => {
                    onChange(proposal.shape as unknown as AnswerShape);
                    setProposal(null);
                  }}
                >
                  Use these fields
                </Button>
                <Button type="button" size="sm" variant="ghost" onClick={() => setProposal(null)}>
                  Discard
                </Button>
              </div>
            </>
          ) : (
            <p className="text-xs text-muted-foreground">
              {proposal.message}{' '}
              <button type="button" className="underline" onClick={() => setProposal(null)}>
                OK
              </button>
            </p>
          )}
        </div>
      )}
      {suggestError && <p className="text-xs text-destructive">Couldn't suggest fields: {suggestError}</p>}

      {value && (
        <div className="space-y-2">
          {shape.map((field, i) => (
            <div key={i} className="space-y-1 rounded border p-2" data-testid="shape-field">
              <div className="flex flex-wrap items-center gap-1">
                <Input
                  aria-label={`Field ${i + 1} name`}
                  className="h-8 w-36 font-mono text-xs"
                  value={field.name}
                  onChange={(e) => update(i, { name: e.target.value })}
                />
                <select
                  aria-label={`Field ${i + 1} kind`}
                  className="h-8 rounded-md border bg-background px-2 text-xs"
                  value={field.kind}
                  onChange={(e) => update(i, { kind: e.target.value as AnswerShapeField['kind'] })}
                >
                  {KINDS.map((k) => (
                    <option key={k} value={k}>
                      {KIND_WORDS[k]}
                    </option>
                  ))}
                </select>
                <span className="ml-auto flex">
                  <Button type="button" size="icon" variant="ghost" className="h-7 w-7" aria-label={`Move field ${i + 1} up`} disabled={i === 0} onClick={() => move(i, -1)}>
                    <ArrowUp className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7"
                    aria-label={`Move field ${i + 1} down`}
                    disabled={i === shape.length - 1}
                    onClick={() => move(i, 1)}
                  >
                    <ArrowDown className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7"
                    aria-label={`Remove field ${i + 1}`}
                    onClick={() => onChange(shape.length > 1 ? shape.filter((_, j) => j !== i) : null)}
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>
                </span>
              </div>
              {field.kind === 'choice' && (
                <Values field={field} label={`Field ${i + 1} values`} onChange={(values) => update(i, { values })} />
              )}
              {field.kind === 'number' && (
                <div className="flex items-center gap-1 text-xs">
                  from
                  <Input
                    aria-label={`Field ${i + 1} lowest`}
                    type="number"
                    className="h-7 w-20 text-xs"
                    value={field.min ?? ''}
                    onChange={(e) => update(i, { min: asNumber(e.target.value) })}
                  />
                  to
                  <Input
                    aria-label={`Field ${i + 1} highest`}
                    type="number"
                    className="h-7 w-20 text-xs"
                    value={field.max ?? ''}
                    onChange={(e) => update(i, { max: asNumber(e.target.value) })}
                  />
                </div>
              )}
            </div>
          ))}
          <Button type="button" size="sm" variant="outline" className="gap-1" disabled={shape.length >= 12} onClick={() => onChange([...shape, blankField(shape)])}>
            <Plus className="h-3.5 w-3.5" /> Add a field
          </Button>
          {problem && (
            <p className="text-xs text-destructive" role="alert">
              {problem}
            </p>
          )}
          {cantEnforce && (
            <p className="text-xs text-muted-foreground">
              {cantEnforce} can't enforce the answer shape; answers are checked after.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
