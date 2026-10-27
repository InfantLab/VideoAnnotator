import { useState } from 'react';
import { Check, Eye, EyeOff, Pencil, Star, Trash2 } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { entryTitle, samplingSummary } from '@/lib/ask';
import { cn } from '@/lib/utils';
import { formatWhen } from '@/lib/when';
import type { AskEntry, AskLibraryPrompt } from '@/types/ask';

export interface HistoryFilters {
  thisVideo: boolean;
  mine: boolean;
}

interface HistoryListProps {
  entries: AskEntry[];
  library: AskLibraryPrompt[];
  filters: HistoryFilters;
  onFilters: (filters: HistoryFilters) => void;
  shown: string[];
  canShow: (entry: AskEntry) => boolean;
  restoredId: string | null;
  onRestore: (entry: AskEntry) => void;
  onToggleShow: (entry: AskEntry) => void;
  onRename: (entry: AskEntry, name: string) => void;
  onStar: (entry: AskEntry, starred: boolean) => void;
  onRestoreLibrary: (prompt: AskLibraryPrompt) => void;
  onLibraryUpdate: (prompt: AskLibraryPrompt, change: { name?: string; starred?: boolean; hidden?: boolean }) => void;
  onLibraryDelete: (prompt: AskLibraryPrompt) => void;
  /** Delete an entry; `alsoQuestion` removes its question from the earlier questions too (UAT #65). */
  onDelete: (entry: AskEntry, alsoQuestion: boolean) => void;
}

const when = (iso: string | null | undefined) => formatWhen(iso);

function DeleteEntry({ entry, onDelete }: { entry: AskEntry; onDelete: HistoryListProps['onDelete'] }) {
  const [open, setOpen] = useState(false);
  const [alsoQuestion, setAlsoQuestion] = useState(true);
  return (
    <>
      <Button size="icon" variant="ghost" className="h-6 w-6" aria-label="Delete" onClick={() => setOpen(true)}>
        <Trash2 className="h-3 w-3" />
      </Button>
      <AlertDialog open={open} onOpenChange={setOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{entryTitle(entry)}”?</AlertDialogTitle>
            <AlertDialogDescription>
              {entry.kind === 'run'
                ? 'It leaves this history; the run and its results stay on the Runs page.'
                : 'Its answers are deleted.'}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <label className="flex items-center gap-2 text-sm">
            <Checkbox checked={alsoQuestion} onCheckedChange={(on) => setAlsoQuestion(on === true)} />
            Also remove the question from earlier questions
          </label>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep it</AlertDialogCancel>
            <AlertDialogAction onClick={() => onDelete(entry, alsoQuestion)}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

function Rename({ initial, onDone }: { initial: string; onDone: (name: string | null) => void }) {
  const [value, setValue] = useState(initial);
  return (
    <form
      className="flex gap-1"
      onSubmit={(e) => {
        e.preventDefault();
        onDone(value);
      }}
    >
      <Input aria-label="Name" autoFocus value={value} onChange={(e) => setValue(e.target.value)} className="h-7 text-xs" />
      <Button type="submit" size="icon" variant="ghost" className="h-7 w-7" aria-label="Save name">
        <Check className="h-3 w-3" />
      </Button>
    </form>
  );
}

/**
 * Every try and run, kept on the server and shared by everyone on it (FR-006).
 * Choosing one puts its question, model and sampling back; it doesn't change it.
 */
export function HistoryList(props: HistoryListProps) {
  const { entries, library, filters, onFilters, shown, canShow, restoredId } = props;
  const [renaming, setRenaming] = useState<string | null>(null);

  return (
    <div className="space-y-2" data-testid="history">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <div className="flex items-center gap-1.5">
          <Switch
            id="history-this-video"
            checked={filters.thisVideo}
            onCheckedChange={(on) => onFilters({ ...filters, thisVideo: on })}
          />
          <Label htmlFor="history-this-video" className="text-xs">
            This video only
          </Label>
        </div>
        <div className="flex items-center gap-1.5">
          <Switch id="history-mine" checked={filters.mine} onCheckedChange={(on) => onFilters({ ...filters, mine: on })} />
          <Label htmlFor="history-mine" className="text-xs">
            Mine
          </Label>
        </div>
      </div>

      {!entries.length && !library.length && <p className="text-xs text-muted-foreground">Nothing tried yet.</p>}

      <ul className="space-y-1">
        {entries.map((entry) => {
          const isShown = shown.includes(entry.id);
          return (
            <li
              key={entry.id}
              data-testid="history-entry"
              className={cn('rounded border p-2 text-xs', restoredId === entry.id && 'border-primary')}
            >
              <div className="flex items-start gap-1">
                <button
                  type="button"
                  className="min-w-0 flex-1 text-left"
                  onClick={() => props.onRestore(entry)}
                  title="Put this question, model and sampling back"
                >
                  {renaming === entry.id ? null : (
                    <span className="block truncate text-sm font-medium">{entryTitle(entry)}</span>
                  )}
                  <span className="block truncate text-muted-foreground">
                    {entry.kind === 'run' ? 'run' : entry.kind === 'draft' ? 'not tried' : 'try'} · {entry.model} ·{' '}
                    {samplingSummary(entry.sampling)}
                  </span>
                  <span className="block truncate text-muted-foreground">
                    {entry.video_name} · {entry.user_name ?? 'someone'} · {when(entry.created_at)}
                    {entry.status === 'running' && ' · going'}
                    {entry.status === 'stopped' && ' · stopped'}
                    {entry.status === 'failed' && ' · failed'}
                  </span>
                </button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  aria-label={entry.starred ? 'Unstar' : 'Star'}
                  onClick={() => props.onStar(entry, !entry.starred)}
                >
                  <Star className={cn('h-3 w-3', entry.starred && 'fill-amber-400 text-amber-500')} />
                </Button>
                <Button size="icon" variant="ghost" className="h-6 w-6" aria-label="Rename" onClick={() => setRenaming(entry.id)}>
                  <Pencil className="h-3 w-3" />
                </Button>
                {entry.kind !== 'draft' && (
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-6 w-6"
                    aria-label={isShown ? 'Hide its answers' : 'Show its answers'}
                    aria-pressed={isShown}
                    disabled={!isShown && !canShow(entry)}
                    onClick={() => props.onToggleShow(entry)}
                  >
                    {isShown ? <Eye className="h-3 w-3 text-primary" /> : <EyeOff className="h-3 w-3" />}
                  </Button>
                )}
                <DeleteEntry entry={entry} onDelete={props.onDelete} />
              </div>
              {renaming === entry.id && (
                <Rename
                  initial={entry.name ?? ''}
                  onDone={(name) => {
                    setRenaming(null);
                    if (name !== null) props.onRename(entry, name);
                  }}
                />
              )}
            </li>
          );
        })}
      </ul>

      {library.length > 0 && (
        <div className="space-y-1 pt-2">
          <p className="text-xs font-medium text-muted-foreground">Earlier questions</p>
          <ul className="space-y-1">
            {library.map((prompt) => (
              <li key={prompt.sha256} className="flex items-start gap-1 rounded border border-dashed p-2 text-xs" data-testid="library-prompt">
                <button type="button" className="min-w-0 flex-1 text-left" onClick={() => props.onRestoreLibrary(prompt)}>
                  {renaming === prompt.sha256 ? null : (
                    <span className="block truncate font-medium">{prompt.name || prompt.text.split('\n')[0]}</span>
                  )}
                  <span className="block truncate text-muted-foreground">
                    {prompt.models?.join(', ')} · {when(prompt.last_used_at)}
                  </span>
                </button>
                {renaming === prompt.sha256 ? (
                  <Rename
                    initial={prompt.name ?? ''}
                    onDone={(name) => {
                      setRenaming(null);
                      if (name !== null) props.onLibraryUpdate(prompt, { name });
                    }}
                  />
                ) : (
                  <>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-6 w-6"
                      aria-label={prompt.starred ? 'Unstar' : 'Star'}
                      onClick={() => props.onLibraryUpdate(prompt, { starred: !prompt.starred })}
                    >
                      <Star className={cn('h-3 w-3', prompt.starred && 'fill-amber-400 text-amber-500')} />
                    </Button>
                    <Button size="icon" variant="ghost" className="h-6 w-6" aria-label="Rename" onClick={() => setRenaming(prompt.sha256)}>
                      <Pencil className="h-3 w-3" />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-6 w-6"
                      aria-label="Hide"
                      onClick={() => props.onLibraryUpdate(prompt, { hidden: true })}
                    >
                      <EyeOff className="h-3 w-3" />
                    </Button>
                    <Button size="icon" variant="ghost" className="h-6 w-6" aria-label="Delete" onClick={() => props.onLibraryDelete(prompt)}>
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
