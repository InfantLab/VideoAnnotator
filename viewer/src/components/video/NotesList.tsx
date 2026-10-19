import { Pencil, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { formatClockTenths } from '@/lib/lanes';
import type { VideoNote } from '@/types/video';

interface NotesListProps {
  notes: VideoNote[];
  onSeek: (t: number) => void;
  onEdit: (note: VideoNote) => void;
  onDelete: (note: VideoNote) => void;
}

/** The Notes tab (spec 028 FR-017): every note on this video, by time. */
export function NotesList({ notes, onSeek, onEdit, onDelete }: NotesListProps) {
  if (!notes.length) {
    return (
      <p className="text-muted-foreground">
        No notes yet. Press <kbd className="rounded border px-1">N</kbd> for a note at this moment, or{' '}
        <kbd className="rounded border px-1">Shift</kbd>+<kbd className="rounded border px-1">N</kbd> twice for a span.
      </p>
    );
  }
  return (
    <ul className="space-y-1" data-testid="notes-list">
      {notes.map((note) => (
        <li key={note.id} className="group flex items-start gap-2 rounded border border-dashed border-amber-500/60 p-1.5">
          <button
            type="button"
            className="min-w-0 flex-1 text-left"
            onClick={() => onSeek(note.start_sec)}
            title="Go to this note"
          >
            <span className="font-mono text-muted-foreground">
              {formatClockTenths(note.start_sec)}
              {note.end_sec !== null && note.end_sec !== undefined ? `–${formatClockTenths(note.end_sec)}` : ''}
            </span>{' '}
            <span className="whitespace-pre-wrap break-words">{note.text}</span>
            <span className="block text-muted-foreground">{note.author_name}</span>
          </button>
          <Button variant="ghost" size="icon" className="h-6 w-6" aria-label="Change this note" onClick={() => onEdit(note)}>
            <Pencil className="h-3 w-3" />
          </Button>
          <Button variant="ghost" size="icon" className="h-6 w-6" aria-label="Delete this note" onClick={() => onDelete(note)}>
            <Trash2 className="h-3 w-3" />
          </Button>
        </li>
      ))}
    </ul>
  );
}
