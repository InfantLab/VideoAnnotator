import { useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { formatClockTenths } from '@/lib/lanes';

export interface EditingNote {
  /** Absent for a new note. */
  id?: string;
  start: number;
  end: number | null;
  text: string;
}

interface NoteEditorProps {
  note: EditingNote;
  onSave: (text: string) => void;
  onCancel: () => void;
  onDelete?: () => void;
}

/**
 * Writing or changing a note (spec 028 US6.1): focused for typing; Enter saves,
 * Shift+Enter starts a new line, Esc cancels. An empty note is never saved.
 */
export function NoteEditor({ note, onSave, onCancel, onDelete }: NoteEditorProps) {
  const [text, setText] = useState(note.text);
  const box = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    box.current?.focus();
  }, []);
  const when = note.end !== null ? `${formatClockTenths(note.start)}–${formatClockTenths(note.end)}` : formatClockTenths(note.start);
  const save = () => (text.trim() ? onSave(text) : onCancel());
  return (
    <div
      className="absolute inset-x-4 bottom-4 z-20 rounded-md border-2 border-dashed border-amber-500 bg-card/95 p-2 text-xs shadow-lg"
      role="dialog"
      aria-label={note.id ? 'Change the note' : 'New note'}
      data-testid="note-editor"
    >
      <p className="mb-1 font-medium">
        {note.id ? 'Note' : 'New note'} at {when}
      </p>
      <textarea
        ref={box}
        className="h-16 w-full resize-none rounded border bg-background p-1 text-sm"
        value={text}
        maxLength={2000}
        aria-label="Note text"
        placeholder="What happens here?"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            save();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            onCancel();
          }
        }}
      />
      <div className="mt-1 flex items-center gap-2">
        <Button size="sm" className="h-6 px-2 text-xs" onClick={save} disabled={!text.trim()}>
          Save
        </Button>
        <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={onCancel}>
          Cancel
        </Button>
        {onDelete && (
          <Button size="sm" variant="ghost" className="ml-auto h-6 px-2 text-xs text-destructive" onClick={onDelete}>
            Delete
          </Button>
        )}
        <span className="ml-auto text-muted-foreground">Enter saves · Esc cancels</span>
      </div>
    </div>
  );
}
