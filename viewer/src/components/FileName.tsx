// A file name truncated in the middle (viewer design "Names", UAT #7): both ends
// stay visible, so "2UWdXP.joke1.rep2…Peekaboo_h265.mp4" and "…rep3…" differ.
import { cn } from '@/lib/utils';

/** Characters always shown at the end: enough for "Peekaboo_h265.mp4". */
const TAIL = 18;

export function FileName({ name, className, title }: { name: string; className?: string; title?: string }) {
  const tail = name.length > TAIL * 2 ? name.slice(-TAIL) : '';
  const head = tail ? name.slice(0, -TAIL) : name;
  return (
    <span className={cn('flex min-w-0', className)} title={title ?? name} data-testid="file-name">
      <span className="truncate">{head}</span>
      {tail && <span className="shrink-0 whitespace-pre">{tail}</span>}
    </span>
  );
}
