import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { DatasetMatch } from '@/lib/datasetMatch';
import type { ManifestEntry } from '@/types/datasets';

interface DatasetDriftDialogProps<T> {
  open: boolean;
  datasetName: string;
  match: DatasetMatch<T>;
  /** How to show an item that was found (a File, a scanned server video). */
  describe: (item: T) => string;
  /** A server folder runs as a whole, so "continue" means "as the folder is now". */
  serverFolder: boolean;
  canUpdate: boolean;
  onContinue: () => void;
  onUpdate: () => void;
  onCancel: () => void;
}

const entryName = (entry: ManifestEntry) => entry.relative_path || entry.filename;

const Section = ({ title, items }: { title: string; items: string[] }) =>
  items.length === 0 ? null : (
    <div>
      <h4 className="text-sm font-medium">
        {title} ({items.length})
      </h4>
      <ul className="mt-1 max-h-28 overflow-y-auto text-xs text-muted-foreground font-mono">
        {items.slice(0, 200).map((item) => (
          <li key={item}>{item}</li>
        ))}
        {items.length > 200 && <li>… and {items.length - 200} more</li>}
      </ul>
    </div>
  );

/**
 * What differs between a saved dataset and the videos found now (spec 018
 * FR-009): shown before anything runs, so nothing proceeds silently.
 */
export function DatasetDriftDialog<T>({
  open,
  datasetName,
  match,
  describe,
  serverFolder,
  canUpdate,
  onContinue,
  onUpdate,
  onCancel,
}: DatasetDriftDialogProps<T>) {
  const usable = match.matched.length + (serverFolder ? match.changed.length + match.added.length : 0);
  return (
    <Dialog open={open} onOpenChange={(next) => !next && onCancel()}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>The videos have changed since “{datasetName}” was saved</DialogTitle>
          <DialogDescription>
            {match.matched.length} of {match.matched.length + match.missing.length + match.changed.length + match.ambiguous.length}{' '}
            saved videos were found as they were.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <Section title="Missing" items={match.missing.map(entryName)} />
          <Section title="Different size" items={match.changed.map(({ entry }) => entryName(entry))} />
          <Section title="Can't tell which file (same name, no folder path saved)" items={match.ambiguous.map(entryName)} />
          <Section title="New, not in the dataset" items={match.added.map(describe)} />
        </div>
        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
          {canUpdate && (
            <Button variant="outline" onClick={onUpdate}>
              Update the dataset to what's here
            </Button>
          )}
          <Button onClick={onContinue} disabled={usable === 0}>
            {serverFolder
              ? 'Use the folder as it is now'
              : `Continue with the ${match.matched.length} matching video${match.matched.length === 1 ? '' : 's'}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
