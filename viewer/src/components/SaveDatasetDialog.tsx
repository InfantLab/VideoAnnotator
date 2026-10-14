import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { parseApiError } from '@/lib/errorHandling';
import type { DatasetCreateRequest, SavedDataset } from '@/types/datasets';

interface SaveDatasetDialogProps {
  open: boolean;
  defaultName: string;
  /** What to save, built when the user confirms (a server folder is scanned then). */
  build: () => Promise<Omit<DatasetCreateRequest, 'name' | 'description'>>;
  onSaved: (dataset: SavedDataset) => void;
  onClose: () => void;
}

/** "Save as dataset" from the videos chosen in the wizard (spec 018). */
export const SaveDatasetDialog = ({ open, defaultName, build, onSaved, onClose }: SaveDatasetDialogProps) => {
  const queryClient = useQueryClient();
  const [name, setName] = useState(defaultName);
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const save = async () => {
    setSaving(true);
    setProblem(null);
    try {
      const dataset = await apiClient.createDataset({
        ...(await build()),
        name: name.trim(),
        description: description.trim() || null,
      });
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      onSaved(dataset);
    } catch (e) {
      setProblem(
        e instanceof APIError && e.status === 409
          ? `You already have a dataset named “${name.trim()}”. Choose another name.`
          : parseApiError(e).message,
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Save these videos as a dataset</DialogTitle>
          <DialogDescription>
            A dataset remembers which videos these are (names, sizes, folders), not the videos themselves, so you can
            choose them again in one step. Everyone on this server can see and use it.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="dataset-name">Name</Label>
            <Input id="dataset-name" value={name} onChange={(e) => setName(e.target.value)} autoFocus />
          </div>
          <div className="space-y-1">
            <Label htmlFor="dataset-description">Description (optional)</Label>
            <Textarea id="dataset-description" value={description} onChange={(e) => setDescription(e.target.value)} rows={2} />
          </div>
          {problem && <p className="text-sm text-destructive">{problem}</p>}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={save} disabled={saving || name.trim() === ''}>
            {saving ? 'Saving…' : 'Save dataset'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
