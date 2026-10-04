import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Database, FolderOpen, HardDrive, Loader2 } from 'lucide-react';
import { apiClient } from '@/api/client';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { DatasetDriftDialog } from '@/components/DatasetDriftDialog';
import type { ServerFolderSelection } from '@/components/ServerFolderPicker';
import { useCurrentUser } from '@/hooks/useCurrentUser';
import {
  fileCandidates,
  hasDifferences,
  manifestFrom,
  matchDataset,
  scanCandidates,
  type Candidate,
  type DatasetMatch,
} from '@/lib/datasetMatch';
import { rememberFolder, rememberedFolder, supportsFolderHandles, videosIn } from '@/lib/datasetHandles';
import { canEdit } from '@/lib/datasets';
import { parseApiError } from '@/lib/errorHandling';
import type { SavedDataset, ScannedVideo } from '@/types/datasets';

export interface PickedFile {
  file: File;
  relativePath: string | null;
}

interface DatasetPickerProps {
  onUseFiles: (files: PickedFile[], dataset: SavedDataset, folder: FileSystemDirectoryHandle | null) => void;
  onUseServerFolder: (selection: ServerFolderSelection, dataset: SavedDataset) => void;
}

type Pending =
  | { kind: 'files'; dataset: SavedDataset; candidates: Candidate<File>[]; match: DatasetMatch<File>; folder: FileSystemDirectoryHandle | null }
  | { kind: 'server'; dataset: SavedDataset; candidates: Candidate<ScannedVideo>[]; match: DatasetMatch<ScannedVideo> };

const picked = (candidates: Candidate<File>[]): PickedFile[] =>
  candidates.map((c) => ({ file: c.item, relativePath: c.relativePath }));

/**
 * "Use a saved dataset" in the job wizard (spec 018): finds the dataset's
 * videos again (the folder this browser remembers, the user re-picking it, or
 * the server folder) and shows any differences before anything runs.
 */
export const DatasetPicker = ({ onUseFiles, onUseServerFolder }: DatasetPickerProps) => {
  const queryClient = useQueryClient();
  const { currentUser } = useCurrentUser();
  const { data, isLoading, error } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => apiClient.listDatasets(),
  });
  const [busy, setBusy] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [needsFolder, setNeedsFolder] = useState<SavedDataset | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  const finishFiles = (dataset: SavedDataset, candidates: Candidate<File>[], folder: FileSystemDirectoryHandle | null) => {
    const match = matchDataset(dataset.video_manifest, candidates);
    if (hasDifferences(match)) {
      setPending({ kind: 'files', dataset, candidates, match, folder });
    } else {
      onUseFiles(picked(candidates), dataset, folder);
    }
  };

  const chooseDataset = async (dataset: SavedDataset) => {
    setProblem(null);
    setBusy(dataset.id);
    try {
      if (dataset.server_folder) {
        const scan = await apiClient.scanServerFolder(dataset.server_folder, !!dataset.server_folder_recursive);
        const candidates = scanCandidates(scan.videos);
        const match = matchDataset(dataset.video_manifest, candidates);
        if (hasDifferences(match)) {
          setPending({ kind: 'server', dataset, candidates, match });
        } else {
          onUseServerFolder(
            { path: scan.path, recursive: scan.recursive, videoCount: scan.videos.length },
            dataset,
          );
        }
        return;
      }
      const folder = await rememberedFolder(dataset.id);
      if (!folder) {
        setNeedsFolder(dataset);
        return;
      }
      finishFiles(dataset, await videosIn(folder), folder);
    } catch (e) {
      setProblem(parseApiError(e).message);
    } finally {
      setBusy(null);
    }
  };

  const pickFolder = async () => {
    const dataset = needsFolder;
    if (!dataset) return;
    if (!supportsFolderHandles()) {
      folderInput.current?.click();
      return;
    }
    try {
      const folder = await (window as unknown as { showDirectoryPicker: (o: object) => Promise<FileSystemDirectoryHandle> })
        .showDirectoryPicker({ mode: 'read' });
      await rememberFolder(dataset.id, folder);
      setNeedsFolder(null);
      finishFiles(dataset, await videosIn(folder), folder);
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return; // picker cancelled
      setProblem(parseApiError(e).message);
    }
  };

  const onFolderInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const dataset = needsFolder;
    const files = Array.from(event.target.files ?? []).filter((f) => f.type.startsWith('video/') || /\.(mp4|webm|avi|mov|mkv|m4v)$/i.test(f.name));
    event.target.value = '';
    if (!dataset || files.length === 0) return;
    setNeedsFolder(null);
    finishFiles(dataset, fileCandidates(files), null);
  };

  const updateAndUse = async () => {
    if (!pending) return;
    setBusy(pending.dataset.id);
    try {
      await apiClient.updateDataset(pending.dataset.id, { video_manifest: manifestFrom<unknown>(pending.candidates) });
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      continuePending(true);
    } catch (e) {
      setProblem(parseApiError(e).message);
      setPending(null);
    } finally {
      setBusy(null);
    }
  };

  const continuePending = (useEverything = false) => {
    if (!pending) return;
    if (pending.kind === 'server') {
      onUseServerFolder(
        { path: pending.dataset.server_folder!, recursive: !!pending.dataset.server_folder_recursive, videoCount: pending.candidates.length },
        pending.dataset,
      );
    } else {
      const files = useEverything
        ? picked(pending.candidates)
        : pending.match.matched.map(({ item }) => ({
            file: item,
            relativePath: pending.candidates.find((c) => c.item === item)?.relativePath ?? null,
          }));
      onUseFiles(files, pending.dataset, pending.folder);
    }
    setPending(null);
  };

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading saved datasets…</p>;
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>Couldn't load saved datasets: {parseApiError(error).message}</AlertDescription>
      </Alert>
    );
  }
  const datasets = data?.datasets ?? [];
  if (datasets.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No saved datasets yet. Choose videos on another tab and use “Save as dataset” to reuse them next time.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <input ref={folderInput} type="file" className="hidden" onChange={onFolderInput} {...{ webkitdirectory: '', directory: '' }} />
      {problem && (
        <Alert variant="destructive">
          <AlertDescription>{problem}</AlertDescription>
        </Alert>
      )}
      {needsFolder && (
        <Alert>
          <FolderOpen className="h-4 w-4" />
          <AlertDescription className="flex items-center justify-between gap-2">
            <span>
              Choose the folder the videos of “{needsFolder.name}” are in. They're read here in the browser; the dataset
              only remembers their names and sizes.
            </span>
            <span className="flex gap-2 shrink-0">
              <Button size="sm" variant="ghost" onClick={() => setNeedsFolder(null)}>
                Cancel
              </Button>
              <Button size="sm" onClick={pickFolder}>
                Choose folder
              </Button>
            </span>
          </AlertDescription>
        </Alert>
      )}
      <ul className="divide-y rounded-md border">
        {datasets.map((dataset) => (
          <li key={dataset.id} className="flex items-center justify-between gap-3 p-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2 font-medium">
                {dataset.server_folder ? <HardDrive className="h-4 w-4" /> : <Database className="h-4 w-4" />}
                <span className="truncate">{dataset.name}</span>
              </div>
              <div className="text-xs text-muted-foreground">
                {dataset.video_manifest.length} video{dataset.video_manifest.length === 1 ? '' : 's'}
                {dataset.server_folder ? ` · server folder ${dataset.server_folder}` : ' · uploaded from a browser'}
                {dataset.owner_name ? ` · saved by ${dataset.owner_name}` : ''}
              </div>
            </div>
            <Button size="sm" onClick={() => chooseDataset(dataset)} disabled={busy !== null}>
              {busy === dataset.id ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Use'}
            </Button>
          </li>
        ))}
      </ul>
      {pending && (
        <DatasetDriftDialog
          open
          datasetName={pending.dataset.name}
          match={pending.match as DatasetMatch<unknown>}
          describe={(item) =>
            item instanceof File
              ? (pending.candidates as Candidate<unknown>[]).find((c) => c.item === item)?.relativePath ?? item.name
              : (item as ScannedVideo).relative_path
          }
          serverFolder={pending.kind === 'server'}
          canUpdate={canEdit(pending.dataset.owner_user_id, currentUser)}
          onContinue={() => continuePending()}
          onUpdate={updateAndUse}
          onCancel={() => setPending(null)}
        />
      )}
    </div>
  );
};
