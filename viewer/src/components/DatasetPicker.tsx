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
  selectionMatch,
  type Candidate,
  type DatasetMatch,
} from '@/lib/datasetMatch';
import { rememberFolder, rememberedFolder, supportsFolderHandles, videosIn } from '@/lib/datasetHandles';
import { canEdit } from '@/lib/datasets';
import { parseApiError } from '@/lib/errorHandling';
import type { SavedDataset, ScannedVideo, StoredVideosResponse } from '@/types/datasets';

export interface PickedFile {
  file: File;
  relativePath: string | null;
}

interface DatasetPickerProps {
  onUseFiles: (files: PickedFile[], dataset: SavedDataset, folder: FileSystemDirectoryHandle | null) => void;
  onUseServerFolder: (selection: ServerFolderSelection, dataset: SavedDataset) => void;
  /** Run on the copies the server kept when these videos were uploaded: no folder, no upload. */
  onUseStored: (dataset: SavedDataset, stored: StoredVideosResponse) => void;
  /** A dataset to point out (opened from the Datasets page's "Start a job"). */
  highlightId?: string;
}

/** The videos of a selection dataset that are still where they were (spec 022). */
function selectionOf(path: string, match: DatasetMatch<ScannedVideo>): ServerFolderSelection {
  const files = match.matched.map(({ item }) => item.relative_path);
  // Reopened in My folders, videos in subfolders are only listed with them.
  const recursive = files.some((f) => f.includes('/'));
  return { path, recursive, videoCount: files.length, files };
}

type Pending =
  | { kind: 'files'; dataset: SavedDataset; candidates: Candidate<File>[]; match: DatasetMatch<File>; folder: FileSystemDirectoryHandle | null }
  | { kind: 'server'; dataset: SavedDataset; candidates: Candidate<ScannedVideo>[]; match: DatasetMatch<ScannedVideo> };

/** Null when the server can't say (one older than this endpoint): fall back to the folder. */
async function storedVideosOf(dataset: SavedDataset): Promise<StoredVideosResponse | null> {
  try {
    return await apiClient.getStoredVideos(dataset.id);
  } catch {
    return null;
  }
}

const picked = (candidates: Candidate<File>[]): PickedFile[] =>
  candidates.map((c) => ({ file: c.item, relativePath: c.relativePath }));

/**
 * "Use a saved dataset" in the job wizard (spec 018): finds the dataset's
 * videos again (the folder this browser remembers, the user re-picking it, or
 * the server folder) and shows any differences before anything runs.
 */
export const DatasetPicker = ({ onUseFiles, onUseServerFolder, onUseStored, highlightId }: DatasetPickerProps) => {
  const queryClient = useQueryClient();
  const { currentUser } = useCurrentUser();
  const { data, isLoading, error } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => apiClient.listDatasets(),
  });
  const [busy, setBusy] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [needsFolder, setNeedsFolder] = useState<SavedDataset | null>(null);
  // Some of needsFolder's videos are still on the server: offer to run those.
  const [partlyStored, setPartlyStored] = useState<StoredVideosResponse | null>(null);
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
        const selection = !!dataset.server_selection;
        // A selection's videos may be in subfolders, whatever was ticked.
        const scan = await apiClient.scanServerFolder(dataset.server_folder, selection || !!dataset.server_folder_recursive);
        const candidates = scanCandidates(scan.videos);
        const match = (selection ? selectionMatch : matchDataset)(dataset.video_manifest, candidates);
        if (hasDifferences(match)) {
          setPending({ kind: 'server', dataset, candidates, match });
        } else if (selection) {
          onUseServerFolder(selectionOf(scan.path, match), dataset);
        } else {
          onUseServerFolder(
            { path: scan.path, recursive: scan.recursive, videoCount: scan.videos.length },
            dataset,
          );
        }
        return;
      }
      // Uploaded videos stay in the folders of the jobs that ran on them, so
      // while those jobs exist the dataset runs from the server's copies.
      const stored = await storedVideosOf(dataset);
      if (stored && stored.stored > 0 && stored.missing === 0) {
        onUseStored(dataset, stored);
        return;
      }
      const folder = await rememberedFolder(dataset.id);
      if (!folder) {
        setPartlyStored(stored && stored.stored > 0 ? stored : null);
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
    if (pending.kind === 'server' && pending.dataset.server_selection) {
      onUseServerFolder(selectionOf(pending.dataset.server_folder!, pending.match), pending.dataset);
    } else if (pending.kind === 'server') {
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
              {partlyStored ? (
                <>
                  {partlyStored.stored} of the {partlyStored.videos.length} videos of “{needsFolder.name}” are still on
                  the server; {partlyStored.missing === 1 ? 'one was' : `${partlyStored.missing} were`} deleted with
                  {partlyStored.missing === 1 ? ' its job' : ' their jobs'}. Run the {partlyStored.stored} on the server,
                  or choose the folder they are all in.
                </>
              ) : (
                <>
                  The server no longer has the videos of “{needsFolder.name}” (their jobs were deleted). Choose the
                  folder they are in; they're read here in the browser.
                </>
              )}
            </span>
            <span className="flex gap-2 shrink-0">
              <Button size="sm" variant="ghost" onClick={() => setNeedsFolder(null)}>
                Cancel
              </Button>
              {partlyStored && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    onUseStored(needsFolder, partlyStored);
                    setNeedsFolder(null);
                  }}
                >
                  Run the {partlyStored.stored} on the server
                </Button>
              )}
              <Button size="sm" onClick={pickFolder}>
                Choose folder
              </Button>
            </span>
          </AlertDescription>
        </Alert>
      )}
      <ul className="divide-y rounded-md border">
        {datasets.map((dataset) => (
          <li
            key={dataset.id}
            className={`flex items-center justify-between gap-3 p-3 ${dataset.id === highlightId ? 'ring-2 ring-inset ring-primary' : ''}`}
          >
            <div className="min-w-0">
              <div className="flex items-center gap-2 font-medium">
                {dataset.server_folder ? <HardDrive className="h-4 w-4" /> : <Database className="h-4 w-4" />}
                <span className="truncate">{dataset.name}</span>
              </div>
              <div className="text-xs text-muted-foreground">
                {dataset.video_manifest.length} video{dataset.video_manifest.length === 1 ? '' : 's'}
                {dataset.server_folder
                  ? dataset.server_selection
                    ? ` · chosen videos in ${dataset.server_folder}`
                    : ` · server folder ${dataset.server_folder}`
                  : ' · uploaded from a browser'}
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
          serverFolder={pending.kind === 'server' && !pending.dataset.server_selection}
          canUpdate={canEdit(pending.dataset.owner_user_id, currentUser) && !pending.dataset.server_selection}
          onContinue={() => continuePending()}
          onUpdate={updateAndUse}
          onCancel={() => setPending(null)}
        />
      )}
    </div>
  );
};
