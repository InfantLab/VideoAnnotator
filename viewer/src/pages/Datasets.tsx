import { useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { ChevronDown, ChevronRight, Database, Download, HardDrive, Pencil, Play, Trash2, Upload } from 'lucide-react';
import { apiClient } from '@/api/client';
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
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { useCurrentUser } from '@/hooks/useCurrentUser';
import { forgetFolder } from '@/lib/datasetHandles';
import { canEdit, createWithFreeName, downloadJSON, exportFileName, parseExport } from '@/lib/datasets';
import { parseApiError } from '@/lib/errorHandling';
import type { DatasetCreateRequest, ManifestEntry, SavedDataset } from '@/types/datasets';

const when = (iso?: string | null) => (iso ? new Date(iso).toLocaleString() : 'never');
const entryName = (e: ManifestEntry) => e.relative_path || e.filename;
const mb = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

/** Start a job from a dataset: the wizard opens on its Saved dataset tab. */
export interface StartFromDatasetState {
  startFromDataset: { id: string; name: string };
}

const DatasetRow = ({ dataset, editable }: { dataset: SavedDataset; editable: boolean }) => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(dataset.name);
  const [description, setDescription] = useState(dataset.description ?? '');
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const refresh = () => queryClient.invalidateQueries({ queryKey: ['datasets'] });
  const update = useMutation({
    mutationFn: (body: Parameters<typeof apiClient.updateDataset>[1]) => apiClient.updateDataset(dataset.id, body),
    onSuccess: () => {
      setProblem(null);
      setEditing(false);
      refresh();
    },
    onError: (e) => setProblem(parseApiError(e).message),
  });
  const remove = useMutation({
    mutationFn: () => apiClient.deleteDataset(dataset.id),
    onSuccess: async () => {
      await forgetFolder(dataset.id);
      refresh();
    },
    onError: (e) => setProblem(parseApiError(e).message),
  });

  const removeVideo = (entry: ManifestEntry) =>
    update.mutate({ video_manifest: dataset.video_manifest.filter((e) => e !== entry) });

  const exportIt = () => {
    const { id: _id, owner_user_id: _owner, owner_name: _ownerName, ...definition } = dataset;
    downloadJSON(exportFileName('dataset', dataset.name), definition);
  };

  const total = dataset.video_manifest.reduce((sum, e) => sum + (e.size_bytes || 0), 0);

  return (
    <li className="p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <button type="button" className="flex items-start gap-2 text-left min-w-0" onClick={() => setOpen(!open)}>
          {open ? <ChevronDown className="h-4 w-4 mt-1" /> : <ChevronRight className="h-4 w-4 mt-1" />}
          <div className="min-w-0">
            <div className="flex items-center gap-2 font-medium">
              {dataset.server_folder ? <HardDrive className="h-4 w-4" /> : <Database className="h-4 w-4" />}
              <span className="truncate">{dataset.name}</span>
            </div>
            {dataset.description && <p className="text-sm text-muted-foreground">{dataset.description}</p>}
            <p className="text-xs text-muted-foreground">
              {dataset.video_manifest.length} video{dataset.video_manifest.length === 1 ? '' : 's'} · {mb(total)}
              {dataset.server_folder
                ? ` · server folder ${dataset.server_folder}${dataset.server_folder_recursive ? ' (with subfolders)' : ''}`
                : ' · uploaded from a browser'}
              {' · saved by '}
              {dataset.owner_name ?? 'unknown'} · created {when(dataset.created_at)} · last used {when(dataset.last_used_at)}
            </p>
          </div>
        </button>
        <div className="flex gap-1 shrink-0">
          <Button
            size="sm"
            onClick={() =>
              navigate('/jobs/new', {
                state: { startFromDataset: { id: dataset.id, name: dataset.name } } satisfies StartFromDatasetState,
              })
            }
          >
            <Play className="h-4 w-4 mr-1" /> Start a job
          </Button>
          <Button size="sm" variant="ghost" onClick={exportIt} aria-label={`Export ${dataset.name}`}>
            <Download className="h-4 w-4" />
          </Button>
          {editable && (
            <>
              <Button size="sm" variant="ghost" onClick={() => setEditing(!editing)} aria-label={`Edit ${dataset.name}`}>
                <Pencil className="h-4 w-4" />
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setConfirmDelete(true)} aria-label={`Delete ${dataset.name}`}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </>
          )}
        </div>
      </div>

      {problem && (
        <Alert variant="destructive">
          <AlertDescription>{problem}</AlertDescription>
        </Alert>
      )}

      {editing && (
        <div className="space-y-2 pl-6">
          <Input value={name} onChange={(e) => setName(e.target.value)} aria-label="Dataset name" />
          <Textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={2} aria-label="Description" />
          <div className="flex gap-2">
            <Button
              size="sm"
              disabled={!name.trim() || update.isPending}
              onClick={() => update.mutate({ name: name.trim(), description: description.trim() })}
            >
              Save
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>
              Cancel
            </Button>
          </div>
        </div>
      )}

      {open && (
        <ul className="pl-6 max-h-64 overflow-y-auto text-sm divide-y">
          {dataset.video_manifest.map((entry, i) => (
            <li key={`${entryName(entry)}-${i}`} className="flex items-center justify-between py-1">
              <span className="font-mono text-xs truncate">{entryName(entry)}</span>
              <span className="flex items-center gap-2 shrink-0 text-xs text-muted-foreground">
                {mb(entry.size_bytes)}
                {editable && !dataset.server_folder && (
                  <Button size="sm" variant="ghost" className="h-6 px-2" onClick={() => removeVideo(entry)}>
                    Remove
                  </Button>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}

      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{dataset.name}”?</AlertDialogTitle>
            <AlertDialogDescription>
              The dataset is removed for everyone on this server. The videos themselves are not touched, and jobs
              already run from it keep their results.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => remove.mutate()}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  );
};

/**
 * Saved datasets (spec 018): named lists of input videos, shared on the
 * server, to run jobs on again. Results downloaded from jobs live in the Library.
 */
const Datasets = () => {
  const queryClient = useQueryClient();
  const { currentUser } = useCurrentUser();
  const importInput = useRef<HTMLInputElement>(null);
  const [notice, setNotice] = useState<{ ok: boolean; text: string } | null>(null);
  const { data, isLoading, error } = useQuery({ queryKey: ['datasets'], queryFn: () => apiClient.listDatasets() });

  const importFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    try {
      const raw = parseExport(await file.text(), ['name', 'video_manifest']);
      const body: DatasetCreateRequest = {
        name: String(raw.name),
        description: (raw.description as string | null | undefined) ?? null,
        video_manifest: raw.video_manifest as ManifestEntry[],
        server_folder: (raw.server_folder as string | null | undefined) ?? null,
        server_folder_recursive: Boolean(raw.server_folder_recursive),
      };
      const { name } = await createWithFreeName(body, (b) => apiClient.createDataset(b));
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      setNotice({ ok: true, text: name === body.name ? `Imported “${name}”.` : `Imported as “${name}” (you already had “${body.name}”).` });
    } catch (e) {
      setNotice({ ok: false, text: `Couldn't import ${file.name}: ${parseApiError(e).message}` });
    }
  };

  const datasets = [...(data?.datasets ?? [])].sort((a, b) =>
    (b.last_used_at ?? b.created_at).localeCompare(a.last_used_at ?? a.created_at),
  );

  return (
    <div className="container mx-auto p-6 max-w-5xl space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Datasets</h1>
          <p className="text-muted-foreground mt-2">
            Named lists of videos to run jobs on, saved from the job wizard and shared with everyone on this server. A
            dataset remembers which videos (names, sizes, folders), not the videos themselves. Results you've
            downloaded are in the Library.
          </p>
        </div>
        <div className="shrink-0">
          <input ref={importInput} type="file" accept=".json,application/json" className="hidden" onChange={importFile} />
          <Button variant="outline" onClick={() => importInput.current?.click()}>
            <Upload className="h-4 w-4 mr-2" /> Import
          </Button>
        </div>
      </div>

      {notice && (
        <Alert variant={notice.ok ? 'default' : 'destructive'}>
          <AlertDescription>{notice.text}</AlertDescription>
        </Alert>
      )}

      <Card>
        {isLoading ? (
          <p className="p-6 text-sm text-muted-foreground">Loading…</p>
        ) : error ? (
          <p className="p-6 text-sm text-destructive">Couldn't load datasets: {parseApiError(error).message}</p>
        ) : datasets.length === 0 ? (
          <div className="p-8 text-center space-y-2">
            <Database className="h-10 w-10 mx-auto text-muted-foreground" />
            <p className="font-medium">No saved datasets yet</p>
            <p className="text-sm text-muted-foreground">
              In a new job, choose your videos (a folder on this computer, or one on the server), then use “Save as
              dataset”. Next time, pick it on the wizard's “Saved dataset” tab.
            </p>
          </div>
        ) : (
          <ul className="divide-y">
            {datasets.map((dataset) => (
              <DatasetRow key={dataset.id} dataset={dataset} editable={canEdit(dataset.owner_user_id, currentUser)} />
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
};

export default Datasets;
