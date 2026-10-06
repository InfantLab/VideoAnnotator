// "My folders": choose videos on this machine, used where they are (spec 022).
//
// A browser cannot tell the server where a file is: neither `<input
// webkitdirectory>` nor the File System Access API exposes a real path, by
// design, so an ordinary file dialog can only ever upload. Instead the server
// lists the folders it may read and this walks them, and the researcher ticks
// the videos to run. Nothing is uploaded or copied: the jobs read each video
// in place, and a whole folder starts in one request.

import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Checkbox } from '@/components/ui/checkbox';
import { ChevronRight, CornerLeftUp, Folder, HardDrive } from 'lucide-react';
import { parseApiError } from '@/lib/errorHandling';
import { useIngestAccess } from '@/hooks/useIngestAccess';
import type { FolderRef, IngestBrowseResponse, Place } from '@/types/ingest';
import { displayPathOf } from '@/lib/folders';
import type { ServerFolderScan } from '@/types/datasets';

export interface ServerFolderSelection {
  path: string;
  /** How many videos will run. */
  videoCount: number;
  recursive: boolean;
  /**
   * Only these videos, as paths relative to `path` (spec 022). Absent means
   * every video in the folder (and its subfolders when `recursive`).
   */
  files?: string[];
}

interface ServerFolderPickerProps {
  selection: ServerFolderSelection | null;
  onSelect: (selection: ServerFolderSelection | null) => void;
  /**
   * Choose a folder, not videos: reports the folder being shown (with
   * `videoCount` 0) as the researcher moves around. For "where are these
   * videos now?" (spec 022).
   */
  folderOnly?: boolean;
}

const DOCS_URL =
  'https://github.com/InfantLab/VideoAnnotator/blob/master/docs/installation/INSTALLATION.md#choosing-videos-on-your-own-computer';

function formatSize(bytes: number | null | undefined): string {
  if (!bytes) return '';
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function sameSelection(a: ServerFolderSelection | null, b: ServerFolderSelection | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    a.path === b.path &&
    a.recursive === b.recursive &&
    a.videoCount === b.videoCount &&
    (a.files ?? []).join('\n') === (b.files ?? []).join('\n')
  );
}

const LAST_FOLDER_KEY = 'videoannotator.myFolders.lastFolder';

function rememberedFolder(): string | null {
  try {
    return localStorage.getItem(LAST_FOLDER_KEY);
  } catch {
    return null;
  }
}

function rememberFolder(path: string) {
  try {
    localStorage.setItem(LAST_FOLDER_KEY, path);
  } catch {
    // Private browsing: just start from Places next time.
  }
}

interface Crumb {
  label: string;
  path: string | null;
}

/** Places › Home › Studies › BabyJokes, each step clickable. */
function crumbsFor(path: string, roots: FolderRef[], places: Place[]): Crumb[] {
  const root = [...roots]
    .sort((a, b) => b.path.length - a.path.length)
    .find((r) => path === r.path || path.startsWith(`${r.path}/`));
  if (!root) return [{ label: 'Places', path: null }, { label: path, path }];
  const rootLabel = places.find((p) => p.path === root.path)?.label ?? root.display_path;
  const crumbs: Crumb[] = [
    { label: 'Places', path: null },
    { label: rootLabel, path: root.path },
  ];
  let current = root.path;
  for (const part of path.slice(root.path.length).split('/').filter(Boolean)) {
    current = `${current}/${part}`;
    crumbs.push({ label: part, path: current });
  }
  return crumbs;
}

export function ServerFolderPicker({ selection, onSelect, folderOnly = false }: ServerFolderPickerProps) {
  const { access } = useIngestAccess();
  const allowed = useMemo(() => access?.allowed_folders ?? [], [access]);
  const places = useMemo<Place[]>(
    () => access?.places ?? allowed.map((f) => ({ ...f, label: f.display_path, has_videos: true })),
    [access, allowed]
  );
  // Where the researcher was last time; otherwise Places (null), not a listing
  // of their home folder, which is mostly clutter.
  const [path, setPath] = useState<string | null>(selection?.path ?? rememberedFolder());
  const [recursive, setRecursive] = useState(selection?.recursive ?? false);
  // 'all' keeps a whole-folder selection (e.g. from a saved dataset) whole as
  // the list loads or grows with subfolders.
  const [chosen, setChosen] = useState<Set<string> | 'all'>(() =>
    selection ? (selection.files ? new Set(selection.files) : 'all') : new Set()
  );

  const browse = useQuery<IngestBrowseResponse>({
    queryKey: ['ingest', 'browse', path],
    queryFn: () => apiClient.browseServerFolders(path as string),
    enabled: path !== null,
    retry: false,
    refetchOnWindowFocus: false,
  });
  const scan = useQuery<ServerFolderScan>({
    queryKey: ['ingest', 'scan', path, recursive],
    queryFn: () => apiClient.scanServerFolder(path as string, recursive),
    enabled: path !== null,
    retry: false,
    refetchOnWindowFocus: false,
  });
  const videos = useMemo(() => scan.data?.videos ?? [], [scan.data]);
  const selected = useMemo(
    () =>
      chosen === 'all'
        ? videos.map((v) => v.relative_path)
        : videos.filter((v) => chosen.has(v.relative_path)).map((v) => v.relative_path),
    [chosen, videos]
  );

  // Report the selection whenever it really changes. Through a ref, so the
  // parent re-rendering with a new callback doesn't count as a change.
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const selectionRef = useRef(selection);
  selectionRef.current = selection;
  useEffect(() => {
    if (!folderOnly || !browse.data?.path) return;
    const next = { path: browse.data.path, videoCount: 0, recursive };
    if (!sameSelection(next, selectionRef.current)) onSelectRef.current(next);
  }, [folderOnly, browse.data, recursive]);
  useEffect(() => {
    if (folderOnly || path === null || !scan.data) return;
    const next: ServerFolderSelection | null =
      selected.length === 0
        ? null
        : selected.length === videos.length
        ? { path: scan.data.path, videoCount: selected.length, recursive }
        : { path: scan.data.path, videoCount: selected.length, recursive, files: selected };
    if (!sameSelection(next, selectionRef.current)) onSelectRef.current(next);
  }, [folderOnly, path, recursive, scan.data, selected, videos.length]);

  const open = (next: string | null) => {
    setPath(next);
    setChosen(new Set());
    if (next !== null) rememberFolder(next);
  };

  const toggle = (relativePath: string) => {
    const next = new Set(chosen === 'all' ? videos.map((v) => v.relative_path) : chosen);
    if (next.has(relativePath)) next.delete(relativePath);
    else next.add(relativePath);
    setChosen(next);
  };

  const allSelected = videos.length > 0 && selected.length === videos.length;
  const error = browse.error ?? scan.error;

  if (error) {
    const parsed = parseApiError(error);
    return (
      <Alert>
        <AlertTitle>Can&apos;t open this folder</AlertTitle>
        <AlertDescription className="space-y-2">
          <p>{parsed.message}</p>
          {parsed.hint && <p className="text-xs">{parsed.hint}</p>}
          {path !== null && (
            <Button variant="outline" size="sm" onClick={() => open(null)}>
              Back to Places
            </Button>
          )}
        </AlertDescription>
      </Alert>
    );
  }

  const crumbs = path === null ? [] : crumbsFor(path, allowed, places);

  return (
    <div className="space-y-3">
      <nav aria-label="Folder" className="flex items-center gap-1 text-sm flex-wrap">
        <HardDrive className="h-4 w-4 text-muted-foreground shrink-0 mr-1" />
        {path === null ? (
          <span className="font-medium">Places</span>
        ) : (
          crumbs.map((crumb, i) => (
            <span key={`${crumb.path}-${i}`} className="flex items-center gap-1">
              {i > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground" />}
              {i === crumbs.length - 1 ? (
                <span className="font-medium" title={displayPathOf(path, allowed)}>
                  {crumb.label}
                </span>
              ) : (
                <button type="button" className="underline-offset-2 hover:underline" onClick={() => open(crumb.path)}>
                  {crumb.label}
                </button>
              )}
            </span>
          ))
        )}
      </nav>

      <Card>
        <CardContent className="p-0 max-h-80 overflow-y-auto">
          {path === null ? (
            <ul className="divide-y">
              {places.length > 0 && !places.some((p) => p.has_videos) && (
                <li className="px-4 py-3 text-sm text-muted-foreground">
                  No videos found in these folders. If yours are on another drive, add that folder to{' '}
                  <code>VIDEOANNOTATOR_INGEST_ROOTS</code> and restart; if they&apos;re on another computer,
                  upload them.
                </li>
              )}
              {places.map((place) => (
                <li key={place.path}>
                  <button
                    type="button"
                    onClick={() => open(place.path)}
                    className="w-full flex items-center gap-2 px-4 py-3 text-sm hover:bg-muted/50 text-left"
                  >
                    <Folder className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className={`flex-1 truncate ${place.has_videos ? '' : 'text-muted-foreground'}`}>
                      <span className="font-medium">{place.label}</span>{' '}
                      {/* Home and its usual folders need no path; a configured folder does. */}
                      {place.label !== 'Home' && allowed.some((f) => f.path === place.path) && (
                        <span className="font-mono text-xs text-muted-foreground">{place.display_path}</span>
                      )}
                    </span>
                    {place.has_videos && (
                      <Badge variant="outline" className="text-xs shrink-0">
                        has videos
                      </Badge>
                    )}
                    <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
                  </button>
                </li>
              ))}
            </ul>
          ) : browse.isLoading ? (
            <div className="p-4 space-y-2">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </div>
          ) : (
            <ul className="divide-y">
              {browse.data?.parent !== null && browse.data?.parent !== undefined && (
                <li>
                  <button
                    type="button"
                    onClick={() => open(browse.data.parent as string)}
                    className="w-full flex items-center gap-2 px-4 py-2 text-sm hover:bg-muted/50 text-left"
                  >
                    <CornerLeftUp className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="text-muted-foreground">Up one level</span>
                  </button>
                </li>
              )}

              {browse.data?.directories.map((dir) => (
                <li key={dir.path}>
                  <button
                    type="button"
                    onClick={() => open(dir.path)}
                    className="w-full flex items-center gap-2 px-4 py-2 text-sm hover:bg-muted/50 text-left"
                  >
                    <Folder className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span
                      className={`truncate flex-1 ${dir.has_videos === false ? 'text-muted-foreground' : ''}`}
                      title={dir.name}
                    >
                      {dir.name}
                    </span>
                    {dir.video_count > 0 ? (
                      <Badge variant="outline" className="text-xs shrink-0">
                        {dir.video_count} video{dir.video_count === 1 ? '' : 's'}
                      </Badge>
                    ) : (
                      dir.has_videos && (
                        <Badge variant="outline" className="text-xs shrink-0">
                          videos inside
                        </Badge>
                      )
                    )}
                    <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
                  </button>
                </li>
              ))}

              {/* Native checkboxes: a corpus folder can list thousands. */}
              {!folderOnly && videos.map((video) => (
                <li key={video.relative_path}>
                  <label className="flex items-center gap-2 px-4 py-2 text-sm hover:bg-muted/50 cursor-pointer">
                    <input
                      type="checkbox"
                      className="h-4 w-4 shrink-0 accent-primary"
                      checked={chosen === 'all' || chosen.has(video.relative_path)}
                      onChange={() => toggle(video.relative_path)}
                    />
                    <span className="truncate flex-1" title={video.relative_path}>
                      {video.relative_path}
                    </span>
                    <span className="text-xs text-muted-foreground shrink-0">
                      {formatSize(video.size_bytes)}
                    </span>
                  </label>
                </li>
              ))}

              {browse.data &&
                browse.data.directories.length === 0 &&
                !scan.isLoading &&
                videos.length === 0 && (
                  <li className="px-4 py-6 text-sm text-muted-foreground text-center">
                    No videos or folders here.
                  </li>
                )}
            </ul>
          )}
        </CardContent>
      </Card>

      {path !== null && (
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={recursive}
              onCheckedChange={(checked) => setRecursive(checked === true)}
            />
            Include subfolders
          </label>
          {!folderOnly && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={videos.length === 0}
              onClick={() => setChosen(allSelected ? new Set() : 'all')}
            >
              {allSelected ? 'Clear selection' : 'Select all'}
            </Button>
          )}
        </div>
      )}

      {!folderOnly && (
        <>
          <p className="text-sm" aria-live="polite">
            <span className="font-medium">
              {selected.length} video{selected.length === 1 ? '' : 's'} selected
            </span>
            <span className="text-muted-foreground">
              {' '}
              &ndash; used where they are, never copied. Deleting a job never deletes the original video.
            </span>
          </p>

          <p className="text-xs text-muted-foreground">
            Videos somewhere else, like an external drive? Add that folder to{' '}
            <code>VIDEOANNOTATOR_INGEST_ROOTS</code> (Docker: <code>VIDEOS_DIR</code>) and restart.{' '}
            <a className="underline" href={DOCS_URL} target="_blank" rel="noreferrer">
              How
            </a>
          </p>
        </>
      )}
    </div>
  );
}
