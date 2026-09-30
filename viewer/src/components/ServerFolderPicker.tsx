// Pick a folder that already exists on the server's machine.
//
// A browser cannot tell the server where a folder is: neither `<input
// webkitdirectory>` nor the File System Access API exposes a real path, by
// design. So rather than asking a researcher to type an absolute path, the
// server lists its own readable folders and this walks them.
//
// The payoff is that a 40-video corpus becomes one request instead of 40
// uploads — nothing is copied, the jobs just point at the videos where they
// already sit.

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Checkbox } from '@/components/ui/checkbox';
import { ChevronRight, CornerLeftUp, Film, Folder, HardDrive } from 'lucide-react';
import { parseApiError } from '@/lib/errorHandling';
import type { IngestBrowseResponse } from '@/types/ingest';

export interface ServerFolderSelection {
  path: string;
  /** Videos directly in the folder, as reported by the server. */
  videoCount: number;
  recursive: boolean;
}

interface ServerFolderPickerProps {
  selection: ServerFolderSelection | null;
  onSelect: (selection: ServerFolderSelection | null) => void;
}

function formatSize(bytes: number | null): string {
  if (!bytes) return '';
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function ServerFolderPicker({ selection, onSelect }: ServerFolderPickerProps) {
  // `null` means "show the roots" — the server decides what those are.
  const [path, setPath] = useState<string | null>(selection?.path ?? null);
  const [recursive, setRecursive] = useState(selection?.recursive ?? false);

  const { data, isLoading, error } = useQuery<IngestBrowseResponse>({
    queryKey: ['ingest', 'browse', path],
    queryFn: () => apiClient.browseServerFolders(path ?? undefined),
    retry: false,
    refetchOnWindowFocus: false,
  });

  const choose = (nextPath: string, videoCount: number) => {
    setPath(nextPath);
    // Selecting and browsing are the same gesture: stepping into a folder that
    // has videos selects it, so there's no separate "use this one" click for
    // the common case of a flat corpus folder.
    onSelect(videoCount > 0 ? { path: nextPath, videoCount, recursive } : null);
  };

  const setRecursiveAndKeep = (next: boolean) => {
    setRecursive(next);
    if (selection) onSelect({ ...selection, recursive: next });
  };

  if (error) {
    const parsed = parseApiError(error);
    return (
      <Alert>
        <AlertTitle>Can&apos;t browse the server&apos;s folders</AlertTitle>
        <AlertDescription className="space-y-2">
          <p>{parsed.message}</p>
          {parsed.hint && <p className="text-xs">{parsed.hint}</p>}
          <p className="text-xs">
            Uploading the files instead works from anywhere — switch back to
            &ldquo;Upload from this computer&rdquo; above.
          </p>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-3">
      {/* Where we are */}
      <div className="flex items-center gap-2 text-sm">
        <HardDrive className="h-4 w-4 text-muted-foreground shrink-0" />
        <span className="font-mono text-xs truncate" title={data?.path ?? 'Server folders'}>
          {data?.path ?? 'Server folders'}
        </span>
      </div>

      <Card>
        <CardContent className="p-0 max-h-72 overflow-y-auto">
          {isLoading ? (
            <div className="p-4 space-y-2">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </div>
          ) : (
            <ul className="divide-y">
              {data?.parent !== null && data?.parent !== undefined && (
                <li>
                  <button
                    type="button"
                    onClick={() => setPath(data.parent)}
                    className="w-full flex items-center gap-2 px-4 py-2 text-sm hover:bg-muted/50 text-left"
                  >
                    <CornerLeftUp className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="text-muted-foreground">Up one level</span>
                  </button>
                </li>
              )}

              {data?.directories.map((dir) => (
                <li key={dir.path}>
                  <button
                    type="button"
                    onClick={() => choose(dir.path, dir.video_count)}
                    className="w-full flex items-center gap-2 px-4 py-2 text-sm hover:bg-muted/50 text-left"
                  >
                    <Folder className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="truncate flex-1" title={dir.name}>
                      {dir.name}
                    </span>
                    {dir.video_count > 0 && (
                      <Badge variant="outline" className="text-xs shrink-0">
                        {dir.video_count} video{dir.video_count === 1 ? '' : 's'}
                      </Badge>
                    )}
                    <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
                  </button>
                </li>
              ))}

              {data && data.directories.length === 0 && data.videos.length === 0 && (
                <li className="px-4 py-6 text-sm text-muted-foreground text-center">
                  Nothing in this folder.
                </li>
              )}

              {/* Videos are shown for confidence, not for individual selection:
                  ingest takes the folder. */}
              {data?.videos.slice(0, 50).map((video) => (
                <li
                  key={video.path}
                  className="flex items-center gap-2 px-4 py-2 text-sm text-muted-foreground"
                >
                  <Film className="h-4 w-4 shrink-0" />
                  <span className="truncate flex-1" title={video.name}>
                    {video.name}
                  </span>
                  <span className="text-xs shrink-0">{formatSize(video.size_bytes)}</span>
                </li>
              ))}
              {data && data.videos.length > 50 && (
                <li className="px-4 py-2 text-xs text-muted-foreground">
                  …and {data.videos.length - 50} more
                </li>
              )}
            </ul>
          )}
        </CardContent>
      </Card>

      {data?.truncated && (
        <p className="text-xs text-muted-foreground">
          This folder has too many entries to list in full.
        </p>
      )}

      {/* Confirm the current folder even when it has no videos of its own,
          so a parent folder can still be chosen with "include subfolders". */}
      {data?.path && (
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={recursive}
              onCheckedChange={(checked) => setRecursiveAndKeep(checked === true)}
            />
            Include videos in subfolders
          </label>

          <Button
            type="button"
            variant={selection?.path === data.path ? 'default' : 'outline'}
            size="sm"
            onClick={() =>
              onSelect({ path: data.path as string, videoCount: data.video_count, recursive })
            }
            disabled={data.video_count === 0 && !recursive}
          >
            {selection?.path === data.path ? 'Selected' : 'Use this folder'}
          </Button>
        </div>
      )}

      {selection && (
        <Alert>
          <Folder className="h-4 w-4" />
          <AlertTitle className="font-mono text-xs break-all">{selection.path}</AlertTitle>
          <AlertDescription className="text-sm">
            {selection.recursive
              ? 'Every video in this folder and its subfolders will be processed.'
              : `${selection.videoCount} video${selection.videoCount === 1 ? '' : 's'} will be processed.`}{' '}
            Nothing is uploaded — the server reads them where they are.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
