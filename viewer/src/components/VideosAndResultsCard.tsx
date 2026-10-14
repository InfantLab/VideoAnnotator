// Settings: which folders My folders may read, and where results go (spec 022),
// with Stop sharing when VideoAnnotator was started by its launcher (spec 024).

import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ResultsLocation } from '@/components/ResultsLocation';
import { useIngestAccess } from '@/hooks/useIngestAccess';
import { parseApiError } from '@/lib/errorHandling';
import { QueryKeys } from '@/types/api';
import type { Share } from '@/types/ingest';

function SharedFolders({ shares, launcher }: { shares: Share[]; launcher: boolean }) {
  const queryClient = useQueryClient();
  const [stopping, setStopping] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  // Answered locally at once, so the note shows before the access re-read.
  const [stopped, setStopped] = useState<Set<string>>(new Set());

  const stop = async (share: Share) => {
    setProblem(null);
    setStopping(share.display_path);
    try {
      await apiClient.stopSharing(share.display_path);
      setStopped((previous) => new Set(previous).add(share.display_path));
      await queryClient.invalidateQueries({ queryKey: QueryKeys.ingestAccess });
    } catch (e) {
      setProblem(parseApiError(e).message);
    } finally {
      setStopping(null);
    }
  };

  return (
    <div className="space-y-1">
      <p className="font-medium">Shared folders</p>
      <ul className="space-y-1">
        {shares.map((share) => {
          const pending = share.stop_requested || stopped.has(share.display_path);
          return (
            <li key={share.display_path} className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <span className="font-mono text-xs break-all">{share.display_path}</span>
              <span className="text-xs text-muted-foreground">
                read-only{!share.present && ', not found at the last start'}
              </span>
              {pending ? (
                <span className="text-xs text-amber-700 dark:text-amber-500">
                  Stops when VideoAnnotator next starts
                </span>
              ) : (
                launcher && (
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    className="h-6 px-2 text-[11px]"
                    disabled={stopping !== null}
                    onClick={() => stop(share)}
                  >
                    Stop sharing
                  </Button>
                )
              )}
            </li>
          );
        })}
      </ul>
      {problem && <p className="text-xs text-destructive">{problem}</p>}
      {launcher ? (
        <p className="text-xs text-muted-foreground">
          To share another folder, run <code>videoannotator-start share</code>.
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">
          To change shared folders, change the compose settings (see the installation guide).
        </p>
      )}
    </div>
  );
}

export function VideosAndResultsCard() {
  const { access, isLoading } = useIngestAccess();
  if (isLoading || !access) return null;
  const shares = access.shares ?? [];
  const launcher = access.managed_by_launcher ?? false;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Videos and results</CardTitle>
        <CardDescription>Where VideoAnnotator reads videos from, and writes results to</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {shares.length > 0 ? (
          <SharedFolders shares={shares} launcher={launcher} />
        ) : access.in_container ? (
          <div className="space-y-1">
            <p className="font-medium">Shared folders</p>
            <p className="text-muted-foreground">{access.reason}</p>
          </div>
        ) : (
          <div className="space-y-1">
            <p className="font-medium">My folders</p>
            {access.can_read_in_place ? (
              <ul className="font-mono text-xs space-y-0.5">
                {access.allowed_folders.map((folder) => (
                  <li key={folder.path} className="break-all">
                    {folder.display_path}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground">{access.reason}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Change these with <code>VIDEOANNOTATOR_INGEST_ROOTS</code> and restart the server.
            </p>
          </div>
        )}

        <ResultsLocation folder={access.results_root} label="Results folder" />
        <p className="text-xs text-muted-foreground">
          Every run gets its own folder here, named after the run and its date. To use another folder (an
          encrypted or backed-up drive, say),{' '}
          {launcher ? (
            <>
              start VideoAnnotator with <code>videoannotator-start --results</code> and the folder.
            </>
          ) : (
            <>
              set <code>VIDEOANNOTATOR_RESULTS_DIR</code> (Docker: <code>RESULTS_DIR</code>) and restart the
              server.
            </>
          )}{' '}
          Runs from before the change stay where they were written.
        </p>
      </CardContent>
    </Card>
  );
}
