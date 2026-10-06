// Settings: which folders My folders may read, and where results go (spec 022).

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ResultsLocation } from '@/components/ResultsLocation';
import { useIngestAccess } from '@/hooks/useIngestAccess';

export function VideosAndResultsCard() {
  const { access, isLoading } = useIngestAccess();
  if (isLoading || !access) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Videos and results</CardTitle>
        <CardDescription>Where VideoAnnotator reads videos from, and writes results to</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
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
            Change these with <code>VIDEOANNOTATOR_INGEST_ROOTS</code> (Docker: <code>VIDEOS_DIR</code>) and
            restart the server.
          </p>
        </div>

        <ResultsLocation folder={access.results_root} label="Results folder" />
        <p className="text-xs text-muted-foreground">
          Every run gets its own folder here, named after the run and its date. To use another folder (an
          encrypted or backed-up drive, say), set <code>VIDEOANNOTATOR_RESULTS_DIR</code> (Docker:{' '}
          <code>RESULTS_DIR</code>) and restart the server. Runs from before the change stay where they were
          written.
        </p>
      </CardContent>
    </Card>
  );
}
