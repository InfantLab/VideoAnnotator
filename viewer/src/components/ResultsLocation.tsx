// Where a run's (or a video's) results are on disk, and how to get at them
// (spec 022). Results go to one visible folder, but that only helps if the
// viewer says where it is: researchers start here, not in a file browser.

import { useState } from 'react';
import { Copy, Check, Download, FolderOpen } from 'lucide-react';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useIngestAccess } from '@/hooks/useIngestAccess';
import type { FolderRef } from '@/types/ingest';

interface ResultsLocationProps {
  folder: FolderRef | null | undefined;
  /** "This run's results" or "This video's results". */
  label: string;
  /** Offer the whole run as one zip, without videos. */
  runZip?: { batchId: string };
}

function lastPart(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? 'results';
}

export function ResultsLocation({ folder, label, runZip }: ResultsLocationProps) {
  const { canOpenFolders } = useIngestAccess();
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState<'open' | 'download' | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  if (!folder) return null;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(folder.display_path);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setMessage('Your browser blocked copying. Select the location above and copy it.');
    }
  };

  const open = async () => {
    setBusy('open');
    setMessage(null);
    try {
      await apiClient.openResultsFolder(folder.path);
    } catch (error) {
      if (error instanceof APIError && error.status === 409) {
        setMessage("This computer can't open folders for VideoAnnotator. Copy the location instead.");
      } else {
        setMessage(error instanceof Error ? error.message : String(error));
      }
    } finally {
      setBusy(null);
    }
  };

  const download = async () => {
    if (!runZip) return;
    setBusy('download');
    setMessage(null);
    try {
      const blob = await apiClient.getBatchResultsZip(runZip.batchId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${lastPart(folder.display_path)}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (error) {
      setMessage(`Couldn't download the results: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Card>
      <CardContent className="p-4 space-y-2">
        <p className="text-sm font-medium">{label}</p>
        <p className="font-mono text-xs break-all" data-testid="results-location">
          {folder.display_path}
        </p>
        <div className="flex flex-wrap gap-2">
          {canOpenFolders && (
            <Button variant="outline" size="sm" onClick={open} disabled={busy !== null}>
              <FolderOpen className="h-4 w-4 mr-2" />
              Open folder
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={copy}>
            {copied ? <Check className="h-4 w-4 mr-2" /> : <Copy className="h-4 w-4 mr-2" />}
            {copied ? 'Copied' : 'Copy location'}
          </Button>
          {runZip && (
            <Button variant="outline" size="sm" onClick={download} disabled={busy !== null}>
              <Download className="h-4 w-4 mr-2" />
              {busy === 'download' ? 'Preparing zip…' : 'Download results'}
            </Button>
          )}
        </div>
        {runZip && (
          <p className="text-xs text-muted-foreground">
            The download has every video&apos;s results in this layout, without the videos.
          </p>
        )}
        {message && (
          <p className="text-sm text-muted-foreground" role="status">
            {message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
