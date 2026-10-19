import { useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Film, FolderOpen, Upload } from 'lucide-react';
import { apiClient } from '@/api/client';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { ServerFolderPicker, type ServerFolderSelection } from '@/components/ServerFolderPicker';
import { useIngestAccess } from '@/hooks/useIngestAccess';
import { parseApiError } from '@/lib/errorHandling';

interface VideoPickerProps {
  onChoose: (path: string) => void;
}

const nameOf = (path: string) => path.split(/[\\/]/).pop() || path;

/**
 * The video to ask about (FR-001): dropped or chosen here and uploaded once,
 * picked in a folder shared with the server, or one asked about or run before.
 */
export function VideoPicker({ onChoose }: VideoPickerProps) {
  const input = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [folder, setFolder] = useState<ServerFolderSelection | null>(null);
  const [showFolders, setShowFolders] = useState(false);
  const { canReadInPlace } = useIngestAccess();

  const { data: entries } = useQuery({
    queryKey: ['ask', 'entries', 'recent-videos'],
    queryFn: () => apiClient.listEntries({ limit: 100 }),
    retry: false,
  });
  const { data: jobs } = useQuery({ queryKey: ['jobs', 'ask-recent'], queryFn: () => apiClient.getJobs(1, 50), retry: false });

  const recent = useMemo(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const path of [
      ...(entries?.entries ?? []).map((e) => e.video_path),
      ...(jobs?.jobs ?? []).map((j) => j.video_path).filter((p): p is string => Boolean(p)),
    ]) {
      if (!seen.has(path)) {
        seen.add(path);
        out.push(path);
      }
    }
    return out.slice(0, 8);
  }, [entries, jobs]);

  const upload = async (file: File) => {
    setError(null);
    setProgress(0);
    try {
      const video = await apiClient.uploadVideo(file, setProgress);
      onChoose(video.path);
    } catch (e) {
      setError(parseApiError(e).message);
    } finally {
      setProgress(null);
    }
  };

  const chooseFolder = (selection: ServerFolderSelection | null) => {
    setFolder(selection);
    const file = selection?.files?.[0];
    if (selection && file) onChoose(`${selection.path.replace(/[\\/]$/, '')}/${file}`);
  };

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <Card
        className={`flex flex-col items-center gap-3 border-2 border-dashed p-10 text-center ${dragging ? 'border-primary bg-muted' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files[0];
          if (file) void upload(file);
        }}
        data-testid="video-drop"
      >
        <Upload className="h-8 w-8 text-muted-foreground" />
        <p className="font-medium">Drop a video here</p>
        <p className="text-sm text-muted-foreground">It's kept on your VideoAnnotator server, once, to ask about and run on.</p>
        <Button onClick={() => input.current?.click()} disabled={progress !== null}>
          Choose a video
        </Button>
        <input
          ref={input}
          type="file"
          accept="video/*"
          className="hidden"
          data-testid="video-input"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void upload(file);
          }}
        />
        {progress !== null && (
          <div className="w-full max-w-sm space-y-1">
            <Progress value={progress * 100} />
            <p className="text-xs text-muted-foreground">Uploading… {Math.round(progress * 100)}%</p>
          </div>
        )}
        {error && (
          <Alert variant="destructive" className="text-left">
            <AlertDescription>Couldn't upload the video: {error}</AlertDescription>
          </Alert>
        )}
      </Card>

      {canReadInPlace && (
        <Card className="space-y-3 p-4">
          <Button variant="outline" className="gap-2" onClick={() => setShowFolders((v) => !v)}>
            <FolderOpen className="h-4 w-4" /> Pick a video in a folder on the server
          </Button>
          {showFolders && (
            <>
              <ServerFolderPicker selection={folder} onSelect={chooseFolder} />
              <p className="text-xs text-muted-foreground">Tick one video to ask about it; it's read where it is.</p>
            </>
          )}
        </Card>
      )}

      {recent.length > 0 && (
        <Card className="p-4">
          <p className="mb-2 text-sm font-medium">Recent videos</p>
          <ul className="space-y-1">
            {recent.map((path) => (
              <li key={path}>
                <button
                  type="button"
                  className="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm hover:bg-muted"
                  onClick={() => onChoose(path)}
                  title={path}
                >
                  <Film className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{nameOf(path)}</span>
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
