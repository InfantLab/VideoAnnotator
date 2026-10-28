import { useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Film, FolderOpen, Upload } from 'lucide-react';
import { apiClient } from '@/api/client';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { ServerFolderPicker, type ServerFolderSelection } from '@/components/ServerFolderPicker';
import { useIngestAccess } from '@/hooks/useIngestAccess';
import { parseApiError } from '@/lib/errorHandling';
import { formatWhen } from '@/lib/when';

const SHOWN = 50;

interface VideoPickerProps {
  onChoose: (path: string) => void;
}

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

  // The server's videos first, so asking about one never means uploading it again (UAT #56).
  const [search, setSearch] = useState('');
  const { data: known } = useQuery({
    queryKey: ['videos', 'ask-picker'],
    queryFn: () => apiClient.listVideos({}),
    retry: false,
  });
  const videos = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return (known?.videos ?? [])
      .filter((v) => v.available && (!needle || v.name.toLowerCase().includes(needle)))
      .sort((x, y) => (y.last_run_at ?? '').localeCompare(x.last_run_at ?? '') || x.name.localeCompare(y.name));
  }, [known, search]);

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
      {(known?.videos.length ?? 0) > 0 && (
        <Card className="space-y-2 p-4" data-testid="server-videos">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-medium">Videos on the server</p>
            <Input
              className="h-8 w-56 text-sm"
              placeholder="Search"
              aria-label="Search the server's videos"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <ul className="max-h-72 space-y-0.5 overflow-y-auto">
            {videos.slice(0, SHOWN).map((video) => (
              <li key={video.path}>
                <button
                  type="button"
                  className="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm hover:bg-muted"
                  onClick={() => onChoose(video.path)}
                  title={video.path}
                  data-testid="server-video"
                >
                  <Film className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="min-w-0 flex-1 truncate">{video.name}</span>
                  {video.last_run_at && <span className="shrink-0 text-xs text-muted-foreground">ran {formatWhen(video.last_run_at)}</span>}
                </button>
              </li>
            ))}
          </ul>
          {videos.length > SHOWN && (
            <p className="text-xs text-muted-foreground">
              {videos.length - SHOWN} more: search to find them.
            </p>
          )}
          {videos.length === 0 && <p className="text-xs text-muted-foreground">No video's name has “{search}” in it.</p>}
        </Card>
      )}
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
        <p className="font-medium">Or drop a video from your computer here</p>
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

    </div>
  );
}
