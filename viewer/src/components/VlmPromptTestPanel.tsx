import { useEffect, useRef, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Label } from '@/components/ui/label';
import { Loader2, Sparkles, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';
import { apiClient } from '@/api/client';
import { handleAPIError } from '@/api/handleError';
import type { VlmPreviewResponse } from '@/types/pipelines';

interface VlmPromptTestPanelProps {
  prompt: string;
  model: string;
  /** A representative video already picked in the wizard's Upload step, if
   * any — this panel extracts a frame from it client-side (canvas capture)
   * rather than relying on a server-side video_path, since at Configure
   * time the video may not be uploaded to the server yet. */
  videoFile?: File;
}

/** Captures the currently-displayed frame of `video` as a JPEG Blob. */
function captureFrame(video: HTMLVideoElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      reject(new Error('Canvas 2D context unavailable'));
      return;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Frame capture failed'))),
      'image/jpeg',
      0.9
    );
  });
}

/**
 * "Test this prompt" panel for vlm_annotation's Configure step
 * (VideoAnnotator spec 009 US1 / viewer-handoff #2): runs the
 * in-progress prompt/model against one frame of the wizard's selected
 * video and shows the result inline, without leaving the wizard or
 * submitting a job. Always tests a single frame — frame_burst preview
 * needs a server-side video reference this wizard step doesn't have yet
 * (the video isn't uploaded until Step 4), so burst behavior can only be
 * verified by actually submitting a job.
 */
export const VlmPromptTestPanel = ({ prompt, model, videoFile }: VlmPromptTestPanelProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [timestamp, setTimestamp] = useState(0);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!videoFile) {
      setVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(videoFile);
    setVideoUrl(url);
    setDuration(0);
    setTimestamp(0);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  const mutation = useMutation<VlmPreviewResponse, Error, void>({
    mutationFn: async () => {
      const video = videoRef.current;
      if (!video) throw new Error('Video preview not ready');
      const frame = await captureFrame(video);
      return apiClient.previewVlmPrompt({
        image: frame,
        prompt,
        model,
        samplingMode: 'single_frame'
      });
    }
  });

  if (!videoFile) {
    return (
      <p className="text-xs text-muted-foreground">
        Select a video in the Upload step to test this prompt against a real frame.
      </p>
    );
  }

  const canTest = prompt.trim() !== '' && model.trim() !== '' && !mutation.isPending;

  return (
    <Card className="p-3 space-y-3 bg-muted/30">
      <div className="flex items-center justify-between">
        <Label className="text-sm flex items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5" />
          Test this prompt
        </Label>
      </div>

      {videoUrl && (
        <div className="space-y-2">
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full max-h-48 rounded border bg-black object-contain"
            onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
            muted
          />
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={0}
              max={duration || 0}
              step={0.1}
              value={timestamp}
              disabled={duration === 0}
              onChange={(event) => {
                const t = Number(event.target.value);
                setTimestamp(t);
                if (videoRef.current) videoRef.current.currentTime = t;
              }}
              className="flex-1"
            />
            <span className="text-xs text-muted-foreground w-12 text-right">
              {timestamp.toFixed(1)}s
            </span>
          </div>
        </div>
      )}

      <Button
        type="button"
        size="sm"
        variant="secondary"
        disabled={!canTest}
        onClick={() => mutation.mutate()}
      >
        {mutation.isPending ? (
          <>
            <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
            Running… (can take a while on a cold model)
          </>
        ) : (
          'Test prompt on this frame'
        )}
      </Button>

      {mutation.isError && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{handleAPIError(mutation.error)}</AlertDescription>
        </Alert>
      )}

      {mutation.isSuccess && (
        <div className="space-y-2 rounded border bg-background p-3">
          <div className="flex items-center justify-between gap-2">
            <Badge>{mutation.data.label}</Badge>
            <span className="text-[11px] text-muted-foreground">
              {mutation.data.totalTime.toFixed(1)}s total
              {mutation.data.loadTime > 1 && ` (${mutation.data.loadTime.toFixed(1)}s model load)`}
            </span>
          </div>
          {mutation.data.reasoning && mutation.data.reasoning !== mutation.data.label && (
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              {expanded ? 'Hide reasoning' : 'Show reasoning'}
            </button>
          )}
          {expanded && (
            <p className="text-xs whitespace-pre-wrap text-muted-foreground">
              {mutation.data.reasoning}
            </p>
          )}
        </div>
      )}
    </Card>
  );
};
