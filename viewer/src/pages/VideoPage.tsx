import { useMemo, useRef } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import { VideoView } from '@/components/video/VideoView';
import { useRunFiles } from '@/hooks/useRunFiles';
import { useVideoRuns } from '@/hooks/useVideoRuns';
import { useVideoStream } from '@/hooks/useVideoStream';
import { defaultSelection, failedPipelines, runRef } from '@/lib/videoRuns';

/**
 * The Video page (spec 028): one video, every run's results on labelled lanes,
 * streamed from the server. `/video?path=…[&from=…][&run=<job id>][&t=<seconds>]`.
 */
export default function VideoPage() {
  const [params] = useSearchParams();
  const path = params.get('path') ?? '';
  const preferredRun = params.get('run');
  const startAt = Number(params.get('t')) || 0;

  const runsQuery = useVideoRuns(path);
  const info = runsQuery.data;
  const runs = info?.runs;

  const selection = useMemo(() => defaultSelection(runs ?? [], preferredRun), [runs, preferredRun]);
  const files = useRunFiles(runs, selection, { name: info?.name ?? '', duration: info?.duration_sec ?? 0 });
  const runRefs = useMemo(() => (runs ?? []).map(runRef), [runs]);
  const failed = useMemo(() => failedPipelines(runs ?? [], selection), [runs, selection]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const stream = useVideoStream(info?.path, !!info?.available, videoRef);

  if (!path) {
    return <PageMessage title="No video chosen" body="Open a video from its results, a run or a project." />;
  }
  if (runsQuery.isLoading) {
    return <PageMessage title="Loading…" />;
  }
  if (runsQuery.error || !info) {
    const error = runsQuery.error as { message?: string; hint?: string } | null;
    return <PageMessage title="Can't open this video" body={error?.message ?? 'The server did not answer.'} hint={error?.hint} />;
  }

  return (
    <VideoView
      videoRef={videoRef}
      startAt={startAt}
      source={{
        name: info.name,
        path: info.path,
        videoUrl: stream.url,
        missingVideoMessage: !info.available
          ? info.unavailable_reason ?? 'This video is no longer on the server.'
          : stream.error
            ? `Can't play the video: ${stream.error.message}`
            : undefined,
        durationHint: info.duration_sec,
        rows: files.rows,
        parsed: files.parsed,
        runs: runRefs,
        failed,
        loading: files.loading,
        error: files.error,
      }}
    />
  );
}

function PageMessage({ title, body, hint }: { title: string; body?: string; hint?: string }) {
  return (
    <div className="mx-auto max-w-lg p-8" role={body ? 'alert' : undefined}>
      <h1 className="text-lg font-semibold">{title}</h1>
      {body && <p className="mt-2 text-sm">{body}</p>}
      {hint && <p className="mt-1 text-sm text-muted-foreground">{hint}</p>}
      <Link to="/" className="mt-4 inline-block text-sm underline">
        Home
      </Link>
    </div>
  );
}
