import { useCallback, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import { apiErrorEnvelope } from '@/api/handleError';
import { VideoView } from '@/components/video/VideoView';
import { useNotes } from '@/hooks/useNotes';
import { useRunFiles } from '@/hooks/useRunFiles';
import { useVideoRuns } from '@/hooks/useVideoRuns';
import { useVideoStream } from '@/hooks/useVideoStream';
import { parseFrom } from '@/lib/listNavigation';
import type { RunControls } from '@/components/lanes/RunMenu';
import {
  defaultSelection,
  failedPipelines,
  picksOf,
  runOptions,
  runRef,
  sharedRun,
  type RunPick,
  type RunSelection,
} from '@/lib/videoRuns';

/**
 * The Video page (spec 028): one video, every run's results on labelled lanes,
 * streamed from the server. `/video?path=…[&from=…][&run=<job id>][&t=<seconds>]`.
 */
export default function VideoPage() {
  const [params] = useSearchParams();
  // Moving to the next video in a list starts that video afresh (its runs, its choices).
  return <VideoPageFor key={params.get('path') ?? ''} />;
}

function VideoPageFor() {
  const [params, setParams] = useSearchParams();
  const path = params.get('path') ?? '';
  const context = useMemo(() => parseFrom(params.get('from')), [params]);
  const preferredRun = params.get('run');
  const startAt = Number(params.get('t')) || 0;

  const runsQuery = useVideoRuns(path);
  const info = runsQuery.data;
  const runs = info?.runs;

  // The run each pipeline starts on comes from the address; a lane's menu changes it here.
  const initial = useMemo(() => defaultSelection(runs ?? [], preferredRun), [runs, preferredRun]);
  const [chosen, setChosen] = useState<RunSelection>({});
  const [compare, setCompare] = useState<RunPick[]>([]);
  const selection = useMemo(() => ({ ...initial, ...chosen }), [initial, chosen]);
  const picks = useMemo(() => picksOf(selection, compare), [selection, compare]);
  const files = useRunFiles(runs, picks, { name: info?.name ?? '', duration: info?.duration_sec ?? 0 });

  const choose = useCallback(
    (pipeline: string, jobId: string) => {
      const next = { ...selection, [pipeline]: jobId };
      setChosen((c) => ({ ...c, [pipeline]: jobId }));
      setCompare((list) => list.filter((c) => !(c.pipeline === pipeline && c.jobId === jobId)));
      const shared = sharedRun(next);
      setParams(
        (p) => {
          const out = new URLSearchParams(p);
          if (shared) out.set('run', shared);
          else out.delete('run');
          return out;
        },
        { replace: true },
      );
    },
    [selection, setParams],
  );
  const runControls = useMemo<RunControls>(
    () => ({
      options: (pipeline) => runOptions(runs ?? [], pipeline),
      isCompare: (pipeline, jobId) => compare.some((c) => c.pipeline === pipeline && c.jobId === jobId),
      choose,
      compare: (pipeline, jobId) =>
        setCompare((list) => (list.some((c) => c.pipeline === pipeline && c.jobId === jobId) ? list : [...list, { pipeline, jobId }])),
      removeCompare: (pipeline, jobId) => setCompare((list) => list.filter((c) => !(c.pipeline === pipeline && c.jobId === jobId))),
    }),
    [runs, compare, choose],
  );
  const runRefs = useMemo(() => (runs ?? []).map(runRef), [runs]);
  const failed = useMemo(() => failedPipelines(runs ?? [], selection), [runs, selection]);

  const notes = useNotes(info?.path);
  const videoRef = useRef<HTMLVideoElement>(null);
  const stream = useVideoStream(info?.path, !!info?.available, videoRef);

  if (!path) {
    return <PageMessage title="No video chosen" body="Open a video from its results, a run or a project." />;
  }
  if (runsQuery.isLoading) {
    return <PageMessage title="Loading…" />;
  }
  if (runsQuery.error || !info) {
    const envelope = runsQuery.error ? apiErrorEnvelope(runsQuery.error) : null;
    return (
      <PageMessage
        title="Can't open this video"
        body={runsQuery.error instanceof Error ? runsQuery.error.message : 'The server did not answer.'}
        hint={envelope?.hint ?? undefined}
      />
    );
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
        runControls,
        provenance: files.provenance,
        files: files.files,
        notes,
        context,
        projects: info.projects,
        preferredRun,
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
