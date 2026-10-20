import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { AlertTriangle, Download, Film, Loader2 } from 'lucide-react';
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { AnswerLane } from '@/components/lanes/AnswerLane';
import { LaneLabel } from '@/components/lanes/LaneLabel';
import { Playhead } from '@/components/lanes/Playhead';
import { SampleTicks } from '@/components/lanes/SampleTicks';
import { TimeAxis } from '@/components/lanes/TimeAxis';
import { AnswersTable, type AnswerColumn } from '@/components/ask/AnswersTable';
import { EndpointBadge } from '@/components/ask/EndpointBadge';
import { HistoryList, type HistoryFilters } from '@/components/ask/HistoryList';
import { QuestionPanel } from '@/components/ask/QuestionPanel';
import { RunControls, type RunState } from '@/components/ask/RunControls';
import { SamplingControls } from '@/components/ask/SamplingControls';
import { Tally } from '@/components/ask/Tally';
import { VideoPicker } from '@/components/ask/VideoPicker';
import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';
import { useVlmModels } from '@/hooks/useVlmModels';
import { useRunAnswers } from '@/hooks/useRunAnswers';
import { entryTitle, framesSentNote, fullSampling, MAX_SHOWN, nearest, samplingSummary } from '@/lib/ask';
import { categoryHues } from '@/lib/categoryColors';
import { parseApiError } from '@/lib/errorHandling';
import { estimate } from '@/lib/estimate';
import { saveBlob, zipBlobs } from '@/lib/exports';
import { formatClock } from '@/lib/lanes';
import { chooseMoments, samplePoints, samplingToConfig } from '@/lib/sampling';
import { askWizardStart, wizardState } from '@/lib/wizardStart';
import { DEFAULT_SAMPLING, type AskAnswer, type AskEntry, type AskLibraryPrompt, type AskSampling } from '@/types/ask';

const TRY_COUNT = 10;
const TRY_POLL_MS = 1000;
const FRAMES_DEBOUNCE_MS = 150;
const TOKEN_REFRESH_MS = 45 * 60 * 1000;

const nameOf = (path: string) => path.split(/[\\/]/).pop() || path;

function samplingFromParams(params: URLSearchParams): AskSampling {
  try {
    const raw = params.get('sampling');
    return raw ? fullSampling(JSON.parse(raw) as Partial<AskSampling>) : DEFAULT_SAMPLING;
  } catch {
    return DEFAULT_SAMPLING;
  }
}

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return debounced;
}

/** Frames shown beside the video: the ones a sample sends, from the server's own decoder. */
function FrameStrip({ path, t, sampling }: { path: string; t: number | null; sampling: AskSampling }) {
  const frames = sampling.frames ?? 1;
  const { data, isFetching } = useQuery({
    queryKey: ['ask', 'frames', path, t, sampling.interval, frames, sampling.spacing],
    queryFn: () =>
      apiClient.getFrames({ path, t: t ?? 0, interval: sampling.interval ?? 5, frames, spacing: sampling.spacing }),
    enabled: t !== null && Boolean(path),
    staleTime: Infinity,
    retry: false,
  });
  if (t === null) return null;
  return (
    <div className="space-y-1" data-testid="frame-strip">
      <div className="flex gap-1 overflow-x-auto">
        {data?.frames.map((f) => (
          <figure key={f.frame_number} className="shrink-0">
            <img src={`data:image/jpeg;base64,${f.jpeg_base64}`} alt={`Frame ${f.frame_number}`} className="h-20 rounded border" />
            <figcaption className="text-center text-[10px] text-muted-foreground tabular-nums">
              {f.timestamp_sec.toFixed(2)} s
            </figcaption>
          </figure>
        ))}
        {isFetching && !data && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
      </div>
      {data && data.frames.length < data.requested && (
        <p className="text-xs text-muted-foreground">
          {data.frames.length} of {data.requested} frames: the rest fall outside the video.
        </p>
      )}
    </div>
  );
}

/** What's missing when there's no model to ask, in words (US1 scenario 4). */
function NoModel({ unreachable, detail }: { unreachable: boolean; detail?: string }) {
  return (
    <Alert data-testid="no-model">
      <AlertTriangle className="h-4 w-4" />
      <AlertTitle>{unreachable ? "Ask AI can't reach the model server" : 'No model to ask yet'}</AlertTitle>
      <AlertDescription className="space-y-1 text-sm">
        {unreachable ? (
          <p>
            Ask AI sends frames to a vision model running in Ollama. Install it from{' '}
            <a className="underline" href="https://ollama.com/download" target="_blank" rel="noreferrer">
              ollama.com/download
            </a>{' '}
            and start it, then reload this page.{detail ? ` ${detail}` : ''}
          </p>
        ) : (
          <p>
            The model server is running but has no model. Get one with <code>ollama pull qwen3.5:9b</code> (or any
            vision model), then reload this page.
          </p>
        )}
        <p>
          The <Link className="underline" to="/pipelines">Pipelines</Link> page shows what's ready.
        </p>
      </AlertDescription>
    </Alert>
  );
}

/**
 * Ask AI (spec 027): a question about a video, tried on a frame or a few,
 * kept in a history shared on the server, then run on the whole video with
 * answers arriving as they come.
 */
const AskAI = () => {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { capabilities, error: serverError, isLoading: connecting } = useServerCapabilitiesContext();

  const videoPath = params.get('video') ?? '';
  const entryParam = params.get('entry');

  const [question, setQuestion] = useState(() => params.get('prompt') ?? '');
  const [model, setModel] = useState(() => params.get('model') ?? '');
  const [sampling, setSampling] = useState<AskSampling>(() => samplingFromParams(params));
  const [samplingKey, setSamplingKey] = useState(0);
  const [restoredId, setRestoredId] = useState<string | null>(null);
  const [shown, setShown] = useState<string[]>([]);
  const [time, setTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState(0);
  const [cantPlay, setCantPlay] = useState(false);
  const [filters, setFilters] = useState<HistoryFilters>({ thisVideo: true, mine: false });
  const [pendingSend, setPendingSend] = useState<null | (() => Promise<void>)>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [hovered, setHovered] = useState<{ answer: AskAnswer; sampling: AskSampling } | null>(null);
  const [selected, setSelected] = useState<{ answer: AskAnswer; sampling: AskSampling } | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const models = useVlmModels({ enabled: Boolean(capabilities) });
  const endpointQuery = useQuery({
    queryKey: ['ask', 'endpoint'],
    queryFn: () => apiClient.getEndpoint(),
    enabled: Boolean(capabilities),
    retry: false,
  });
  const endpoint = models.data?.endpoint ?? endpointQuery.data;

  useEffect(() => {
    if (!model && models.data?.models.length) setModel(models.data.models[0]);
  }, [model, models.data]);

  const stream = useQuery({
    queryKey: ['ask', 'stream', videoPath],
    queryFn: () => apiClient.streamToken(videoPath),
    enabled: Boolean(videoPath && capabilities),
    refetchInterval: TOKEN_REFRESH_MS,
    retry: false,
  });
  const duration = stream.data?.duration_sec || mediaDuration;

  const history = useQuery({
    queryKey: ['ask', 'entries', filters.thisVideo ? videoPath : '', filters.mine],
    queryFn: () => apiClient.listEntries({ videoPath: filters.thisVideo ? videoPath : undefined, mine: filters.mine }),
    enabled: Boolean(capabilities),
    retry: false,
  });
  const thisVideoHistory = useQuery({
    queryKey: ['ask', 'entries', videoPath, false],
    queryFn: () => apiClient.listEntries({ videoPath }),
    enabled: Boolean(capabilities && videoPath),
    retry: false,
  });

  const shownQueries = useQueries({
    queries: shown.map((id) => ({
      queryKey: ['ask', 'entry', id],
      queryFn: () => apiClient.getEntry(id),
      refetchInterval: (query: { state: { data?: AskEntry } }) =>
        query.state.data?.status === 'running' ? TRY_POLL_MS : false,
      retry: false,
    })),
  });
  const shownEntries = shownQueries.map((q) => q.data).filter((e): e is AskEntry => Boolean(e));
  const runJobs = shownEntries.filter((e) => e.kind === 'run' && e.job_id).map((e) => e.job_id as string);
  const runAnswers = useRunAnswers(runJobs);

  // A finished try or run belongs in the history list as finished.
  const settled = shownEntries.filter((e) => e.status !== 'running').map((e) => e.id).join(',');
  useEffect(() => {
    if (settled) void queryClient.invalidateQueries({ queryKey: ['ask', 'entries'] });
  }, [settled, queryClient]);

  useEffect(() => {
    setShown([]);
    setSelected(null);
    setCantPlay(false);
    setMediaDuration(0);
  }, [videoPath]);

  const points = useMemo(() => {
    if (!(duration > 0) || !((sampling.interval ?? 0) > 0)) return [];
    return samplePoints(duration, sampling.interval ?? 5, sampling.start ?? 0, sampling.end ?? null, sampling.moments ?? null);
  }, [duration, sampling]);
  const tryMoments = useMemo(() => chooseMoments(points, TRY_COUNT), [points]);
  const allHistory = useMemo(() => thisVideoHistory.data?.entries ?? [], [thisVideoHistory.data]);
  const runEstimate = estimate(points.length, sampling.frames ?? 1, model, allHistory);

  const answersOf = useCallback(
    (entry: AskEntry): AskAnswer[] =>
      entry.kind === 'run' && entry.job_id ? runAnswers[entry.job_id]?.answers ?? [] : entry.answers ?? [],
    [runAnswers],
  );
  const columns: AnswerColumn[] = shownEntries.map((entry) => ({
    key: entry.id,
    title: entryTitle(entry),
    subtitle: `${entry.kind === 'run' ? 'run' : 'try'} · ${entry.model} · ${samplingSummary(entry.sampling)}`,
    answers: answersOf(entry),
    frames: entry.sampling.frames ?? 1,
  }));
  const shownAnswers = columns.flatMap((c) => c.answers);
  const hues = categoryHues(shownAnswers.filter((a) => a.status === 'answered').map((a) => a.label ?? ''));
  const nearestSample = nearest(points, time);
  const nearestAnswer = nearest(
    shownAnswers.map((a) => a.t),
    time,
  );
  const debouncedSample = useDebounced(nearestSample, FRAMES_DEBOUNCE_MS);

  const latestRun = shownEntries.find((e) => e.kind === 'run' && e.job_id);
  const latestRunState = latestRun?.job_id ? runAnswers[latestRun.job_id] : undefined;
  const run: RunState | null = latestRun
    ? {
        status: (latestRunState?.status ?? latestRun.status) as RunState['status'],
        planned: latestRunState?.planned ?? 0,
        done: latestRunState?.done ?? 0,
      }
    : null;
  const anyTryRunning = shownEntries.some((e) => e.kind === 'try' && e.status === 'running');

  const latest = (() => {
    for (const entry of shownEntries) {
      const answers = answersOf(entry);
      if (answers.length) return { answer: answers[answers.length - 1], sampling: fullSampling(entry.sampling) };
    }
    return null;
  })();
  const featured = hovered ?? selected ?? latest;

  const seek = (t: number) => {
    setTime(t);
    if (videoRef.current) videoRef.current.currentTime = t;
  };

  const select = (entry: AskEntry, answer: AskAnswer) => {
    setSelected({ answer, sampling: fullSampling(entry.sampling) });
    seek(answer.t);
  };

  const show = (id: string) => setShown((prev) => [id, ...prev.filter((x) => x !== id)].slice(0, MAX_SHOWN));

  /** Sends nothing to a remote model server until someone has agreed to it (FR-015). */
  const send = (action: () => Promise<void>) => {
    if (endpoint && endpoint.class === 'remote' && !endpoint.confirmed) setPendingSend(() => action);
    else void action();
  };

  const asked = async (make: () => Promise<AskEntry>) => {
    setNotice(null);
    setBusy(true);
    try {
      const entry = await make();
      queryClient.setQueryData(['ask', 'entry', entry.id], entry);
      show(entry.id);
      setRestoredId(entry.id);
      void queryClient.invalidateQueries({ queryKey: ['ask', 'entries'] });
    } catch (e) {
      if (e instanceof APIError && e.status === 409) {
        void queryClient.invalidateQueries({ queryKey: ['ask', 'endpoint'] });
        void queryClient.invalidateQueries({ queryKey: ['videoannotator', 'vlm', 'models'] });
        setPendingSend(() => () => asked(make));
      } else {
        setNotice(parseApiError(e).message);
      }
    } finally {
      setBusy(false);
    }
  };

  const request = { video_path: videoPath, prompt: question, model, base_url: endpoint?.baseUrl ?? null, think: false, sampling };
  const tryHere = () => send(() => asked(() => apiClient.createTry({ ...request, at: Math.round(time * 1000) / 1000 })));
  const tryMany = () => send(() => asked(() => apiClient.createTry({ ...request, count: TRY_COUNT })));
  const runWhole = () => send(() => asked(() => apiClient.createRun(request)));

  const stop = async () => {
    if (!latestRun?.job_id) return;
    try {
      await apiClient.cancelJob(latestRun.job_id);
    } catch (e) {
      setNotice(`Couldn't stop the run: ${parseApiError(e).message}`);
    }
  };

  const config = () => samplingToConfig(sampling);
  const moreVideos = () => navigate('/jobs/new', { state: wizardState(askWizardStart(question, model, config())) });
  const saveWorkflow = async () => {
    try {
      const preset = await apiClient.createPreset({
        name: `${entryTitle({ name: null, prompt: question })} (${model})`,
        selected_pipelines: ['vlm_annotation'],
        config: { vlm_annotation: { ...config(), prompt: question, model } },
      });
      setNotice(`Saved as workflow “${preset.name}”.`);
    } catch (e) {
      setNotice(`Couldn't save the workflow: ${parseApiError(e).message}`);
    }
  };

  const knownQuestions = useMemo(
    () =>
      new Set([
        ...(history.data?.entries ?? []).map((e) => e.prompt),
        ...(history.data?.library ?? []).map((p) => p.text),
        ...allHistory.map((e) => e.prompt),
      ]),
    [history.data, allHistory],
  );

  const restoreFields = useCallback((entry: Pick<AskEntry, 'prompt' | 'model' | 'sampling'>) => {
    setQuestion(entry.prompt);
    setModel(entry.model);
    setSampling(fullSampling(entry.sampling));
    setSamplingKey((k) => k + 1);
  }, []);

  /** A question typed but never tried is kept before another is put back (US2 scenario 2). */
  const keepDraft = async () => {
    if (!question.trim() || knownQuestions.has(question) || !videoPath || !model) return;
    try {
      await apiClient.createDraft({ video_path: videoPath, prompt: question, model, think: false, sampling });
    } catch {
      // The restore still happens; the draft was a courtesy.
    }
  };

  const restore = async (entry: AskEntry) => {
    await keepDraft();
    restoreFields(entry);
    setRestoredId(entry.id);
    const next = new URLSearchParams(params);
    next.set('entry', entry.id);
    next.set('video', entry.video_path);
    setParams(next);
    if (entry.kind !== 'draft' && entry.video_path === videoPath) show(entry.id);
    void queryClient.invalidateQueries({ queryKey: ['ask', 'entries'] });
  };

  const restoreLibrary = async (prompt: AskLibraryPrompt) => {
    await keepDraft();
    setQuestion(prompt.text);
    if (prompt.models?.length) setModel(prompt.models[0]);
  };

  // Opened at /ask?entry=<id>: put that entry back once.
  useEffect(() => {
    if (!entryParam || entryParam === restoredId || !capabilities) return;
    let cancelled = false;
    void apiClient
      .getEntry(entryParam)
      .then((entry) => {
        if (cancelled) return;
        restoreFields(entry);
        setRestoredId(entry.id);
        if (entry.kind !== 'draft') setShown((prev) => (prev.includes(entry.id) ? prev : [entry.id, ...prev].slice(0, MAX_SHOWN)));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [entryParam, restoredId, capabilities, restoreFields]);

  const exportShown = async () => {
    try {
      const files = await Promise.all(
        shownEntries.map(async (entry) => {
          const download = await apiClient.exportEntryTable(entry.id);
          return { name: download.filename ?? `${entryTitle(entry)}.csv`, blob: download.blob };
        }),
      );
      if (files.length === 1) saveBlob(files[0].blob, files[0].name);
      else if (files.length > 1) saveBlob(await zipBlobs(files), `${nameOf(videoPath)} answers.zip`);
    } catch (e) {
      setNotice(`Couldn't export the answers: ${parseApiError(e).message}`);
    }
  };

  if (!capabilities) {
    return (
      <div className="container mx-auto max-w-3xl p-6">
        <h1 className="mb-4 text-3xl font-bold">Ask AI</h1>
        {connecting ? (
          <p className="text-sm text-muted-foreground">Connecting to the server…</p>
        ) : (
          <Alert data-testid="needs-server">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Ask AI needs a VideoAnnotator server and a model</AlertTitle>
            <AlertDescription>
              The server reads your video and asks a vision model running beside it. Connect to a server in{' '}
              <Link className="underline" to="/settings">
                Settings
              </Link>
              , or start one on this computer (
              <a
                className="underline"
                href="https://github.com/InfantLab/VideoAnnotator/blob/master/docs/installation/INSTALLATION.md"
                target="_blank"
                rel="noreferrer"
              >
                how
              </a>
              ).{serverError ? ` (${parseApiError(serverError).message})` : ''}
            </AlertDescription>
          </Alert>
        )}
      </div>
    );
  }

  const noModels = models.isError || (models.data && models.data.models.length === 0);
  const unreachableDetail = models.error ? parseApiError(models.error).hint : undefined;

  return (
    <div className="container mx-auto max-w-[1600px] space-y-4 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Ask AI</h1>
          {videoPath && (
            <p className="flex items-center gap-1 text-sm text-muted-foreground" title={videoPath}>
              <Film className="h-4 w-4" /> {nameOf(videoPath)}
              <Button variant="link" size="sm" className="h-auto p-0 pl-2" onClick={() => setParams(new URLSearchParams())}>
                Choose another video
              </Button>
            </p>
          )}
        </div>
        <EndpointBadge endpoint={endpoint} model={model} />
      </div>

      {noModels && <NoModel unreachable={models.isError} detail={unreachableDetail} />}
      {notice && (
        <Alert>
          <AlertDescription>{notice}</AlertDescription>
        </Alert>
      )}

      {!videoPath ? (
        <VideoPicker
          onChoose={(path) => {
            const next = new URLSearchParams(params);
            next.set('video', path);
            setParams(next);
          }}
        />
      ) : (
        <>
          <div className="grid gap-4 lg:grid-cols-[minmax(320px,420px)_1fr]">
            <Card className="space-y-5 p-4">
              <QuestionPanel
                question={question}
                onQuestion={setQuestion}
                model={model}
                onModel={setModel}
                onTryHere={tryHere}
                onTryMoments={tryMany}
                tryCount={tryMoments.length}
                busy={busy || anyTryRunning}
                disabled={Boolean(noModels)}
              />
              <SamplingControls key={samplingKey} sampling={sampling} onChange={setSampling} count={points.length} estimate={runEstimate} />
              <RunControls
                run={run}
                samples={points.length}
                disabled={Boolean(noModels) || !question.trim() || !model}
                onRun={runWhole}
                onStop={() => void stop()}
                onMoreVideos={moreVideos}
                onSaveWorkflow={() => void saveWorkflow()}
              />
              <div className="space-y-2 border-t pt-3">
                <h2 className="text-sm font-semibold">History</h2>
                <HistoryList
                  entries={history.data?.entries ?? []}
                  library={history.data?.library ?? []}
                  filters={filters}
                  onFilters={setFilters}
                  shown={shown}
                  canShow={(entry) => entry.video_path === videoPath && shown.length < MAX_SHOWN}
                  restoredId={restoredId}
                  onRestore={(entry) => void restore(entry)}
                  onToggleShow={(entry) =>
                    setShown((prev) => (prev.includes(entry.id) ? prev.filter((x) => x !== entry.id) : [...prev, entry.id].slice(0, MAX_SHOWN)))
                  }
                  onRename={(entry, name) =>
                    void apiClient.updateEntry(entry.id, { name }).then(() => queryClient.invalidateQueries({ queryKey: ['ask'] }))
                  }
                  onStar={(entry, starred) =>
                    void apiClient.updateEntry(entry.id, { starred }).then(() => queryClient.invalidateQueries({ queryKey: ['ask'] }))
                  }
                  onRestoreLibrary={(prompt) => void restoreLibrary(prompt)}
                  onLibraryUpdate={(prompt, change) =>
                    void apiClient
                      .updatePrompt(prompt.sha256, change)
                      .then(() => queryClient.invalidateQueries({ queryKey: ['ask', 'entries'] }))
                  }
                  onLibraryDelete={(prompt) =>
                    void apiClient
                      .deletePrompt(prompt.sha256)
                      .then(() => queryClient.invalidateQueries({ queryKey: ['ask', 'entries'] }))
                      .catch((e) => setNotice(`Couldn't delete that question: ${parseApiError(e).message}`))
                  }
                />
              </div>
            </Card>

            <div className="space-y-3">
              {stream.data ? (
                <video
                  ref={videoRef}
                  data-testid="ask-video"
                  className="max-h-[60vh] w-full rounded bg-black"
                  src={apiClient.streamUrl(videoPath, stream.data.token)}
                  controls
                  preload="metadata"
                  onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
                  onSeeked={(e) => setTime(e.currentTarget.currentTime)}
                  onLoadedMetadata={(e) => {
                    const v = e.currentTarget;
                    if (Number.isFinite(v.duration)) setMediaDuration(v.duration);
                    if (v.videoWidth === 0) setCantPlay(true);
                  }}
                  onError={() => setCantPlay(true)}
                />
              ) : stream.isError ? (
                <Alert variant="destructive">
                  <AlertDescription>Can't open this video: {parseApiError(stream.error).message}</AlertDescription>
                </Alert>
              ) : (
                <div className="flex aspect-video items-center justify-center rounded bg-muted">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              )}
              {cantPlay && (
                <Alert data-testid="cant-play">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    This browser can't show this video: its format (often H.265) isn't supported here. Questions and runs
                    still work, because the server reads the video itself. Chrome, Edge or Safari usually play it.
                  </AlertDescription>
                </Alert>
              )}

              {(sampling.frames ?? 1) > 1 && (
                <div>
                  <p className="text-xs text-muted-foreground">
                    The frames the sample at {debouncedSample !== null ? formatClock(debouncedSample) : '…'} sends:
                  </p>
                  <FrameStrip path={videoPath} t={debouncedSample} sampling={sampling} />
                </div>
              )}

              {featured && (
                <Card className="space-y-2 p-3" data-testid="featured-answer">
                  <p className="text-sm">
                    <span className="text-muted-foreground tabular-nums">{formatClock(featured.answer.t)} </span>
                    {featured.answer.status === 'answered' ? (
                      <span className="font-semibold">{featured.answer.label}</span>
                    ) : featured.answer.status === 'error' ? (
                      <span className="text-destructive">No answer: {featured.answer.error}</span>
                    ) : (
                      <span className="italic text-muted-foreground">Skipped: no readable frame here</span>
                    )}
                    {framesSentNote(featured.answer, featured.sampling.frames ?? 1) && (
                      <span className="text-xs text-muted-foreground"> ({framesSentNote(featured.answer, featured.sampling.frames ?? 1)})</span>
                    )}
                  </p>
                  {featured.answer.reasoning && featured.answer.reasoning !== featured.answer.label && (
                    <p className="whitespace-pre-wrap text-xs text-muted-foreground">{featured.answer.reasoning}</p>
                  )}
                  {featured.answer.status !== 'skipped' && (
                    <FrameStrip path={videoPath} t={featured.answer.t} sampling={featured.sampling} />
                  )}
                </Card>
              )}
            </div>
          </div>

          <Card className="space-y-2 p-3">
            <div className="grid grid-cols-[minmax(120px,200px)_1fr] items-center gap-x-3 gap-y-1">
              <span />
              <TimeAxis duration={duration} />
              <span className="text-xs text-muted-foreground">
                {points.length} sample{points.length === 1 ? '' : 's'}
              </span>
              <div className="relative">
                <SampleTicks times={points} duration={duration} highlight={nearestSample} onSeek={seek} />
                <Playhead time={time} duration={duration} />
              </div>
              {shownEntries.map((entry) => (
                <div key={entry.id} className="contents">
                  <LaneLabel title={entryTitle(entry)} subtitle={`${entry.kind === 'run' ? 'run' : 'try'} · ${entry.model}`} />
                  <div className="relative">
                    <AnswerLane
                      markers={answersOf(entry)}
                      duration={duration}
                      hues={hues}
                      highlight={nearestAnswer}
                      onSeek={(t) => {
                        const answer = answersOf(entry).find((a) => Math.abs(a.t - t) < 1e-6);
                        if (answer) select(entry, answer);
                        else seek(t);
                      }}
                      onHover={(m) => setHovered(m ? { answer: m, sampling: fullSampling(entry.sampling) } : null)}
                    />
                    <Playhead time={time} duration={duration} />
                  </div>
                </div>
              ))}
            </div>
            {!shownEntries.length && (
              <p className="text-xs text-muted-foreground">Answers appear here as tracks, one per try or run you show.</p>
            )}
          </Card>

          <div className="grid gap-4 lg:grid-cols-[1fr_minmax(240px,320px)]">
            <AnswersTable columns={columns} highlight={nearestAnswer} onSeek={seek} />
            <Card className="space-y-3 p-3">
              <h2 className="text-sm font-semibold">Answers so far</h2>
              <Tally answers={shownAnswers} />
              <Button variant="outline" size="sm" className="gap-1" disabled={!shownEntries.length} onClick={() => void exportShown()}>
                <Download className="h-4 w-4" /> Export table
              </Button>
            </Card>
          </div>
        </>
      )}

      <AlertDialog open={pendingSend !== null} onOpenChange={(open) => !open && setPendingSend(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Send frames to {endpoint?.host}?</AlertDialogTitle>
            <AlertDialogDescription>
              The model server at {endpoint?.host} is not on this computer or your local network. Frames from your video
              will leave this computer and be sent there. Agreeing counts for everyone using this VideoAnnotator server.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Don't send</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                const action = pendingSend;
                setPendingSend(null);
                if (!endpoint?.baseUrl || !action) return;
                void apiClient
                  .confirmEndpoint(endpoint.baseUrl)
                  .then(async () => {
                    await queryClient.invalidateQueries({ queryKey: ['ask', 'endpoint'] });
                    await queryClient.invalidateQueries({ queryKey: ['videoannotator', 'vlm', 'models'] });
                    await action();
                  })
                  .catch((e) => setNotice(`Couldn't record your agreement: ${parseApiError(e).message}`));
              }}
            >
              Send to {endpoint?.host}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default AskAI;
