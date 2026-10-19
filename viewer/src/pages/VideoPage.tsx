import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, PanelRightClose, PanelRightOpen, Sparkles } from 'lucide-react';

import { apiClient } from '@/api/client';
import { VideoPlayer } from '@/components/VideoPlayer';
import type { LaneToggles } from '@/components/lanes/LaneHeader';
import { LaneTimeline } from '@/components/lanes/LaneTimeline';
import { LayersMenu } from '@/components/lanes/LayersMenu';
import { DataTable } from '@/components/video/DataTable';
import { KeysHelp } from '@/components/video/KeysHelp';
import { PlayerBar } from '@/components/video/PlayerBar';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useCurrentUser } from '@/hooks/useCurrentUser';
import { useRunFiles } from '@/hooks/useRunFiles';
import { useVideoKeys } from '@/hooks/useVideoKeys';
import { useVideoRuns } from '@/hooks/useVideoRuns';
import {
  ZOOM_STEP,
  adjacentEvent,
  buildLanes,
  followView,
  frameRateFromRows,
  fullView,
  zoomView,
  type ViewRange,
} from '@/lib/lanes';
import {
  TIMELINE_HEIGHT,
  defaultLanePrefs,
  isCollapsed,
  lanePrefsKey,
  loadLanePrefs,
  overlayOn,
  saveLanePrefs,
  type LanePrefs,
} from '@/lib/lanePrefs';
import { defaultSelection, failedPipelines, runRef } from '@/lib/videoRuns';
import type { StandardAnnotationData } from '@/types/annotations';

const SIDE_PANEL_WIDTH = 360;
const MAX_TIMELINE_SHARE = 0.7;

const clampTimelineHeight = (h: number) =>
  Math.round(Math.min(Math.max(h, TIMELINE_HEIGHT.min), window.innerHeight * MAX_TIMELINE_SHARE));

/** The timeline's top edge: drag to make it taller or shorter (spec 028 FR-007). */
function HeightHandle({ height, onResize, onDone }: { height: number; onResize: (h: number) => void; onDone: (h: number) => void }) {
  const start = useRef<{ y: number; height: number } | null>(null);
  const last = useRef(height);
  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      aria-label="Timeline height"
      aria-valuenow={height}
      tabIndex={0}
      className="h-1.5 shrink-0 cursor-row-resize bg-border/60 hover:bg-primary/40"
      data-testid="timeline-resize"
      onPointerDown={(e) => {
        start.current = { y: e.clientY, height };
        e.currentTarget.setPointerCapture?.(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (!start.current) return;
        last.current = clampTimelineHeight(start.current.height + start.current.y - e.clientY);
        onResize(last.current);
      }}
      onPointerUp={() => {
        if (start.current) onDone(last.current);
        start.current = null;
      }}
      onKeyDown={(e) => {
        if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
        e.preventDefault();
        e.stopPropagation();
        onDone(clampTimelineHeight(height + (e.key === 'ArrowUp' ? 20 : -20)));
      }}
    />
  );
}
const FALLBACK_FPS = 30;

/** Lane preferences, loaded once per user and saved on every change. */
function useLanePrefs(key: string) {
  const [prefs, setPrefs] = useState<LanePrefs>(defaultLanePrefs);
  const loadedFor = useRef<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    loadLanePrefs(key).then((loaded) => {
      if (cancelled) return;
      loadedFor.current = key;
      setPrefs(loaded);
    });
    return () => {
      cancelled = true;
    };
  }, [key]);
  const update = useCallback(
    (change: (prefs: LanePrefs) => LanePrefs) =>
      setPrefs((current) => {
        const next = change(current);
        if (loadedFor.current === key) void saveLanePrefs(key, next);
        return next;
      }),
    [key],
  );
  return [prefs, update] as const;
}

/**
 * The Video page (spec 028): one video, every run's results on labelled lanes,
 * streamed from the server. `/video?path=…[&from=…][&run=<job id>][&t=<seconds>]`.
 */
export default function VideoPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
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
  const lanes = useMemo(() => buildLanes(files.rows, runRefs, failed), [files.rows, runRefs, failed]);
  const frameRate = useMemo(() => frameRateFromRows(files.rows), [files.rows]);

  const token = useQuery({
    queryKey: ['streamToken', info?.path],
    queryFn: () => apiClient.streamToken(info!.path),
    enabled: !!info?.available,
    staleTime: 50 * 60 * 1000,
    retry: false,
  });
  const videoUrl = info && token.data ? apiClient.streamUrl(info.path, token.data.token) : undefined;

  const { currentUser } = useCurrentUser();
  const [prefs, updatePrefs] = useLanePrefs(lanePrefsKey(apiClient.baseURL || window.location.origin, currentUser ? String(currentUser.id) : null));

  const toggles = useMemo<LaneToggles>(
    () => ({
      collapsed: (kind) => isCollapsed(prefs, kind),
      overlayOn: (key) => overlayOn(prefs, key),
      toggleCollapsed: (kind) =>
        updatePrefs((p) => ({ ...p, collapsed: { ...p.collapsed, [kind]: !isCollapsed(p, kind) } })),
      setOverlay: (key, on) => updatePrefs((p) => ({ ...p, overlay: { ...p.overlay, [key]: on } })),
    }),
    [prefs, updatePrefs],
  );
  const showLanes = useCallback(
    (kinds: string[], shown: boolean) =>
      updatePrefs((p) => ({ ...p, collapsed: { ...p.collapsed, ...Object.fromEntries(kinds.map((k) => [k, !shown])) } })),
    [updatePrefs],
  );
  const setOverlays = useCallback(
    (keys: string[], on: boolean) =>
      updatePrefs((p) => ({ ...p, overlay: { ...p.overlay, ...Object.fromEntries(keys.map((k) => [k, on])) } })),
    [updatePrefs],
  );
  const [draggedHeight, setDraggedHeight] = useState<number | null>(null);
  const timelineHeight = draggedHeight ?? prefs.timelineHeight;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [elementDuration, setElementDuration] = useState(0);
  const lastEnd = useMemo(() => files.rows.reduce((m, r) => Math.max(m, r.end ?? r.start), 0), [files.rows]);
  const duration = elementDuration || info?.duration_sec || lastEnd;

  const [chosenView, setChosenView] = useState<ViewRange | null>(null);
  const view = chosenView && chosenView.end <= duration + 1e-6 ? chosenView : fullView(duration);

  const [selectedLaneId, setSelectedLaneId] = useState<string | null>(null);
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [tab, setTab] = useState('data');
  const laneOfRow = useMemo(() => {
    const map = new Map<number, string>();
    for (const lane of lanes) for (const row of lane.rows) for (const e of row.events) if (!map.has(e.row)) map.set(e.row, lane.id);
    return map;
  }, [lanes]);
  const runLabels = useMemo(() => Object.fromEntries(runRefs.map((r) => [r.id, r.label])), [runRefs]);
  const selectedLane = lanes.find((l) => l.id === selectedLaneId) ?? lanes.find((l) => l.status === 'ok') ?? null;

  const seek = useCallback(
    (t: number) => {
      const time = Math.max(0, duration > 0 ? Math.min(t, duration) : t);
      if (videoRef.current) videoRef.current.currentTime = time;
      setCurrentTime(time);
    },
    [duration],
  );

  const startedAt = useRef(false);
  const onDurationChange = useCallback(
    (d: number) => {
      if (Number.isFinite(d) && d > 0) setElementDuration(d);
      if (!startedAt.current && startAt > 0) {
        startedAt.current = true;
        seek(startAt);
      }
    },
    [seek, startAt],
  );

  useEffect(() => {
    if (isPlaying && duration > 0) setChosenView((v) => (v ? followView(v, currentTime, duration) : v));
  }, [isPlaying, currentTime, duration]);

  const playPause = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play();
    else video.pause();
  }, []);

  const stepEvent = useCallback(
    (direction: 1 | -1) => {
      if (!selectedLane) return;
      const t = adjacentEvent(selectedLane.rows, currentTime, direction);
      if (t !== null) seek(t);
    },
    [selectedLane, currentTime, seek],
  );

  useVideoKeys({
    playPause,
    step: (seconds) => seek(currentTime + seconds),
    frameStep: (direction) => seek(currentTime + direction / (frameRate ?? FALLBACK_FPS)),
    previousEvent: () => stepEvent(-1),
    nextEvent: () => stepEvent(1),
    zoomIn: () => setChosenView(zoomView(view, ZOOM_STEP, currentTime, duration)),
    zoomOut: () => setChosenView(zoomView(view, 1 / ZOOM_STEP, currentTime, duration)),
    fit: () => setChosenView(fullView(duration)),
  });

  const annotationData = useMemo<StandardAnnotationData>(
    () => files.parsed ?? { video_info: { filename: info?.name ?? '', duration, width: 0, height: 0 } },
    [files.parsed, info?.name, duration],
  );

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

  const panelOpen = prefs.panelOpen;
  return (
    <div className="flex h-screen flex-col bg-background" data-testid="video-page">
      <header className="flex h-12 shrink-0 items-center gap-2 border-b px-2">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          aria-label="Back"
          title="Back"
          onClick={() => (window.history.length > 1 ? navigate(-1) : navigate('/'))}
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h1 className="min-w-0 max-w-[16rem] truncate text-sm font-semibold" title={info.path}>
          {info.name}
        </h1>
        <div className="mx-2 h-6 w-px bg-border" />
        <PlayerBar
          isPlaying={isPlaying}
          currentTime={currentTime}
          duration={duration}
          playbackRate={playbackRate}
          frameRate={frameRate}
          selectedLane={selectedLane?.label ?? null}
          onPlayPause={playPause}
          onPreviousEvent={() => stepEvent(-1)}
          onNextEvent={() => stepEvent(1)}
          onPlaybackRateChange={(rate) => {
            setPlaybackRate(rate);
            if (videoRef.current) videoRef.current.playbackRate = rate;
          }}
        />
        <div className="ml-auto flex items-center gap-1">
          <Button asChild variant="ghost" size="sm" className="h-8">
            <Link to={`/ask?${new URLSearchParams({ video: info.path })}`}>
              <Sparkles className="mr-1 h-4 w-4" />
              Ask AI
            </Link>
          </Button>
          <KeysHelp />
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            aria-label={panelOpen ? 'Hide the side panel' : 'Show the side panel'}
            title={panelOpen ? 'Hide the side panel' : 'Show the side panel'}
            onClick={() => updatePrefs((p) => ({ ...p, panelOpen: !p.panelOpen }))}
          >
            {panelOpen ? <PanelRightClose className="h-4 w-4" /> : <PanelRightOpen className="h-4 w-4" />}
          </Button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1 bg-black">
          <VideoPlayer
            ref={videoRef}
            videoFile={null}
            videoUrl={videoUrl}
            missingVideoMessage={info.available ? undefined : info.unavailable_reason ?? 'This video is no longer on the server.'}
            annotationData={annotationData}
            currentTime={currentTime}
            overlays={prefs.overlay}
            onTimeUpdate={setCurrentTime}
            onDurationChange={onDurationChange}
            onPlayStateChange={setIsPlaying}
          />
        </div>
        {panelOpen && (
          <aside className="flex shrink-0 flex-col border-l" style={{ width: SIDE_PANEL_WIDTH }} data-testid="side-panel">
            <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col">
              <TabsList className="m-2 grid grid-cols-4">
                <TabsTrigger value="data">Data</TabsTrigger>
                <TabsTrigger value="notes">Notes</TabsTrigger>
                <TabsTrigger value="runs">Runs</TabsTrigger>
                <TabsTrigger value="about">About</TabsTrigger>
              </TabsList>
              <TabsContent value="data" className="mt-0 min-h-0 flex-1">
                <DataTable
                  rows={files.rows}
                  currentTime={currentTime}
                  selectedRow={selectedRow}
                  onRowClick={(index) => {
                    const row = files.rows[index];
                    setSelectedRow(index);
                    const laneId = laneOfRow.get(index);
                    if (laneId) setSelectedLaneId(laneId);
                    if (row) seek(row.start);
                  }}
                  filter={prefs.dataFilter}
                  onFilterChange={(kind) => updatePrefs((p) => ({ ...p, dataFilter: kind }))}
                  runLabels={runLabels}
                />
              </TabsContent>
              <TabsContent value="notes" className="px-3 text-xs text-muted-foreground">
                Notes on this video will be listed here.
              </TabsContent>
              <TabsContent value="runs" className="min-h-0 flex-1 overflow-auto px-3 text-xs">
                <RunsList runs={runRefs} />
              </TabsContent>
              <TabsContent value="about" className="min-h-0 flex-1 overflow-auto px-3 text-xs">
                <ul className="space-y-2">
                  {lanes.map((lane) => (
                    <li key={lane.id}>
                      <span className="font-medium">{lane.label}</span>
                      <span className="text-muted-foreground">
                        {' '}
                        · {lane.pipeline ?? 'notes'}
                        {lane.run ? ` · ${lane.run.label}` : ''}
                      </span>
                      {lane.source.length > 0 && <div className="text-muted-foreground">{lane.source.join(', ')}</div>}
                    </li>
                  ))}
                </ul>
              </TabsContent>
            </Tabs>
          </aside>
        )}
      </div>

      <HeightHandle
        height={timelineHeight}
        onResize={setDraggedHeight}
        onDone={(h) => {
          setDraggedHeight(null);
          updatePrefs((p) => ({ ...p, timelineHeight: h }));
        }}
      />
      <div className="shrink-0" style={{ height: timelineHeight }} data-testid="timeline-area">
        {files.error ? (
          <p className="border-t p-3 text-xs text-destructive">Could not load the results: {files.error.message}</p>
        ) : (
          <LaneTimeline
            lanes={lanes}
            duration={duration}
            currentTime={currentTime}
            view={view}
            onViewChange={setChosenView}
            onSeek={seek}
            selectedLaneId={selectedLane?.id ?? null}
            onSelectLane={setSelectedLaneId}
            selectedRow={selectedRow}
            onSelectRow={(index) => {
              setSelectedRow(index);
              setTab('data');
            }}
            toggles={toggles}
            toolbar={
              <>
                <LayersMenu lanes={lanes} toggles={toggles} showLanes={showLanes} setOverlays={setOverlays} />
                {files.loading && <span className="text-xs text-muted-foreground">Loading results…</span>}
              </>
            }
          />
        )}
      </div>
    </div>
  );
}

function RunsList({ runs }: { runs: { id: string; label: string }[] }) {
  if (!runs.length) return <p className="text-muted-foreground">No runs on this video yet.</p>;
  return (
    <ul className="space-y-1">
      {runs.map((r) => (
        <li key={r.id}>{r.label}</li>
      ))}
    </ul>
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
