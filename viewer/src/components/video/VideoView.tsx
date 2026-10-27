import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { userMessage } from '@/lib/userMessage';
import { PageHelp } from '@/components/PageHelp';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, PanelRightClose, PanelRightOpen, Sparkles, StickyNote } from 'lucide-react';

import { apiClient } from '@/api/client';
import { VideoPlayer } from '@/components/VideoPlayer';
import type { LaneToggles } from '@/components/lanes/LaneHeader';
import { LaneTimeline } from '@/components/lanes/LaneTimeline';
import { LayersMenu } from '@/components/lanes/LayersMenu';
import type { RunControls } from '@/components/lanes/RunMenu';
import type { ProvenanceByRun, RunFileRef } from '@/hooks/useRunFiles';
import { AboutPanel } from '@/components/video/AboutPanel';
import { DataTable } from '@/components/video/DataTable';
import { KeysHelp } from '@/components/video/KeysHelp';
import { HeaderActions, VideoContext } from '@/components/video/Header';
import { DebugPanel } from '@/components/DebugPanel';
import { Footer } from '@/components/Footer';
import { videosHref, type ListContext } from '@/lib/listNavigation';
import { NoteEditor, type EditingNote } from '@/components/video/NoteEditor';
import { NotesList } from '@/components/video/NotesList';
import type { NotesApi } from '@/hooks/useNotes';
import { NOTES_KIND } from '@/lib/laneKinds';
import type { ProjectRef, VideoNote, VideoRun } from '@/types/video';
import { RunsPanel } from '@/components/video/RunsPanel';
import { PlayerBar } from '@/components/video/PlayerBar';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useCurrentUser } from '@/hooks/useCurrentUser';
import { useVideoKeys } from '@/hooks/useVideoKeys';
import {
  ZOOM_STEP,
  adjacentEvent,
  buildLanes,
  followView,
  formatClockTenths,
  frameRateFromRows,
  fullView,
  zoomView,
  type FailedPipeline,
  type RunRef,
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
import { noteRow, type Row } from '@/lib/table/rows';
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

/** Where the page's video and results come from: the server, or files from the computer. */
export interface VideoSource {
  name: string;
  /** The server path; null for local files. */
  path: string | null;
  videoUrl?: string;
  videoFile?: File | null;
  missingVideoMessage?: string;
  /** The server's duration, before the element knows its own. */
  durationHint?: number | null;
  rows: Row[];
  parsed: StandardAnnotationData | null;
  runs: RunRef[];
  failed: FailedPipeline[];
  loading: boolean;
  error: Error | null;
  /** Standalone `/viewer`: no runs, no context, notes need a server. */
  local?: boolean;
  /** Choosing and comparing runs (server only). */
  runControls?: RunControls;
  /** `${run}/${pipeline}` → provenance record, for About. */
  provenance?: ProvenanceByRun;
  /** Result files with their download addresses, for About. */
  files?: RunFileRef[];
  /** Every run on the video in full, for the Runs tab (spec 029 R12). */
  runDetails?: VideoRun[];
  /** The video's notes on the server; absent for local files (spec 028 US6). */
  notes?: NotesApi;
  /** The list it was opened from (`from=`), and the projects holding it (spec 028 US7). */
  context?: ListContext | null;
  projects?: ProjectRef[];
  /** The `run=` job, to place the video in a run's list. */
  preferredRun?: string | null;
}

interface VideoViewProps {
  source: VideoSource;
  videoRef: RefObject<HTMLVideoElement>;
  startAt?: number;
  /** Where Back goes; history back when absent. */
  onBack?: () => void;
}

/** The Video page's layout and controls (spec 028), for either source. */
export function VideoView({ source, videoRef, startAt = 0, onBack }: VideoViewProps) {
  const navigate = useNavigate();
  const files = source;
  const runRefs = source.runs;
  const failed = source.failed;
  const compared = useMemo(() => {
    const keys = new Set<string>();
    for (const f of source.files ?? []) if (f.compare) keys.add(`${f.jobId}/${f.pipeline}`);
    return keys;
  }, [source.files]);
  const notes = source.notes;
  // The table's rows and the notes as `note` rows, in time order; which note each row is.
  const { rows, noteAt } = useMemo(() => {
    const stem = source.name.replace(/\.[^.]+$/, '');
    const all: { row: Row; note?: VideoNote }[] = [
      ...source.rows.map((row) => ({ row })),
      ...(notes?.notes ?? []).map((note, i) => ({ row: noteRow(note, stem, i + 1), note })),
    ];
    all.sort((a, b) => a.row.start - b.row.start);
    return { rows: all.map((x) => x.row), noteAt: all.map((x) => x.note) };
  }, [source.rows, source.name, notes?.notes]);
  const lanes = useMemo(() => {
    const built = buildLanes(rows, runRefs, failed, compared);
    if (built.some((l) => l.kind === NOTES_KIND.kind)) return built;
    // The Notes lane is always there, last, saying how to add one (or why not).
    return [
      ...built,
      {
        id: 'notes',
        kind: NOTES_KIND.kind,
        pipeline: null,
        label: NOTES_KIND.label,
        description: NOTES_KIND.description,
        run: null,
        form: NOTES_KIND.form,
        rows: [],
        status: 'empty' as const,
        error: notes ? 'N: note · Shift+N twice: span' : 'Notes need a server',
        overlay: null,
        source: [],
      },
    ];
  }, [rows, runRefs, failed, compared, notes]);
  const frameRate = useMemo(() => frameRateFromRows(source.rows), [source.rows]);


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

  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [elementDuration, setElementDuration] = useState(0);
  const lastEnd = useMemo(() => source.rows.reduce((m, r) => Math.max(m, r.end ?? r.start), 0), [source.rows]);
  const duration = elementDuration || source.durationHint || lastEnd;

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
    [duration, videoRef],
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
  }, [videoRef]);

  const stepEvent = useCallback(
    (direction: 1 | -1) => {
      if (!selectedLane) return;
      const t = adjacentEvent(selectedLane.rows, currentTime, direction);
      if (t !== null) seek(t);
    },
    [selectedLane, currentTime, seek],
  );

  const [editing, setEditing] = useState<EditingNote | null>(null);
  const [spanStart, setSpanStart] = useState<number | null>(null);
  const newNote = useCallback(() => {
    if (!notes) return;
    setSpanStart(null);
    setEditing({ start: currentTime, end: null, text: '' });
  }, [notes, currentTime]);
  const spanNote = useCallback(() => {
    if (!notes) return;
    if (spanStart === null) {
      setSpanStart(currentTime);
      return;
    }
    if (Math.abs(currentTime - spanStart) < 1e-3) return;
    setEditing({ start: Math.min(spanStart, currentTime), end: Math.max(spanStart, currentTime), text: '' });
    setSpanStart(null);
  }, [notes, spanStart, currentTime]);
  const saveNote = (text: string) => {
    if (!notes || !editing) return;
    const done = editing.id
      ? notes.update(editing.id, { text })
      : notes.create({ start: editing.start, end: editing.end, text });
    setEditing(null);
    void done.catch(() => {});
  };
  const editNote = (note: VideoNote) => setEditing({ id: note.id, start: note.start_sec, end: note.end_sec ?? null, text: note.text });

  const [debugOpen, setDebugOpen] = useState(false);
  const shownRuns = useMemo(() => {
    const seen = new Map<string, RunRef>();
    for (const lane of lanes) if (lane.run && !seen.has(lane.run.id)) seen.set(lane.run.id, lane.run);
    return [...seen.values()];
  }, [lanes]);
  const context = source.context ?? null;
  const backTo = context ? videosHref(context) : null;

  useVideoKeys({
    toggleDebug: () => setDebugOpen((open) => !open),
    note: newNote,
    spanNote,
    playPause,
    step: (seconds) => seek(currentTime + seconds),
    frameStep: (direction) => seek(currentTime + direction / (frameRate ?? FALLBACK_FPS)),
    previousEvent: () => stepEvent(-1),
    nextEvent: () => stepEvent(1),
    zoomIn: () => setChosenView(zoomView(view, ZOOM_STEP, currentTime, duration)),
    zoomOut: () => setChosenView(zoomView(view, 1 / ZOOM_STEP, currentTime, duration)),
    fit: () => setChosenView(fullView(duration)),
  });

  const selectRow = (index: number) => {
    setSelectedRow(index);
    const note = noteAt[index];
    if (note) editNote(note);
    else setTab('data');
  };

  const annotationData = useMemo<StandardAnnotationData>(
    () => files.parsed ?? { video_info: { filename: source.name, duration, width: 0, height: 0 } },
    [files.parsed, source.name, duration],
  );

  const panelOpen = prefs.panelOpen;
  return (
    <div className="flex h-screen flex-col bg-background" data-testid="video-page">
      <header className="flex h-12 shrink-0 items-center gap-2 border-b px-2">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          aria-label={backTo ? 'Back to the list' : 'Back'}
          title={backTo ? 'Back to the list' : 'Back'}
          onClick={() =>
            onBack ? onBack() : backTo ? navigate(backTo) : window.history.length > 1 ? navigate(-1) : navigate('/')
          }
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h1 className="min-w-0 max-w-[24rem] truncate text-sm font-semibold" title={source.path ?? source.name}>
          {source.name}
        </h1>
        {source.path && (
          <VideoContext
            path={source.path}
            context={source.context ?? null}
            run={source.preferredRun}
            projects={source.projects ?? []}
          />
        )}
        <div className="ml-auto flex items-center gap-1">
          {source.path && (
            <Button asChild variant="ghost" size="sm" className="h-8">
              <Link to={`/ask?${new URLSearchParams({ video: source.path })}`}>
                <Sparkles className="mr-1 h-4 w-4" />
                Ask AI
              </Link>
            </Button>
          )}
          {notes && (
            <Button variant="ghost" size="sm" className="h-8" onClick={newNote} title="A note at this moment (N); Shift+N twice for a span">
              <StickyNote className="mr-1 h-4 w-4" />
              Note
            </Button>
          )}
          <HeaderActions
            video={source.local || !source.path ? null : { path: source.path, name: source.name }}
            shownRuns={source.local ? [] : shownRuns}
            onDebug={() => setDebugOpen(true)}
            onLoadOther={source.local ? onBack : undefined}
          />
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
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1 bg-black">
            <VideoPlayer
              ref={videoRef}
              videoFile={source.videoFile ?? null}
              videoUrl={source.videoUrl}
              missingVideoMessage={source.missingVideoMessage}
              annotationData={annotationData}
              currentTime={currentTime}
              overlays={prefs.overlay}
              onTimeUpdate={setCurrentTime}
              onDurationChange={onDurationChange}
              onPlayStateChange={setIsPlaying}
            />
            {editing && notes && (
              <NoteEditor
                key={editing.id ?? `new-${editing.start}`}
                note={editing}
                onSave={saveNote}
                onCancel={() => setEditing(null)}
                onDelete={
                  editing.id
                    ? () => {
                        void notes.remove(editing.id!).catch(() => {});
                        setEditing(null);
                      }
                    : undefined
                }
              />
            )}
          </div>
          <div className="shrink-0 border-t bg-card px-2 py-1">
            <PlayerBar
              isPlaying={isPlaying}
              currentTime={currentTime}
              duration={duration}
              playbackRate={playbackRate}
              frameRate={frameRate}
              selectedLane={selectedLane?.label ?? null}
              onPlayPause={playPause}
              onStep={(seconds) => seek(currentTime + seconds)}
              onFrameStep={(direction) => seek(currentTime + direction / (frameRate ?? FALLBACK_FPS))}
              onPreviousEvent={() => stepEvent(-1)}
              onNextEvent={() => stepEvent(1)}
              onSeek={seek}
              onPlaybackRateChange={(rate) => {
                setPlaybackRate(rate);
                if (videoRef.current) videoRef.current.playbackRate = rate;
              }}
            />
          </div>
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
                  rows={rows}
                  currentTime={currentTime}
                  selectedRow={selectedRow}
                  onRowClick={(index) => {
                    const row = rows[index];
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
              <TabsContent value="notes" className="min-h-0 flex-1 overflow-auto px-3 text-xs" data-testid="notes-tab">
                {notes ? (
                  <NotesList
                    notes={notes.notes}
                    onSeek={seek}
                    onEdit={editNote}
                    onDelete={(note) => void notes.remove(note.id).catch(() => {})}
                  />
                ) : (
                  <p className="text-muted-foreground">
                    Notes need a server: they are kept with the video there, for everyone using it.
                  </p>
                )}
                {notes?.error && <p className="mt-2 text-destructive">{userMessage(notes.error)}</p>}
              </TabsContent>
              <TabsContent value="runs" className="min-h-0 flex-1 overflow-auto px-3 text-xs">
                {source.local ? (
                  <p className="text-muted-foreground">Files from this computer, not a run on a server.</p>
                ) : (
                  <RunsPanel runs={source.runDetails ?? []} videoName={source.name} />
                )}
              </TabsContent>
              <TabsContent value="about" className="min-h-0 flex-1 overflow-auto px-3 text-xs">
                <div className="py-2">
                  <PageHelp page="videoPage">Each track is one pipeline&apos;s results; hover a track for where it came from.</PageHelp>
                </div>
                <AboutPanel lanes={lanes} provenance={source.provenance ?? {}} files={source.files ?? []} compared={compared} />
              </TabsContent>
            </Tabs>
          </aside>
        )}
      </div>

      <DebugPanel isOpen={debugOpen} onClose={() => setDebugOpen(false)} />
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
          <p className="border-t p-3 text-xs text-destructive">Could not load the results: {userMessage(files.error)}</p>
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
            onSelectRow={selectRow}
            toggles={toggles}
            runControls={source.runControls}
            toolbar={
              <>
                <LayersMenu lanes={lanes} toggles={toggles} showLanes={showLanes} setOverlays={setOverlays} />
                {files.loading && <span className="text-xs text-muted-foreground">Loading results…</span>}
                {spanStart !== null && (
                  <span className="text-xs text-amber-600" data-testid="span-pending">
                    Span from {formatClockTenths(spanStart)}: Shift+N again to end it
                  </span>
                )}
              </>
            }
          />
        )}
      </div>
      <Footer />
    </div>
  );
}
