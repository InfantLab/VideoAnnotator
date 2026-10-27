import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowDown, ArrowRight, ArrowUp, Search } from 'lucide-react';

import { FileName } from '@/components/FileName';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { kindColour, kindOfPipeline } from '@/lib/resultKinds';
import type { ListContext } from '@/lib/listNavigation';
import { formatDay } from '@/lib/when';
import { folderLabel, hrefOf, shortFolder, videoLink } from '@/lib/videosScope';
import { useResultValues, valueKey } from '@/hooks/useResultValues';
import type { ResultValue, ResultValues, VideoItem, VideoResult } from '@/types/videos';

const ROW_HEIGHT = 40;
/** Past this many rows only those on screen are drawn (as the Video page's Data tab). */
const WINDOW_FROM = 200;
const OVERSCAN = 10;
const VIEWPORT = 600;

const MARKS: Record<string, { mark: string; word: string; className: string }> = {
  done: { mark: '✓', word: 'done', className: 'text-emerald-600 dark:text-emerald-400' },
  running: { mark: '●', word: 'running', className: 'text-blue-600 dark:text-blue-400 animate-pulse' },
  waiting: { mark: '◌', word: 'waiting', className: 'text-muted-foreground' },
  failed: { mark: '✕', word: 'failed', className: 'text-destructive' },
};

function ResultMark({
  title,
  result,
  inRun,
  value,
}: {
  title: string;
  result: VideoResult | undefined;
  inRun?: string;
  value?: ResultValue;
}) {
  if (!result) {
    return (
      <span className="text-muted-foreground" title={`${title}: not run`} aria-label={`${title}: not run`}>
        –
      </span>
    );
  }
  if (result.now && result.now_run_id === inRun) result = { ...result, state: result.now, error: null };
  else if (result.now && result.now_run_id) {
    // A newer run is going on over an older result (UAT #43): show that, linked to the run.
    const now = MARKS[result.now] ?? MARKS.waiting;
    const label = `${title}: ${now.word} again in a newer run (open it)`;
    return (
      <Link
        to={hrefOf({ run: result.now_run_id })}
        className={cn('inline-flex items-center gap-1 text-xs', now.className)}
        title={label}
        aria-label={label}
        onClick={(event) => event.stopPropagation()}
        data-testid="result-now"
      >
        {now.mark} {now.word}
      </Link>
    );
  }
  // What was found, not that it ran (UAT #10, viewer design P4).
  if (result.state === 'done' && value) {
    const label = `${title}: ${value.text}${value.detail ? ` (${value.detail})` : ''}`;
    return (
      <span className={cn('whitespace-nowrap', value.count === 0 && 'text-muted-foreground')} title={label} aria-label={label}>
        {value.text}
      </span>
    );
  }
  const look = MARKS[result.state] ?? MARKS.waiting;
  const label = `${title}: ${look.word}${result.error ? `: ${result.error}` : ''}`;
  if (result.state === 'failed') {
    return (
      <span className={cn('whitespace-nowrap text-xs', look.className)} title={label} aria-label={label}>
        {look.mark} failed
      </span>
    );
  }
  return (
    <span className={look.className} title={label} aria-label={label}>
      {look.mark}
    </span>
  );
}

type SortKey = { column: 'name' | 'folder' | string; descending: boolean };

/** A pipeline column's sort value: what was found, then done without a value, failed, not run. */
function rank(video: VideoItem, pipeline: string, values: ResultValues): number {
  const result = video.results[pipeline];
  if (!result) return -3;
  if (result.state === 'failed') return -2;
  if (result.state !== 'done') return -1.5;
  const value = values[valueKey(result.job_id, pipeline)];
  return value ? value.count : -1;
}

function sorted(videos: VideoItem[], sort: SortKey | null, values: ResultValues): VideoItem[] {
  if (!sort) return videos;
  const sign = sort.descending ? -1 : 1;
  const byName = (a: VideoItem, b: VideoItem) => a.name.localeCompare(b.name, undefined, { numeric: true });
  return [...videos].sort((a, b) => {
    if (sort.column === 'name') return sign * byName(a, b);
    if (sort.column === 'folder') return sign * a.folder.localeCompare(b.folder) || byName(a, b);
    const ra = rank(a, sort.column, values);
    const rb = rank(b, sort.column, values);
    // Not run and failed stay at the bottom either way.
    if (ra < 0 || rb < 0) return rb - ra || byName(a, b);
    return sign * (ra - rb) || byName(a, b);
  });
}

/** Where a video is: its folder, or for an upload when it came (UAT #8: copies of one name). */
function whereText(video: VideoItem): string {
  if (video.folder !== 'uploads') return shortFolder(video.folder);
  const day = formatDay(video.added_at);
  return day ? `Uploaded ${day}` : 'Uploaded';
}

interface VideosTableProps {
  videos: VideoItem[];
  pipelines: string[];
  nameOf: (pipeline: string) => string;
  taskOf?: (pipeline: string) => string;
  selected: Set<string>;
  onSelect: (paths: string[], on: boolean) => void;
  /** The list rows open in, so ‹ › on the Video page move through it. */
  context: ListContext;
  filter: string;
  onFilter: (text: string) => void;
  /** What the table is, above it: "Videos in this run (3)" (UAT #4). */
  caption?: string;
  /** An explicit Open → per row, where opening a video is the next step (a run's page, UAT #3). */
  openColumn?: boolean;
}

/**
 * One row per video: its folder, projects and, per pipeline, where its newest
 * result stands (spec 029 FR-002). `/` focuses the filter; `Enter` opens the
 * focused row and `Space` selects it.
 */
export function VideosTable({
  videos,
  pipelines,
  nameOf,
  taskOf = () => '',
  selected,
  onSelect,
  context,
  filter,
  onFilter,
  caption,
  openColumn = false,
}: VideosTableProps) {
  // In a project, every row is in it: show only its other projects (UAT #45).
  const shownProjects = (video: VideoItem) =>
    context.kind === 'project' ? video.projects.filter((p) => p.id !== context.id) : video.projects;
  // An empty Projects column only takes room (UAT #5).
  const anyProjects = videos.some((v) => shownProjects(v).length > 0);
  const values = useResultValues(videos, pipelines);
  const [sort, setSort] = useState<SortKey | null>(null);
  const shown = useMemo(() => sorted(videos, sort, values), [videos, sort, values]);
  const sortBy = (column: string) =>
    setSort((now) =>
      now?.column === column
        ? { column, descending: !now.descending }
        : { column, descending: column !== 'name' && column !== 'folder' },
    );
  const navigate = useNavigate();
  const filterRef = useRef<HTMLInputElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const windowed = videos.length > WINDOW_FROM;

  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key !== '/' || event.ctrlKey || event.metaKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return;
      event.preventDefault();
      filterRef.current?.focus();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const [first, last] = useMemo(() => {
    if (!windowed) return [0, videos.length];
    const start = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
    const end = Math.min(videos.length, Math.ceil((scrollTop + VIEWPORT) / ROW_HEIGHT) + OVERSCAN);
    return [start, end];
  }, [windowed, scrollTop, videos.length]);

  const allSelected = videos.length > 0 && videos.every((v) => selected.has(v.path));
  const someSelected = videos.some((v) => selected.has(v.path));

  const onRowKey = (event: KeyboardEvent<HTMLTableRowElement>, video: VideoItem) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === 'Enter') {
      event.preventDefault();
      navigate(videoLink(video, context));
    } else if (event.key === ' ') {
      event.preventDefault();
      onSelect([video.path], !selected.has(video.path));
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const sibling = (event.key === 'ArrowDown'
        ? event.currentTarget.nextElementSibling
        : event.currentTarget.previousElementSibling) as HTMLElement | null;
      sibling?.focus();
    }
  };

  const rows = shown.slice(first, last).map((video) => (
    <tr
      key={video.path}
      tabIndex={0}
      data-testid="video-row"
      aria-selected={selected.has(video.path)}
      className={cn(
        'cursor-pointer border-b hover:bg-muted/50 focus:bg-muted/60 focus:outline-none',
        selected.has(video.path) && 'bg-primary/5',
      )}
      style={{ height: ROW_HEIGHT }}
      onClick={() => navigate(videoLink(video, context))}
      onKeyDown={(event) => onRowKey(event, video)}
    >
      <td className="w-10 px-3" onClick={(event) => event.stopPropagation()}>
        <Checkbox
          checked={selected.has(video.path)}
          onCheckedChange={(on) => onSelect([video.path], on === true)}
          aria-label={`Select ${video.name}`}
        />
      </td>
      <td className="max-w-[24rem] px-2">
        <div className="flex min-w-0 items-center gap-2">
          <Link
            to={videoLink(video, context)}
            className="min-w-0 font-medium hover:underline"
            onClick={(event) => event.stopPropagation()}
          >
            <FileName name={video.name} title={video.path} />
          </Link>
          {!video.available && (
            <Badge variant="outline" className="shrink-0 text-[10px] text-muted-foreground" title="The file has moved or been deleted; its results are kept">
              not on the server
            </Badge>
          )}
        </div>
      </td>
      {openColumn && (
        <td className="w-20 px-2">
          <Link
            to={videoLink(video, context)}
            className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
            onClick={(event) => event.stopPropagation()}
            data-testid="open-video"
          >
            Open <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
          </Link>
        </td>
      )}
      <td className="max-w-[14rem] truncate px-2 text-muted-foreground" title={folderLabel(video.folder)}>
        {whereText(video)}
      </td>
      {anyProjects && (
        <td className="max-w-[12rem] px-2" onClick={(event) => event.stopPropagation()}>
          <ProjectChips projects={shownProjects(video)} />
        </td>
      )}
      {pipelines.map((pipeline) => (
        <td key={pipeline} className="px-2 text-center">
          <ResultMark
            title={nameOf(pipeline)}
            result={video.results[pipeline]}
            inRun={context.kind === 'run' ? context.id : undefined}
            value={video.results[pipeline] ? values[valueKey(video.results[pipeline].job_id, pipeline)] : undefined}
          />
        </td>
      ))}
    </tr>
  ));

  return (
    <div className="space-y-2">
      {caption && <h2 className="text-base font-semibold">{caption}</h2>}
      <div className="relative max-w-xs">
        <Search className="pointer-events-none absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" aria-hidden="true" />
        <Input
          ref={filterRef}
          value={filter}
          onChange={(event) => onFilter(event.target.value)}
          placeholder="Filter by name  ( / )"
          aria-label="Filter videos by name"
          className="h-9 pl-8"
        />
      </div>
      <div
        className="overflow-auto rounded-md border"
        style={windowed ? { maxHeight: VIEWPORT } : undefined}
        onScroll={windowed ? (event) => setScrollTop(event.currentTarget.scrollTop) : undefined}
        data-testid="videos-table"
      >
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10 bg-background">
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="w-10 px-3 py-2">
                <Checkbox
                  checked={allSelected ? true : someSelected ? 'indeterminate' : false}
                  onCheckedChange={(on) => onSelect(videos.map((v) => v.path), on === true)}
                  aria-label="Select every video shown"
                />
              </th>
              {/* Table cells ignore max-width, so the name needs a floor to show both ends (UAT #7). */}
              <th className="min-w-[20rem] px-2 py-2 font-medium" aria-sort={ariaSort(sort, 'name')}>
                <SortButton column="name" sort={sort} onSort={sortBy}>Video</SortButton>
              </th>
              {openColumn && <th className="w-20 px-2 py-2" aria-label="Open" />}
              <th className="px-2 py-2 font-medium" aria-sort={ariaSort(sort, 'folder')}>
                <SortButton column="folder" sort={sort} onSort={sortBy}>Folder</SortButton>
              </th>
              {anyProjects && <th className="px-2 py-2 font-medium">{context.kind === 'project' ? 'Also in' : 'Projects'}</th>}
              {pipelines.map((pipeline) => {
                const kind = kindOfPipeline(pipeline);
                const Icon = kind?.icon;
                const task = taskOf(pipeline);
                return (
                  <th
                    key={pipeline}
                    className="max-w-[7.5rem] px-2 py-2 text-center align-bottom font-medium leading-tight"
                    title={task ? `${nameOf(pipeline)} · ${task}` : nameOf(pipeline)}
                    aria-sort={ariaSort(sort, pipeline)}
                  >
                    <button
                      type="button"
                      className="mx-auto flex flex-col items-center gap-1 rounded hover:text-foreground"
                      onClick={() => sortBy(pipeline)}
                    >
                      {Icon && <Icon className="h-4 w-4" style={{ color: kindColour(kind) }} aria-hidden="true" />}
                      <span className="line-clamp-2">{nameOf(pipeline)}</span>
                      <SortArrow column={pipeline} sort={sort} />
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {windowed && first > 0 && <tr style={{ height: first * ROW_HEIGHT }} aria-hidden="true" />}
            {rows}
            {windowed && last < videos.length && (
              <tr style={{ height: (videos.length - last) * ROW_HEIGHT }} aria-hidden="true" />
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ariaSort(sort: SortKey | null, column: string): 'ascending' | 'descending' | undefined {
  if (sort?.column !== column) return undefined;
  return sort.descending ? 'descending' : 'ascending';
}

function SortArrow({ column, sort }: { column: string; sort: SortKey | null }) {
  if (sort?.column !== column) return null;
  const Arrow = sort.descending ? ArrowDown : ArrowUp;
  return <Arrow className="h-3 w-3" aria-hidden="true" />;
}

function SortButton({
  column,
  sort,
  onSort,
  children,
}: {
  column: string;
  sort: SortKey | null;
  onSort: (column: string) => void;
  children: string;
}) {
  return (
    <button type="button" className="inline-flex items-center gap-1 hover:text-foreground" onClick={() => onSort(column)}>
      {children}
      <SortArrow column={column} sort={sort} />
    </button>
  );
}

/** One chip and "+2", on one line, the rest on hover (UAT #45). */
function ProjectChips({ projects }: { projects: VideoItem['projects'] }) {
  if (projects.length === 0) return null;
  const [first, ...rest] = projects;
  return (
    <div className="flex min-w-0 items-center gap-1" title={projects.map((p) => p.name).join(', ')}>
      <Link to={hrefOf({ project: first.id })} className="min-w-0">
        <Badge variant="secondary" className="block max-w-full truncate text-[10px] font-normal">
          {first.name}
        </Badge>
      </Link>
      {rest.length > 0 && <span className="shrink-0 text-xs text-muted-foreground">+{rest.length}</span>}
    </div>
  );
}
