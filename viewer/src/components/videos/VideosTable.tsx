import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { kindColour, kindOfPipeline } from '@/lib/resultKinds';
import type { ListContext } from '@/lib/listNavigation';
import { folderLabel, hrefOf, shortFolder, videoLink } from '@/lib/videosScope';
import type { VideoItem, VideoResult } from '@/types/videos';

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

function ResultMark({ title, result }: { title: string; result: VideoResult | undefined }) {
  if (!result) {
    return (
      <span className="text-muted-foreground" title={`${title}: not run`} aria-label={`${title}: not run`}>
        –
      </span>
    );
  }
  const look = MARKS[result.state] ?? MARKS.waiting;
  const label = `${title}: ${look.word}${result.error ? `: ${result.error}` : ''}`;
  return (
    <span className={look.className} title={label} aria-label={label}>
      {look.mark}
    </span>
  );
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
}

/**
 * One row per video: its folder, projects and, per pipeline, where its newest
 * result stands (spec 029 FR-002). `/` focuses the filter; `Enter` opens the
 * focused row and `Space` selects it.
 */
export function VideosTable({ videos, pipelines, nameOf, taskOf = () => '', selected, onSelect, context, filter, onFilter }: VideosTableProps) {
  // An empty Projects column only takes room (UAT #5).
  const anyProjects = videos.some((v) => v.projects.length > 0);
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

  const rows = videos.slice(first, last).map((video) => (
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
      <td className="max-w-[18rem] px-2">
        <div className="flex items-center gap-2">
          <Link
            to={videoLink(video, context)}
            className="truncate font-medium hover:underline"
            title={video.path}
            onClick={(event) => event.stopPropagation()}
          >
            {video.name}
          </Link>
          {!video.available && (
            <Badge variant="outline" className="shrink-0 text-[10px] text-muted-foreground" title="The file has moved or been deleted; its results are kept">
              not on the server
            </Badge>
          )}
        </div>
      </td>
      <td className="max-w-[14rem] truncate px-2 text-muted-foreground" title={folderLabel(video.folder)}>
        {shortFolder(video.folder)}
      </td>
      {anyProjects && <td className="px-2" onClick={(event) => event.stopPropagation()}>
        <div className="flex flex-wrap gap-1">
          {video.projects.map((project) => (
            <Link key={project.id} to={hrefOf({ project: project.id })}>
              <Badge variant="secondary" className="text-[10px] font-normal">
                {project.name}
              </Badge>
            </Link>
          ))}
        </div>
      </td>}
      {pipelines.map((pipeline) => (
        <td key={pipeline} className="px-2 text-center">
          <ResultMark title={nameOf(pipeline)} result={video.results[pipeline]} />
        </td>
      ))}
    </tr>
  ));

  return (
    <div className="space-y-2">
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
              <th className="px-2 py-2 font-medium">Video</th>
              <th className="px-2 py-2 font-medium">Folder</th>
              {anyProjects && <th className="px-2 py-2 font-medium">Projects</th>}
              {pipelines.map((pipeline) => {
                const kind = kindOfPipeline(pipeline);
                const Icon = kind?.icon;
                const task = taskOf(pipeline);
                return (
                  <th
                    key={pipeline}
                    className="max-w-[7.5rem] px-2 py-2 text-center align-bottom font-medium leading-tight"
                    title={task ? `${nameOf(pipeline)} · ${task}` : nameOf(pipeline)}
                  >
                    <span className="flex flex-col items-center gap-1">
                      {Icon && <Icon className="h-4 w-4" style={{ color: kindColour(kind) }} aria-hidden="true" />}
                      <span className="line-clamp-2">{nameOf(pipeline)}</span>
                    </span>
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
