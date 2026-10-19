import { useQueries } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';

import { apiClient } from '@/api/client';
import { detectFileType, mergeAnnotationData } from '@/lib/parsers/merger';
import { rowsFromFiles, type Row } from '@/lib/table/rows';
import type { RunSelection } from '@/lib/videoRuns';
import type { StandardAnnotationData } from '@/types/annotations';
import type { VideoRun } from '@/types/video';

export interface RunFileRef {
  jobId: string;
  pipeline: string;
  name: string;
  url: string;
}

export interface RunFilesState {
  /** For the overlays: the chosen run's files of every pipeline, parsed. */
  parsed: StandardAnnotationData | null;
  /** For the lanes and the Data tab; `run` is the job id. */
  rows: Row[];
  files: RunFileRef[];
  loading: boolean;
  error: Error | null;
  warnings: string[];
}

/** The result and provenance files of the chosen job for each pipeline. */
export function filesToFetch(runs: VideoRun[] | undefined, selection: RunSelection): RunFileRef[] {
  const files: RunFileRef[] = [];
  for (const [pipeline, jobId] of Object.entries(selection)) {
    const p = runs?.find((r) => r.job_id === jobId)?.pipelines.find((x) => x.name === pipeline);
    for (const f of p?.files ?? []) {
      if (f.kind === 'result' || f.kind === 'provenance') files.push({ jobId, pipeline, name: f.name, url: f.download_url });
    }
  }
  return files;
}

/**
 * Fetches only the files the shown lanes need (spec 028 R4, FR-013), each once
 * per session, and turns them into rows and overlay data. Nothing is written
 * to the browser's local library or a folder.
 */
export function useRunFiles(
  runs: VideoRun[] | undefined,
  selection: RunSelection,
  video: { name: string; duration: number },
): RunFilesState {
  const files = useMemo(() => filesToFetch(runs, selection), [runs, selection]);
  const queries = useQueries({
    queries: files.map((f) => ({
      queryKey: ['runFile', f.jobId, f.name],
      queryFn: () => apiClient.downloadResultFile(f.url, f.name),
      staleTime: Infinity,
      gcTime: Infinity,
      retry: 1,
    })),
  });
  const loaded = queries.every((q) => q.isSuccess);
  const error = (queries.find((q) => q.error)?.error as Error | undefined) ?? null;
  const signature = loaded ? files.map((f) => `${f.jobId}/${f.name}`).join('|') : null;

  const [built, setBuilt] = useState<{ signature: string; parsed: StandardAnnotationData | null; rows: Row[]; warnings: string[] } | null>(null);

  useEffect(() => {
    if (signature === null) return;
    let cancelled = false;
    const fetched = queries.map((q) => q.data as File);
    (async () => {
      const texts = await Promise.all(fetched.map((f) => f.text()));
      const byJob = new Map<string, { name: string; text: string }[]>();
      files.forEach((f, i) => {
        const list = byJob.get(f.jobId) ?? [];
        list.push({ name: f.name, text: texts[i] });
        byJob.set(f.jobId, list);
      });
      const stem = video.name.replace(/\.[^.]+$/, '');
      const rows = [...byJob].flatMap(([jobId, list]) => rowsFromFiles(list, stem, jobId));
      rows.sort((a, b) => a.start - b.start);

      let parsed: StandardAnnotationData | null = null;
      let warnings: string[] = [];
      if (fetched.length) {
        const detected = (await Promise.all(fetched.map((f) => detectFileType(f)))).filter((d) => d.type !== 'unknown');
        const merged = await mergeAnnotationData(detected, undefined, {
          videoInfo: { filename: video.name, duration: video.duration, width: 0, height: 0 },
        });
        parsed = merged.data;
        warnings = merged.metadata.warnings;
      }
      if (!cancelled) setBuilt({ signature, parsed, rows, warnings });
    })().catch((e) => {
      if (!cancelled) setBuilt({ signature, parsed: null, rows: [], warnings: [String(e instanceof Error ? e.message : e)] });
    });
    return () => {
      cancelled = true;
    };
    // `queries` changes identity every render; the signature says when its data did.
  }, [signature, video.name, video.duration]); // eslint-disable-line react-hooks/exhaustive-deps

  const current = built && built.signature === signature ? built : null;
  return {
    parsed: current?.parsed ?? null,
    rows: current?.rows ?? [],
    files,
    loading: files.length > 0 && !current && !error,
    error,
    warnings: current?.warnings ?? [],
  };
}
