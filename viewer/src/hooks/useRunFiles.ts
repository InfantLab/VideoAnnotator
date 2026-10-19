import { useQueries } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';

import { apiClient } from '@/api/client';
import { detectFileType, mergeAnnotationData } from '@/lib/parsers/merger';
import { readFiles, videoRows, type Row } from '@/lib/table/rows';
import type { RunPick } from '@/lib/videoRuns';
import type { StandardAnnotationData } from '@/types/annotations';
import type { VideoRun } from '@/types/video';

export interface RunFileRef {
  jobId: string;
  pipeline: string;
  name: string;
  url: string;
  /** Shown as a second lane only: not drawn on the video. */
  compare: boolean;
}

/** `${jobId}/${pipeline}` → the provenance record of that pipeline's result in that run. */
export type ProvenanceByRun = Record<string, Record<string, unknown> | null>;

export const runPipelineKey = (jobId: string, pipeline: string) => `${jobId}/${pipeline}`;

export interface RunFilesState {
  /** For the overlays: the chosen run's files of every pipeline, parsed. */
  parsed: StandardAnnotationData | null;
  /** For the lanes and the Data tab; `run` is the job id. */
  rows: Row[];
  files: RunFileRef[];
  provenance: ProvenanceByRun;
  loading: boolean;
  error: Error | null;
  warnings: string[];
}

/** The result and provenance files of each picked run and pipeline. */
export function filesToFetch(runs: VideoRun[] | undefined, picks: RunPick[]): RunFileRef[] {
  const files: RunFileRef[] = [];
  const seen = new Set<string>();
  for (const { pipeline, jobId, compare } of picks) {
    const key = runPipelineKey(jobId, pipeline);
    if (seen.has(key)) continue;
    seen.add(key);
    const p = runs?.find((r) => r.job_id === jobId)?.pipelines.find((x) => x.name === pipeline);
    for (const f of p?.files ?? []) {
      if (f.kind === 'result' || f.kind === 'provenance') {
        files.push({ jobId, pipeline, name: f.name, url: f.download_url, compare: !!compare });
      }
    }
  }
  return files;
}

interface Built {
  signature: string;
  parsed: StandardAnnotationData | null;
  rows: Row[];
  provenance: ProvenanceByRun;
  warnings: string[];
}

/**
 * Fetches only the files the shown lanes need (spec 028 R4, FR-013), each once
 * per session, and turns them into rows and overlay data. Nothing is written
 * to the browser's local library or a folder.
 */
export function useRunFiles(
  runs: VideoRun[] | undefined,
  picks: RunPick[],
  video: { name: string; duration: number },
): RunFilesState {
  const files = useMemo(() => filesToFetch(runs, picks), [runs, picks]);
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
  const signature = loaded ? files.map((f) => `${f.jobId}/${f.pipeline}/${f.name}/${f.compare}`).join('|') : null;

  const [built, setBuilt] = useState<Built | null>(null);

  useEffect(() => {
    if (signature === null) return;
    let cancelled = false;
    const fetched = queries.map((q) => q.data as File);
    (async () => {
      const texts = await Promise.all(fetched.map((f) => f.text()));
      const byRun = new Map<string, { jobId: string; list: { name: string; text: string }[] }>();
      files.forEach((f, i) => {
        // Grouped by job and pipeline: a compared pipeline's files read with their own run only.
        const key = runPipelineKey(f.jobId, f.pipeline);
        const entry = byRun.get(key) ?? { jobId: f.jobId, list: [] };
        entry.list.push({ name: f.name, text: texts[i] });
        byRun.set(key, entry);
      });
      const stem = video.name.replace(/\.[^.]+$/, '');
      const byJob = new Map<string, ReturnType<typeof readFiles>['read']>();
      const provenance: ProvenanceByRun = {};
      for (const [key, { jobId, list }] of byRun) {
        const { read } = readFiles(list);
        byJob.set(jobId, [...(byJob.get(jobId) ?? []), ...read]);
        provenance[key] = read[0]?.provenance ?? null;
      }
      const rows = [...byJob].flatMap(([jobId, read]) => videoRows(read, stem, jobId));
      rows.sort((a, b) => a.start - b.start);

      let parsed: StandardAnnotationData | null = null;
      let warnings: string[] = [];
      const shown = fetched.filter((_, i) => !files[i].compare);
      if (shown.length) {
        const detected = (await Promise.all(shown.map((f) => detectFileType(f)))).filter((d) => d.type !== 'unknown');
        const merged = await mergeAnnotationData(detected, undefined, {
          videoInfo: { filename: video.name, duration: video.duration, width: 0, height: 0 },
        });
        parsed = merged.data;
        warnings = merged.metadata.warnings;
      }
      if (!cancelled) setBuilt({ signature, parsed, rows, provenance, warnings });
    })().catch((e) => {
      if (!cancelled) {
        setBuilt({ signature, parsed: null, rows: [], provenance: {}, warnings: [String(e instanceof Error ? e.message : e)] });
      }
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
    provenance: current?.provenance ?? {},
    loading: files.length > 0 && !current && !error,
    error,
    warnings: current?.warnings ?? [],
  };
}
