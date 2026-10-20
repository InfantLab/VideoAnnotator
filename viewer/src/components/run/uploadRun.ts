/**
 * Videos from this computer: one upload per video, all under one run (spec
 * 008's batch id), as the old wizard did. Moved from `pages/NewJob.tsx`.
 */
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { userMessage } from '@/lib/userMessage';

export interface UploadRunResult {
  runId: string;
  name: string;
  created: string[];
  failed: { name: string; reason: string }[];
}

/** A name for a run when the user doesn't type one: the folder picked, else the selection. */
export function defaultRunName(files: File[]): string {
  const folder = (files[0] as File & { webkitRelativePath?: string })?.webkitRelativePath?.split('/')[0];
  if (folder) return folder;
  const stamp = new Date().toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
  return files.length === 1 ? `${files[0]?.name ?? '1 video'} — ${stamp}` : `${files.length} videos — ${stamp}`;
}

export async function uploadRun(
  files: File[],
  selectedPipelines: string[],
  config: Record<string, unknown> | undefined,
  options: { name?: string; projectId?: string | null; onProgress?: (done: number) => void } = {},
): Promise<UploadRunResult> {
  const runId = crypto.randomUUID();
  const name = options.name?.trim() || defaultRunName(files);
  const created: string[] = [];
  const failed: UploadRunResult['failed'] = [];
  for (const file of files) {
    try {
      const job = await apiClient.submitJob(file, selectedPipelines, config, {
        id: runId,
        name,
        datasetId: options.projectId ?? undefined,
      });
      created.push(job.id);
    } catch (error) {
      failed.push({ name: file.name, reason: error instanceof APIError ? userMessage(error) : String(error) });
    }
    options.onProgress?.(created.length + failed.length);
  }
  return { runId, name, created, failed };
}
