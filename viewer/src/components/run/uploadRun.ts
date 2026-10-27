/**
 * Videos from this computer: one upload per video, all under one run (spec
 * 008's batch id), as the old wizard did. Moved from `pages/NewJob.tsx`.
 */
import { apiClient } from '@/api/client';
import { APIError } from '@/api/handleError';
import { runSubject } from '@/lib/runName';
import { userMessage } from '@/lib/userMessage';

export interface UploadRunResult {
  runId: string;
  name: string;
  created: string[];
  failed: { name: string; reason: string }[];
}

export async function uploadRun(
  files: File[],
  selectedPipelines: string[],
  config: Record<string, unknown> | undefined,
  options: { name?: string; projectId?: string | null; onProgress?: (done: number) => void } = {},
): Promise<UploadRunResult> {
  const runId = crypto.randomUUID();
  const name = options.name?.trim() || runSubject(null, null, files);
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
