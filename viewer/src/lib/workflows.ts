/** Workflows (the API's presets, spec 029 R7). */
import { apiClient } from '@/api/client';
import type { Preset, PresetCreateRequest } from '@/types/presets';

/** A project's settings set from a workflow: a copy, so later edits to it change nothing. */
export function applyBody(workflow: Preset) {
  return {
    current_settings: {
      selected_pipelines: workflow.selected_pipelines,
      config: workflow.config,
      source: { kind: 'workflow' as const, id: workflow.id, name: workflow.name },
    },
  };
}

/** The settings a run ran with: its first video's. */
export async function runSettingsOf(jobId: string) {
  const job = await apiClient.getJob(jobId);
  return { selected: job.selected_pipelines ?? [], config: (job.config ?? {}) as Record<string, unknown> };
}

/** What a project's next run starts from. */
export async function projectSettingsOf(projectId: string) {
  const found = await apiClient.getRunSettings({ project: projectId });
  return { selected: found.selected_pipelines, config: found.config };
}

/** The fields a workflow file carries, so an exported one imports as it was. */
export function workflowFile(workflow: Preset): PresetCreateRequest {
  return {
    name: workflow.name,
    description: workflow.description ?? undefined,
    selected_pipelines: workflow.selected_pipelines,
    config: workflow.config,
    export_choices: workflow.export_choices ?? null,
    source: workflow.source ?? null,
  };
}
