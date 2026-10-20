import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import type { Preset, PresetCreateRequest, PresetUpdateRequest } from '@/types/presets';

export const WORKFLOWS_KEY = ['workflows'] as const;

/** Saved workflows (the API's presets, spec 029 R7). */
export function useWorkflows(enabled = true) {
  return useQuery({
    queryKey: WORKFLOWS_KEY,
    queryFn: async () => (await apiClient.listPresets()).presets,
    enabled,
    retry: false,
  });
}

export function useWorkflowActions() {
  const client = useQueryClient();
  const refresh = () => client.invalidateQueries({ queryKey: WORKFLOWS_KEY });
  return {
    create: useMutation({ mutationFn: (body: PresetCreateRequest) => apiClient.createPreset(body), onSuccess: refresh }),
    update: useMutation({
      mutationFn: ({ id, body }: { id: string; body: PresetUpdateRequest }) => apiClient.updatePreset(id, body),
      onSuccess: refresh,
    }),
    remove: useMutation({ mutationFn: (workflow: Preset) => apiClient.deletePreset(workflow.id), onSuccess: refresh }),
  };
}
