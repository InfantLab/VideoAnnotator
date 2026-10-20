import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import { notify } from '@/lib/notify';
import { videosWord } from '@/lib/runs';
import { userMessage } from '@/lib/userMessage';
import { hrefOf } from '@/lib/videosScope';
import type { DatasetUpdateRequest, SavedDataset } from '@/types/datasets';

/** One project (the API's saved dataset), with its settings and owner. */
export function useProject(id: string | null | undefined) {
  return useQuery({
    queryKey: ['videos', 'project', id],
    queryFn: () => apiClient.getDataset(id!),
    enabled: !!id,
    retry: false,
  });
}

const failed = (what: string) => (error: unknown) => notify({ title: `Couldn't ${what}`, body: userMessage(error), tone: 'error' });

/** Make, fill, rename and remove projects (spec 029 US2), each saying what it did. */
export function useProjects() {
  const client = useQueryClient();
  const refresh = () => client.invalidateQueries({ queryKey: ['videos'] });

  const create = useMutation({
    mutationFn: async ({ name, paths = [] }: { name: string; paths?: string[] }) => {
      const project = await apiClient.createDataset({ name, video_manifest: [], server_selection: true });
      return paths.length ? apiClient.addProjectVideos(project.id, paths) : project;
    },
    onSuccess: (project, { name, paths = [] }) => {
      refresh();
      notify({
        title: `Project ${name} made`,
        body: paths.length ? `${videosWord(paths.length)} in it.` : 'Add videos by selecting them anywhere.',
        action: { label: 'Show it', to: hrefOf({ project: project.id }) },
        tone: 'success',
      });
    },
    onError: failed('make the project'),
  });

  const add = useMutation({
    mutationFn: ({ project, paths }: { project: { id: string; name: string }; paths: string[] }) =>
      apiClient.addProjectVideos(project.id, paths),
    onSuccess: (_, { project, paths }) => {
      refresh();
      notify({ title: `${videosWord(paths.length)} added to ${project.name}`, action: { label: 'Show it', to: hrefOf({ project: project.id }) } });
    },
    onError: failed('add the videos'),
  });

  const remove = useMutation({
    mutationFn: ({ project, paths }: { project: { id: string; name: string }; paths: string[] }) =>
      apiClient.removeProjectVideos(project.id, paths),
    onSuccess: (_, { project, paths }) => {
      refresh();
      notify({ title: `${videosWord(paths.length)} taken out of ${project.name}`, body: 'The videos and their results stay where they are.' });
    },
    onError: failed('take the videos out'),
  });

  const update = useMutation({
    mutationFn: ({ id, body }: { id: string; body: DatasetUpdateRequest }) => apiClient.updateDataset(id, body),
    onSuccess: refresh,
    onError: failed('save the project'),
  });

  const destroy = useMutation({
    mutationFn: (project: SavedDataset | { id: string; name: string }) => apiClient.deleteDataset(project.id),
    onSuccess: (_, project) => {
      refresh();
      notify({ title: `Project ${project.name} removed`, body: 'Its videos and their results are still here.', action: { label: 'All videos', to: '/videos' } });
    },
    onError: failed('remove the project'),
  });

  return { create, add, remove, update, destroy };
}
