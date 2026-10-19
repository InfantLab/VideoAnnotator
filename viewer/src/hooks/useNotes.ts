import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/api/client';
import type { VideoNote } from '@/types/video';

export interface NoteDraft {
  start: number;
  end: number | null;
  text: string;
}

export interface NotesApi {
  notes: VideoNote[];
  error: Error | null;
  create: (draft: NoteDraft) => Promise<void>;
  update: (id: string, change: Partial<NoteDraft>) => Promise<void>;
  remove: (id: string) => Promise<void>;
}

const byTime = (a: VideoNote, b: VideoNote) => a.start_sec - b.start_sec || a.created_at.localeCompare(b.created_at);

/**
 * A video's notes on the server (spec 028 US6). Changes show at once and are
 * undone if the server refuses them.
 */
export function useNotes(path: string | null | undefined): NotesApi {
  const client = useQueryClient();
  const key = ['videoNotes', path];
  const query = useQuery({
    queryKey: key,
    queryFn: () => apiClient.listNotes(path!),
    enabled: !!path,
    retry: false,
  });

  const optimistic = <T,>(apply: (notes: VideoNote[], input: T) => VideoNote[]) => ({
    onMutate: async (input: T) => {
      await client.cancelQueries({ queryKey: key });
      const before = client.getQueryData<VideoNote[]>(key) ?? [];
      client.setQueryData<VideoNote[]>(key, apply(before, input).sort(byTime));
      return { before };
    },
    onError: (_e: unknown, _input: T, context?: { before: VideoNote[] }) => {
      if (context) client.setQueryData(key, context.before);
    },
    onSettled: () => client.invalidateQueries({ queryKey: key }),
  });

  const create = useMutation({
    mutationFn: (draft: NoteDraft) =>
      apiClient.createNote({ path: path!, start_sec: draft.start, end_sec: draft.end, text: draft.text }),
    ...optimistic<NoteDraft>((notes, draft) => [
      ...notes,
      {
        id: `pending-${Date.now()}`,
        video_path: path ?? '',
        start_sec: draft.end !== null ? Math.min(draft.start, draft.end) : draft.start,
        end_sec: draft.end !== null ? Math.max(draft.start, draft.end) : null,
        text: draft.text.trim(),
        author_id: '',
        author_name: 'you',
        created_at: new Date().toISOString(),
        updated_at: null,
        updated_by: null,
      },
    ]),
  });

  const update = useMutation({
    mutationFn: ({ id, change }: { id: string; change: Partial<NoteDraft> }) =>
      apiClient.updateNote(id, {
        ...(change.text !== undefined ? { text: change.text } : {}),
        ...(change.start !== undefined ? { start_sec: change.start } : {}),
        ...(change.end !== undefined ? { end_sec: change.end } : {}),
      }),
    ...optimistic<{ id: string; change: Partial<NoteDraft> }>((notes, { id, change }) =>
      notes.map((n) =>
        n.id === id
          ? {
              ...n,
              text: change.text ?? n.text,
              start_sec: change.start ?? n.start_sec,
              end_sec: change.end !== undefined ? change.end : n.end_sec,
            }
          : n,
      ),
    ),
  });

  const remove = useMutation({
    mutationFn: (id: string) => apiClient.deleteNote(id),
    ...optimistic<string>((notes, id) => notes.filter((n) => n.id !== id)),
  });

  return {
    notes: query.data ?? [],
    error: (query.error as Error | null) ?? (create.error as Error | null) ?? (update.error as Error | null) ?? (remove.error as Error | null),
    create: async (draft) => {
      await create.mutateAsync(draft);
    },
    update: async (id, change) => {
      await update.mutateAsync({ id, change });
    },
    remove: async (id) => {
      await remove.mutateAsync(id);
    },
  };
}
