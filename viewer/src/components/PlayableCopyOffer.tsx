import { useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, Wand2 } from 'lucide-react';

import { apiClient } from '@/api/client';
import { Button } from '@/components/ui/button';
import { userMessage } from '@/lib/userMessage';

const POLL_MS = 2_000;

/**
 * For a video this browser can't show (viewer design D7, UAT #54): make a copy
 * it can play, on the server, and switch to it once it's there. A copy made
 * before is used straight away.
 */
export function PlayableCopyOffer({ path, onReady }: { path: string; onReady: () => void }) {
  const queryClient = useQueryClient();
  const key = ['playableCopy', path];
  const copy = useQuery({
    queryKey: key,
    queryFn: () => apiClient.getPlayableCopy(path),
    refetchInterval: (q) => (q.state.data?.state === 'making' ? POLL_MS : false),
    retry: false,
  });
  const make = useMutation({
    mutationFn: () => apiClient.makePlayableCopy(path),
    onSuccess: (data) => queryClient.setQueryData(key, data),
  });
  const state = copy.data?.state;

  useEffect(() => {
    if (state === 'ready') onReady();
  }, [state]); // eslint-disable-line react-hooks/exhaustive-deps

  if (copy.isLoading) return null;
  if (state === 'unavailable') {
    return (
      <div className="space-y-1 text-muted-foreground" data-testid="playable-copy">
        <p>{copy.data?.error} To see the video, re-encode it to H.264, e.g.</p>
        <code className="block break-all rounded bg-muted px-2 py-1 font-mono text-xs">
          ffmpeg -i in.mp4 -c:v libx264 -crf 18 -c:a copy out.mp4
        </code>
      </div>
    );
  }
  if (state === 'making' || make.isPending) {
    return (
      <p className="flex items-center gap-2 text-muted-foreground" data-testid="playable-copy">
        <Loader2 className="h-4 w-4 animate-spin" /> Making a copy this browser can play: seconds for a short clip, a few minutes
        for a long video. It plays here when it&apos;s ready.
      </p>
    );
  }
  if (state === 'ready') return null;
  return (
    <div className="space-y-1" data-testid="playable-copy">
      <Button size="sm" onClick={() => make.mutate()}>
        <Wand2 className="mr-1 h-4 w-4" /> Make a copy this browser can play
      </Button>
      <p className="text-xs text-muted-foreground">
        The server keeps an H.264 copy for watching here; results still come from the original.
      </p>
      {state === 'failed' && <p className="text-xs text-destructive">{copy.data?.error}</p>}
      {(make.error || copy.error) && <p className="text-xs text-destructive">{userMessage(make.error ?? copy.error)}</p>}
    </div>
  );
}
