import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';

import { apiClient } from '@/api/client';

/** Fetch a new token this long before the old one expires. */
export const REFRESH_BEFORE_MS = 5 * 60 * 1000;

interface Stream {
  url: string;
  expiresAt: number;
}

/**
 * The `<video>` src for a server path (spec 028 R3): 027's stream token, renewed
 * before it expires and whenever the stream answers 401 (expired, or the server
 * restarted with a new secret), keeping the time and play state across the swap.
 * Nothing is requested for a video that is no longer on the server.
 */
export function useVideoStream(path: string | null | undefined, available: boolean, videoRef: RefObject<HTMLVideoElement>) {
  const [stream, setStream] = useState<Stream | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const resumeAt = useRef<{ time: number; playing: boolean } | null>(null);
  const fetching = useRef(false);

  const fetchToken = useCallback(
    async (keepPlace: boolean) => {
      if (!path || !available || fetching.current) return;
      fetching.current = true;
      try {
        const video = videoRef.current;
        if (keepPlace && video) resumeAt.current = { time: video.currentTime, playing: !video.paused };
        const token = await apiClient.streamToken(path);
        setError(null);
        setStream({ url: apiClient.streamUrl(path, token.token), expiresAt: new Date(token.expires_at).getTime() });
      } catch (e) {
        setError(e instanceof Error ? e : new Error(String(e)));
      } finally {
        fetching.current = false;
      }
    },
    [path, available, videoRef],
  );

  useEffect(() => {
    setStream(null);
    resumeAt.current = null;
    void fetchToken(false);
  }, [fetchToken]);

  useEffect(() => {
    if (!stream) return;
    const wait = Math.max(0, stream.expiresAt - REFRESH_BEFORE_MS - Date.now());
    const timer = window.setTimeout(() => void fetchToken(true), wait);
    return () => window.clearTimeout(timer);
  }, [stream, fetchToken]);

  // After a swap the element starts again from 0: put it back where it was.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stream) return;
    const restore = () => {
      const at = resumeAt.current;
      if (!at) return;
      resumeAt.current = null;
      video.currentTime = at.time;
      if (at.playing) void video.play();
    };
    video.addEventListener('loadedmetadata', restore);
    return () => video.removeEventListener('loadedmetadata', restore);
  }, [stream, videoRef]);

  // A media error doesn't say why; ask the server whether the token is the problem.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stream) return;
    const onError = async () => {
      try {
        const probe = await fetch(stream.url, { headers: { Range: 'bytes=0-0' } });
        if (probe.status === 401) void fetchToken(true);
      } catch {
        // Offline or refused: the player's own message stands.
      }
    };
    video.addEventListener('error', onError);
    return () => video.removeEventListener('error', onError);
  }, [stream, videoRef, fetchToken]);

  return { url: stream?.url, error };
}
