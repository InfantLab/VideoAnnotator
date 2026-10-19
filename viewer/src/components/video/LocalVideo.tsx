import { useEffect, useRef, useState } from 'react';

import type { LoadedFiles } from '@/components/FileUploader';
import { VideoView } from '@/components/video/VideoView';
import { readFiles, videoRows, type Row } from '@/lib/table/rows';
import type { ProvenanceByRun } from '@/hooks/useRunFiles';

/**
 * Standalone `/viewer` on the Video page (spec 028 T034): the files a researcher
 * opened from their computer, with no server: no runs menu, no context, and
 * notes say they need a server.
 */
export function LocalVideo({ loaded, onClose }: { loaded: LoadedFiles; onClose: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [rows, setRows] = useState<Row[] | null>(null);
  const [provenance, setProvenance] = useState<ProvenanceByRun>({});
  const [error, setError] = useState<Error | null>(null);
  const name = loaded.video?.name ?? loaded.data.video_info.filename ?? 'Local files';

  useEffect(() => {
    let cancelled = false;
    Promise.all(loaded.results.map(async (f) => ({ name: f.name, text: await f.text() })))
      .then((texts) => {
        if (cancelled) return;
        const { read } = readFiles(texts);
        setProvenance(Object.fromEntries(read.map((r) => [`/${r.pipeline}`, r.provenance])));
        setRows(videoRows(read, name.replace(/\.[^.]+$/, '')).sort((a, b) => a.start - b.start));
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e : new Error(String(e))));
    return () => {
      cancelled = true;
    };
  }, [loaded, name]);

  return (
    <VideoView
      videoRef={videoRef}
      onBack={onClose}
      source={{
        name,
        path: null,
        videoFile: loaded.video,
        missingVideoMessage: loaded.video ? undefined : 'No video was opened with these files.',
        durationHint: loaded.data.video_info.duration || null,
        rows: rows ?? [],
        parsed: loaded.data,
        runs: [],
        failed: [],
        loading: rows === null && !error,
        error,
        local: true,
        provenance,
      }}
    />
  );
}
