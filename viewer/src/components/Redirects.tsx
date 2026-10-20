import { useEffect } from 'react';
import { Navigate, useLocation, useNavigate, useParams } from 'react-router-dom';

import { apiClient } from '@/api/client';
import { notify } from '@/lib/notify';
import { userMessage } from '@/lib/userMessage';
import { hrefOf } from '@/lib/videosScope';

/** An old address that carried router state (the old wizard's): passed on. */
export function RedirectWithState({ to }: { to: string }) {
  const location = useLocation();
  return <Navigate to={to} state={location.state} replace />;
}

/** `/batches/:batchId` → the run as a filter of Videos. */
export function BatchRedirect() {
  const { batchId = '' } = useParams();
  return <Navigate to={hrefOf({ run: batchId })} replace />;
}

/** `/jobs/:jobId` → the Video page on that run (spec 029 R13). */
export function JobRedirect() {
  const { jobId = '' } = useParams();
  const navigate = useNavigate();
  useEffect(() => {
    let cancelled = false;
    apiClient
      .getJob(jobId)
      .then((job) => {
        if (cancelled) return;
        if (job.video_path) {
          const params = new URLSearchParams({ path: job.video_path, run: jobId });
          if (job.batch_id) params.set('from', `run:${job.batch_id}`);
          navigate(`/video?${params}`, { replace: true });
        }
        else navigate(hrefOf({ run: job.batch_id || jobId }), { replace: true });
      })
      .catch((error) => {
        if (cancelled) return;
        notify({ title: "That run isn't here any more", body: userMessage(error), action: { label: 'Videos', to: '/videos' } });
        navigate('/videos', { replace: true });
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, navigate]);
  return <p className="p-8 text-sm text-muted-foreground">Opening…</p>;
}
