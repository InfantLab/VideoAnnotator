import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { GitCompare } from 'lucide-react';
import { apiClient, type JobResponse } from '@/api/client';
import { Button } from '@/components/ui/button';

type JobWithVideo = JobResponse & { video_filename?: string; video_size_bytes?: number; rerun_of?: string | null };

const isVlm = (job: JobResponse) => (job.selected_pipelines ?? []).includes('vlm_annotation');

/**
 * "Compare with…" on a VLM job's page (spec 021): other finished VLM runs of
 * the same video, its reruns and original first.
 */
export const CompareWith = ({ job }: { job: JobWithVideo }) => {
  const navigate = useNavigate();
  const [other, setOther] = useState('');
  const { data } = useQuery({ queryKey: ['jobs', 'compare-candidates'], queryFn: () => apiClient.getJobs(1, 100), retry: false });
  if (!isVlm(job) || job.status !== 'completed') return null;

  const related = new Set([job.rerun_of, ...((job as { reruns?: string[] }).reruns ?? [])].filter(Boolean));
  const candidates = ((data?.jobs ?? []) as JobWithVideo[])
    .filter(
      (j) =>
        j.id !== job.id &&
        j.status === 'completed' &&
        isVlm(j) &&
        j.video_filename === job.video_filename &&
        j.video_size_bytes === job.video_size_bytes,
    )
    .sort((x, y) => Number(related.has(y.id)) - Number(related.has(x.id)) || (y.created_at ?? '').localeCompare(x.created_at ?? ''));

  const compare = (b: string) => navigate(`/compare?a=${encodeURIComponent(job.id)}&b=${encodeURIComponent(b)}`);

  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      {job.rerun_of && (
        <Button size="sm" variant="outline" onClick={() => navigate(`/compare?a=${encodeURIComponent(job.rerun_of!)}&b=${encodeURIComponent(job.id)}`)}>
          <GitCompare className="h-4 w-4 mr-2" /> Compare with original
        </Button>
      )}
      {candidates.length > 0 ? (
        <>
          <select
            aria-label="VLM run to compare with"
            className="h-8 rounded-md border border-input bg-background px-2 text-sm"
            value={other}
            onChange={(e) => setOther(e.target.value)}
          >
            <option value="">Compare with…</option>
            {candidates.map((c) => (
              <option key={c.id} value={c.id}>
                {related.has(c.id) ? (c.id === job.rerun_of ? 'Original · ' : 'Rerun · ') : ''}
                job {c.id.slice(0, 8)} · {new Date(c.created_at ?? '').toLocaleString()}
              </option>
            ))}
          </select>
          <Button size="sm" variant="outline" disabled={!other} onClick={() => compare(other)}>
            Compare
          </Button>
        </>
      ) : (
        !job.rerun_of && <span className="text-muted-foreground">No other VLM run of this video to compare with yet.</span>
      )}
    </div>
  );
};
