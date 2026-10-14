import { useQuery } from '@tanstack/react-query';
import { Bookmark, History } from 'lucide-react';
import { apiClient, type JobResponse } from '@/api/client';
import { Button } from '@/components/ui/button';
import type { Preset } from '@/types/presets';

interface StartFromRecentProps {
  onUseJob: (job: JobResponse, label: string) => void;
  onApplyPreset: (preset: Preset) => void;
}

const labelOf = (job: JobResponse) =>
  ((job as JobResponse & Record<string, unknown>).video_filename as string | undefined) ?? `job ${job.id.slice(0, 8)}`;

/**
 * The wizard's first step offers what was used before the blank form (spec
 * 019): the latest finished jobs' settings and the most recently used presets.
 * Shows nothing on a new server.
 */
export const StartFromRecent = ({ onUseJob, onApplyPreset }: StartFromRecentProps) => {
  const { data: jobsData } = useQuery({ queryKey: ['jobs', 'recent-finished'], queryFn: () => apiClient.getJobs(1, 30), retry: false });
  const { data: presetData } = useQuery({ queryKey: ['presets'], queryFn: () => apiClient.listPresets(), retry: false });

  const jobs = (jobsData?.jobs ?? [])
    .filter((job) => job.status === 'completed' && (job.selected_pipelines?.length ?? 0) > 0)
    .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
    .slice(0, 5);
  const presets = [...(presetData?.presets ?? [])]
    .sort((a, b) => (b.last_used_at ?? b.created_at).localeCompare(a.last_used_at ?? a.created_at))
    .slice(0, 5);
  if (jobs.length === 0 && presets.length === 0) return null;

  return (
    <div className="rounded-md border p-3 space-y-2 text-sm">
      <p className="font-medium">Start from what you used before</p>
      {jobs.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <History className="h-4 w-4 text-muted-foreground" aria-hidden />
          {jobs.map((job) => (
            <Button
              key={job.id}
              size="sm"
              variant="outline"
              className="h-7"
              title={`Use the pipelines and settings of ${labelOf(job)}: ${(job.selected_pipelines ?? []).join(', ')}`}
              onClick={() => onUseJob(job, labelOf(job))}
            >
              {labelOf(job)}
            </Button>
          ))}
        </div>
      )}
      {presets.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <Bookmark className="h-4 w-4 text-muted-foreground" aria-hidden />
          {presets.map((preset) => (
            <Button
              key={preset.id}
              size="sm"
              variant="outline"
              className="h-7"
              title={`Apply preset: ${preset.selected_pipelines.join(', ')}`}
              onClick={() => onApplyPreset(preset)}
            >
              {preset.name}
            </Button>
          ))}
        </div>
      )}
    </div>
  );
};
