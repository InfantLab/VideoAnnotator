/**
 * Where a run's results open (spec 028): the Video page for its video, starting
 * on this run, in the list of its batch when it has one.
 */
export interface JobLike {
  id: string;
  video_path?: string | null;
  batch_id?: string | null;
}

export function videoPageLink(job: JobLike): string {
  // Without a recorded path, the old address explains what it can.
  if (!job.video_path) return `/view/${encodeURIComponent(job.id)}`;
  const params = new URLSearchParams({ path: job.video_path, run: job.id });
  if (job.batch_id) params.set('from', `run:${job.batch_id}`);
  return `/video?${params}`;
}
