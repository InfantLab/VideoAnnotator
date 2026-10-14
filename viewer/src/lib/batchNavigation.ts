// Where a video sits in its batch, for the results viewer's previous/next.

interface BatchMember {
  id: string;
  status: string;
  video_filename?: string | null;
}

export interface BatchPosition<T extends BatchMember> {
  /** 1-based position of this video among all of the batch's videos. */
  position: number;
  total: number;
  /** Nearest videos before and after with results to open; failed or unfinished ones are skipped. */
  previous: T | null;
  next: T | null;
}

const hasResults = (job: BatchMember) => job.status === 'completed';

export function batchPosition<T extends BatchMember>(jobs: readonly T[], jobId: string): BatchPosition<T> | null {
  const index = jobs.findIndex((job) => job.id === jobId);
  if (index < 0) return null;
  const before = jobs.slice(0, index).reverse().find(hasResults) ?? null;
  const after = jobs.slice(index + 1).find(hasResults) ?? null;
  return { position: index + 1, total: jobs.length, previous: before, next: after };
}
