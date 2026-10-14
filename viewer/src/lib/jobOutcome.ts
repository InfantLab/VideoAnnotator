/**
 * How a finished job went, beyond its status.
 *
 * A job the server finishes can still have pipelines that produced nothing (a
 * pipeline it couldn't import, a model that failed to load). VideoAnnotator
 * reports that as `status: "completed"` with an `error_message` like
 * "Completed with errors. Failed pipelines: person_tracking", and puts each
 * pipeline's reason in GET /api/v1/jobs/{id}/results.
 */

/** GET /api/v1/jobs/{id}/results, the parts the viewer reads. */
export interface PipelineResult {
  pipeline_name?: string;
  status: string;
  error_message?: string | null;
}

export interface JobResults {
  job_id: string;
  status: string;
  pipeline_results: Record<string, PipelineResult>;
}

export function isCompletedWithErrors(job: { status: string; error_message?: string | null }): boolean {
  return job.status === 'completed' && !!job.error_message?.trim();
}

/** pipeline name -> why it failed, for the pipelines of a job that didn't produce results. */
export function failedPipelinesOf(results: JobResults | null | undefined): Record<string, string> {
  const failed: Record<string, string> = {};
  Object.entries(results?.pipeline_results ?? {}).forEach(([name, result]) => {
    if (result?.status === 'failed') {
      failed[name] = result.error_message?.trim() || 'the server reported it as failed, without a reason';
    }
  });
  return failed;
}

const SERVER_OUTCOME_PREFIX = /^(Completed with errors|All pipelines failed)\.\s*/;

/**
 * Why a failed or partly failed job went wrong, short enough for a table row:
 * the job's error_message without the server's outcome prefix (the status
 * badge already says that), i.e. "pipeline: reason; pipeline: reason".
 */
export function jobErrorSummary(job: { status: string; error_message?: string | null }): string | null {
  if (job.status !== 'failed' && !isCompletedWithErrors(job)) return null;
  const message = job.error_message?.trim();
  if (!message) return null;
  return message.replace(SERVER_OUTCOME_PREFIX, '') || message;
}
