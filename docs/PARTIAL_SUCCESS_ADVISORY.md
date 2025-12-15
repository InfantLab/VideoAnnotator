# Advisory: Partial Success Handling in Job Processing

**Date:** December 7, 2025
**Topic:** Change in Job Status behavior for multi-pipeline jobs.

## Summary
We have updated the `JobProcessor` logic to support "Partial Success". Previously, if a job requested multiple pipelines (e.g., `face_analysis` and `speaker_diarization`) and *any* single pipeline failed, the entire job was marked as `failed`, and no results were accessible.

Now, the system attempts to run all requested pipelines regardless of individual failures.

## New Behavior

| Scenario | Old Behavior | New Behavior |
| :--- | :--- | :--- |
| **All pipelines succeed** | Status: `completed` | Status: `completed` |
| **Some succeed, some fail** | Status: `failed` (No results returned) | Status: `completed` (Successful results returned) |
| **All pipelines fail** | Status: `failed` | Status: `failed` |

## Client Implementation Changes

Clients consuming the API need to be aware that `status: completed` no longer guarantees that *every* requested pipeline finished successfully.

### 1. Check for Warnings
If a job completes with partial failures, the `error_message` field on the job object will now contain a summary of the failures.

**Example Response (Partial Success):**
```json
{
  "id": "job_123",
  "status": "completed",
  "error_message": "Job completed with errors. Failed pipelines: speaker_diarization",
  "pipeline_results": {
    "face_analysis": { ... },  // Available!
    "speaker_diarization": null // Missing or error info
  }
}
```

### 2. Iterate Results
Always check the `pipeline_results` dictionary to confirm which specific pipelines generated output.

## Why this change?
This ensures that a fragile pipeline (like `speaker_diarization` which might fail on short audio clips) does not block access to robust results from other pipelines (like `face_analysis`) running on the same video.
