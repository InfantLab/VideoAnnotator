/**
 * Server errors in the viewer's words (spec 029 R15, FR-019).
 *
 * The API keeps its names (job, batch, dataset, preset); the viewer says run,
 * project and workflow. Known error codes get a sentence of their own; any
 * other message is shown as the server wrote it only when it uses none of the
 * API's words, else a neutral sentence with the code. The debug panel keeps
 * the raw message.
 */
import { APIError, apiErrorEnvelope } from '@/api/handleError';

const MESSAGES: Record<string, string> = {
  JOB_NOT_FOUND: "That run isn't on this server any more.",
  BATCH_NOT_FOUND: "That run isn't on this server any more.",
  RUN_NOT_FOUND: "That run isn't on this server any more.",
  DATASET_NOT_FOUND: "That project isn't on this server any more.",
  NOT_DATASET_OWNER: 'Only the person who made this project, or an administrator, can remove it.',
  NOT_PRESET_OWNER: 'Only the person who saved this workflow, or an administrator, can change or remove it.',
  PRESET_NOT_FOUND: "That workflow isn't on this server any more.",
  DATASET_NAME_CONFLICT: 'You already have a project with that name. Choose another.',
  PRESET_NAME_CONFLICT: 'You already have a workflow with that name. Choose another.',
  DATASET_NOT_STORED: "The server no longer has these videos. Choose the folder they're in, or upload them again.",
  NOTHING_TO_EXPORT: 'There are no results to export here yet.',
  NOTHING_TO_RUN: "None of these videos can be run: they may have been moved, or aren't in a shared folder.",
  PATH_NOT_ALLOWED: "That folder isn't shared with VideoAnnotator.",
  VIDEO_NOT_FOUND: 'The video file is missing: it may have been moved or deleted.',
  RERUN_VIDEO_MISSING: "The video is no longer where it was, so it can't be run again.",
  JOB_NOT_FINISHED: "It's still running. Wait for it to finish, or stop it first.",
  OWNER_IDENTITY_REQUIRED: 'Sign in with your API key (Settings) to save this.',
  PARQUET_NOT_INSTALLED: 'Parquet needs an optional component on the server; CSV always works.',
};

const API_WORDS = /\b(jobs?|batch(es)?|datasets?|presets?)\b/i;

export function errorCode(error: unknown): string | undefined {
  return apiErrorEnvelope(error).code;
}

export function userMessage(error: unknown): string {
  const code = errorCode(error);
  if (code && MESSAGES[code]) return MESSAGES[code];
  if (error instanceof APIError && error.status === 0) {
    return "Can't reach the server. Check it's running, and the address in Settings.";
  }
  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : '';
  if (message && !API_WORDS.test(message)) return message;
  return code ? `The server couldn't do that (${code}).` : "The server couldn't do that.";
}
