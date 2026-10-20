import { describe, expect, it } from 'vitest';

import { APIError } from '@/api/handleError';
import { userMessage } from '@/lib/userMessage';

const failure = (code: string, message = 'whatever the server said') =>
  new APIError(message, 404, undefined, { error: { code, message } });

describe('userMessage', () => {
  it.each([
    ['JOB_NOT_FOUND', "That run isn't on this server any more."],
    ['BATCH_NOT_FOUND', "That run isn't on this server any more."],
    ['DATASET_NOT_FOUND', "That project isn't on this server any more."],
    ['NOT_DATASET_OWNER', 'Only the person who made this project, or an administrator, can remove it.'],
    ['NOTHING_TO_RUN', expect.stringContaining('None of these videos can be run')],
    ['PATH_NOT_ALLOWED', "That folder isn't shared with VideoAnnotator."],
  ])('words %s in the viewer\'s terms', (code, expected) => {
    expect(userMessage(failure(code))).toEqual(expected);
  });

  it('keeps a server message that uses none of the API words', () => {
    expect(userMessage(failure('SOMETHING_NEW', 'The disk is full.'))).toBe('The disk is full.');
  });

  it('hides a message that names a job, batch, dataset or preset', () => {
    expect(userMessage(failure('SOMETHING_NEW', 'Job abc failed to start'))).toBe(
      "The server couldn't do that (SOMETHING_NEW).",
    );
    expect(userMessage(new Error('Dataset not found'))).toBe("The server couldn't do that.");
  });

  it('says when the server cannot be reached', () => {
    expect(userMessage(new APIError('Network error: Failed to fetch', 0))).toMatch(/Can't reach the server/);
  });

  it('never says an API word for any code it knows', () => {
    for (const code of ['JOB_NOT_FOUND', 'DATASET_NAME_CONFLICT', 'PRESET_NAME_CONFLICT', 'DATASET_NOT_STORED']) {
      expect(userMessage(failure(code))).not.toMatch(/\b(jobs?|batch(es)?|datasets?|presets?)\b/i);
    }
  });
});
