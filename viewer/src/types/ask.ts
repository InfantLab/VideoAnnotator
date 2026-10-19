/** Ask AI (VideoAnnotator spec 027): shapes from the server's OpenAPI schema. */
import type { components } from '@/api/schema';

type Schemas = components['schemas'];

export type AskSampling = Schemas['Sampling'];
export type AskAnswer = Schemas['Answer'];
export type AskEntry = Schemas['Entry'];
export type AskLibraryPrompt = Schemas['LibraryPrompt'];
export type AskEntriesResponse = Schemas['EntriesResponse'];
export type AskTryRequest = Schemas['TryRequest'];
export type AskRunRequest = Schemas['RunRequest'];
export type AskDraftRequest = Schemas['DraftRequest'];
export type AskRunAnswers = Schemas['RunAnswers'];
export type AskFrames = Schemas['FramesResponse'];
export type UploadedVideo = Schemas['VideoUploadResponse'];
export type StreamToken = Schemas['StreamTokenResponse'];

export interface ModelEndpoint {
  baseUrl?: string;
  host: string;
  class: 'local' | 'remote';
  confirmed: boolean;
}

export const DEFAULT_SAMPLING: AskSampling = {
  interval: 5,
  start: 0,
  end: null,
  moments: null,
  frames: 1,
  spacing: null,
};
