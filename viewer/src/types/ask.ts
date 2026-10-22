/** Ask AI (VideoAnnotator specs 027, 033): shapes from the server's OpenAPI schema. */
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
export type ModelServer = Schemas['ModelServerOut'];
export type ModelServersResponse = Schemas['ModelServersResponse'];
export type ModelServerAdd = Schemas['ModelServerAdd'];
export type ModelServerPatch = Schemas['ModelServerPatch'];
export type ModelServerCheck = Schemas['ModelServerCheck'];
export type ModelServerCheckRequest = Schemas['ModelServerCheckRequest'];
export type ModelServerSettings = Schemas['ModelServerSettings'];
export type SuggestShapeResponse = Schemas['SuggestShapeResponse'];
export type ModelServerKind = 'ollama' | 'openai_compatible';

export const DEFAULT_SERVER_ID = 'default';

/** One field of an answer shape (spec 033 data-model.md). */
export interface AnswerShapeField {
  name: string;
  kind: 'choice' | 'yes_no' | 'number' | 'text';
  values?: string[];
  min?: number | null;
  max?: number | null;
}

export type AnswerShape = AnswerShapeField[];

export interface ModelEndpoint {
  baseUrl?: string;
  host: string;
  class: 'local' | 'remote';
  confirmed: boolean;
  /** Frames may go there now (spec 033): local, or confirmed and allowed. */
  sendable?: boolean;
  /** Confirmed before remote servers needed switching on. */
  beforeSwitch?: boolean;
}

export const DEFAULT_SAMPLING: AskSampling = {
  interval: 5,
  start: 0,
  end: null,
  moments: null,
  frames: 1,
  spacing: null,
};
