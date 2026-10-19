/** The Video page (VideoAnnotator spec 028): shapes from the server's OpenAPI schema. */
import type { components } from '@/api/schema';

type Schemas = components['schemas'];

export type VideoRuns = Schemas['VideoRunsResponse'];
export type VideoRun = Schemas['RunInfo'];
export type VideoRunPipeline = Schemas['PipelineRunInfo'];
export type VideoRunFile = Schemas['RunFile'];
export type ProjectRef = Schemas['ProjectRef'];
export type VideoNote = Schemas['Note'];
export type VideoNoteCreate = Schemas['NoteCreate'];
export type VideoNoteUpdate = Schemas['NoteUpdate'];
