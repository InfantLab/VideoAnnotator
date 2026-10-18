// The pipeline guide (VideoAnnotator spec 026). Older servers send none of it,
// so everything falls back to the display name and description.
import type { PipelineDescriptor } from '@/types/pipelines';

export const pipelineSummary = (pipeline: PipelineDescriptor): string =>
  pipeline.summary ?? pipeline.description ?? 'No description provided.';
