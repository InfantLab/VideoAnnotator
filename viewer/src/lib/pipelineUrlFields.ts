import type { PipelineDescriptor, PipelineParameterSchema } from '@/types/pipelines';

/**
 * A URL-valued string field (e.g. vlm_annotation's `base_url`). The generic
 * single-line input strips newlines, so a prompt pasted into it by mistake
 * turns into a plausible-looking one-liner the server only rejects at
 * submission (400 INVALID_URL). Same rule as the server's `_base_url_error`.
 */
export const isUrlParameter = (parameter: PipelineParameterSchema) =>
  parameter.type === 'string' && /(^|_)url$/.test(parameter.name);

export const urlValueError = (value: unknown): string | null => {
  if (value === undefined || value === null || value === '') return null;
  const text = String(value);
  if (!/\s/.test(text)) {
    try {
      const url = new URL(text);
      if ((url.protocol === 'http:' || url.protocol === 'https:') && url.hostname) {
        return null;
      }
    } catch {
      // fall through to the error below
    }
  }
  return 'Not an http(s) URL. Use e.g. http://localhost:11434, or leave it empty for the server default.';
};

/** Every invalid URL field among the selected pipelines' current config. */
export const findUrlFieldErrors = (
  pipelines: PipelineDescriptor[],
  selectedPipelineIds: string[],
  config: Record<string, unknown>
): string[] =>
  pipelines
    .filter((pipeline) => selectedPipelineIds.includes(pipeline.id))
    .flatMap((pipeline) =>
      (pipeline.parameters ?? [])
        .filter(isUrlParameter)
        .filter((parameter) =>
          urlValueError((config[pipeline.id] as Record<string, unknown> | undefined)?.[parameter.name])
        )
        .map((parameter) => `${pipeline.id}.${parameter.name}`)
    );
