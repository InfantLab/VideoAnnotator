/** Reading a pipeline's settings against its schema: defaults, what changed, a summary (UAT #35). */
import type { PipelineDescriptor, PipelineParameterSchema } from '@/types/pipelines';

export const getPipelineConfig = (
  config: Record<string, unknown>,
  pipelineId: string
): Record<string, unknown> => {
  const value = config[pipelineId];
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
};

export const coerceBoolean = (value: unknown, fallback: boolean | undefined) => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    if (value === 'true') return true;
    if (value === 'false') return false;
  }
  return fallback ?? false;
};

export const getParameterDefault = (parameter: PipelineParameterSchema) => {
  if (parameter.default !== undefined) {
    return parameter.default;
  }
  if (parameter.type === 'boolean') {
    return false;
  }
  if (parameter.type === 'multiselect') {
    return [];
  }
  return '';
};

export const normalizeValue = (parameter: PipelineParameterSchema, value: unknown) => {
  if (value === undefined || value === null) {
    return getParameterDefault(parameter);
  }

  switch (parameter.type) {
    case 'boolean':
      return coerceBoolean(value, !!parameter.default);
    case 'integer':
    case 'number':
      return typeof value === 'number' ? value : Number(value);
    case 'multiselect':
      return Array.isArray(value) ? value : [];
    default:
      return value;
  }
};

/**
 * The same normalized/defaulted value a field's widget currently displays
 * (schema default until the user actually edits it), by name — since raw
 * `pipelineConfig[name]` is `undefined` until then, a consumer outside the
 * parameter-render loop (e.g. the Open in Ask AI link) that read the raw config
 * directly would see an empty value even while the form visibly shows a
 * default like the pipeline's own DEFAULT_PROMPT or default model.
 */
export const resolveParameterValue = (
  parameters: PipelineParameterSchema[] | undefined,
  pipelineConfig: Record<string, unknown>,
  name: string
): unknown => {
  const parameter = parameters?.find((p) => p.name === name);
  if (!parameter) return pipelineConfig[name];
  return normalizeValue(parameter, pipelineConfig[name]);
};

/** Whether a setting differs from its default: an untouched field is never changed. */
export const isChanged = (parameter: PipelineParameterSchema, pipelineConfig: Record<string, unknown>) => {
  const raw = pipelineConfig[parameter.name];
  if (raw === undefined) return false;
  return JSON.stringify(normalizeValue(parameter, raw)) !== JSON.stringify(getParameterDefault(parameter));
};

export const shown = (value: unknown): string => {
  if (value === null || value === '') return 'none';
  if (typeof value === 'boolean') return value ? 'on' : 'off';
  if (typeof value === 'string') return value.length > 60 ? `${value.slice(0, 57)}…` : value;
  return JSON.stringify(value);
};

/** A collapsed row's cue: "defaults", or what differs ("Age on · Gender on"); Ask AI by its question. */
export function settingsSummary(pipeline: PipelineDescriptor, config: Record<string, unknown>): string {
  const pipelineConfig = getPipelineConfig(config, pipeline.id);
  const parameters = pipeline.parameters ?? [];
  if (pipeline.id === 'vlm_annotation') {
    const prompt = resolveParameterValue(parameters, pipelineConfig, 'prompt');
    const every = resolveParameterValue(parameters, pipelineConfig, 'frame_interval_sec');
    const question = typeof prompt === 'string' && prompt.trim() ? `“${shown(prompt.trim().split('\n')[0])}”` : 'no question yet';
    return typeof every === 'number' ? `${question} · every ${every} s` : question;
  }
  const changed = parameters.filter((p) => isChanged(p, pipelineConfig));
  if (!changed.length) return parameters.length ? 'defaults' : '';
  const words = changed.map((p) => {
    const value = normalizeValue(p, pipelineConfig[p.name]);
    const label = p.label || p.name;
    return p.type === 'boolean' ? `${label} ${value ? 'on' : 'off'}` : `${label}: ${shown(value)}${p.unit ? ` ${p.unit}` : ''}`;
  });
  return words.length > 3 ? `${words.slice(0, 2).join(' · ')} · ${words.length - 2} more` : words.join(' · ');
}
