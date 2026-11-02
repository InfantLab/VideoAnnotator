/** Each pipeline's settings at their defaults, keyed by pipeline (the shape a run's `config` has). */
import type { PipelineDescriptor } from '@/types/pipelines';

export function pipelineDefaults(pipeline: PipelineDescriptor): Record<string, unknown> | null {
  const defaults: Record<string, unknown> = {};
  for (const param of pipeline.parameters ?? []) {
    if (param.default !== undefined) defaults[param.name] = param.default;
  }
  return Object.keys(defaults).length > 0 ? defaults : null;
}

/** `config` with every pipeline's defaults under what it already sets. */
export function withDefaults(pipelines: PipelineDescriptor[], config: Record<string, unknown>): Record<string, unknown> {
  const next: Record<string, unknown> = { ...config };
  for (const pipeline of pipelines) {
    const defaults = pipelineDefaults(pipeline);
    if (!defaults) continue;
    const current = next[pipeline.id];
    next[pipeline.id] = current && typeof current === 'object' ? { ...defaults, ...(current as Record<string, unknown>) } : defaults;
  }
  return next;
}

/** Only the selected pipelines' settings, or undefined when there are none. */
export function configFor(selected: string[], config: Record<string, unknown>): Record<string, unknown> | undefined {
  const entries = Object.entries(config).filter(([id]) => selected.includes(id));
  return entries.length ? Object.fromEntries(entries) : undefined;
}

/** `config` with your own defaults under what it already sets, and the pipelines they filled anything in (UAT #71). */
export function withYours(
  yours: Record<string, Record<string, unknown>>,
  config: Record<string, unknown>,
): { config: Record<string, unknown>; filled: string[] } {
  const next: Record<string, unknown> = { ...config };
  const filled: string[] = [];
  for (const [pipeline, values] of Object.entries(yours)) {
    const current = next[pipeline] && typeof next[pipeline] === 'object' ? (next[pipeline] as Record<string, unknown>) : {};
    if (Object.keys(values).some((name) => current[name] === undefined)) filled.push(pipeline);
    next[pipeline] = { ...values, ...current };
  }
  return { config: next, filled };
}
