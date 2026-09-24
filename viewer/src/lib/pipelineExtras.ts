import type { PipelineDescriptor } from "@/types/pipelines";

/**
 * Extracts the bracketed extras-group name from a VideoAnnotator install hint,
 * e.g. "pip install videoannotator[face]" -> "face".
 *
 * Returns null when the hint doesn't match the expected shape, so callers can
 * fall back to showing the locked pipeline without an Install action rather
 * than calling the install endpoint with a bad/guessed group name.
 */
export function extraNameFromInstallHint(hint: string | undefined | null): string | null {
  if (!hint) return null;
  const match = hint.match(/\[([a-zA-Z0-9_-]+)\]/);
  return match ? match[1] : null;
}

/**
 * How a pipeline's card should behave, from its readiness (VideoAnnotator spec
 * 011) or, on older servers without it, from `available` alone.
 * Only `selectable` pipelines can be chosen for a job.
 */
export type PipelineCardMode =
  | 'selectable'
  | 'install'
  | 'installing'
  | 'restart'
  | 'setup'
  | 'unavailable';

export function pipelineCardMode(pipeline: PipelineDescriptor): PipelineCardMode {
  const readiness = pipeline.readiness;
  if (!readiness) return pipeline.available === false ? 'install' : 'selectable';
  switch (readiness.state) {
    case 'ready':
      return 'selectable';
    case 'not_installed':
      return 'install';
    case 'installing':
      return 'installing';
    case 'restart_required':
      return 'restart';
    case 'needs_setup':
      return 'setup';
    default:
      // A state added by a newer server: not usable, show whatever it says.
      return 'unavailable';
  }
}

export const PIPELINE_CARD_BADGE: Record<Exclude<PipelineCardMode, 'selectable'>, string> = {
  install: 'Not installed',
  installing: 'Installing',
  restart: 'Restart needed',
  setup: 'Needs setup',
  unavailable: 'Not available'
};

/** The extras group that installs a pipeline: from readiness, else parsed from the install hint. */
export function extrasGroupOf(pipeline: PipelineDescriptor): string | null {
  return pipeline.readiness?.extrasGroup ?? extraNameFromInstallHint(pipeline.installHint);
}
