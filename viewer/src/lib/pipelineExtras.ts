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
  | 'restoring'
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
    case 'restoring':
      // Installed before the server's container was recreated (VideoAnnotator spec 024).
      return 'restoring';
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
  restoring: 'Restoring',
  restart: 'Restart needed',
  setup: 'Needs setup',
  unavailable: 'Not available'
};

/** The extras group that installs a pipeline: from readiness, else parsed from the install hint. */
export function extrasGroupOf(pipeline: PipelineDescriptor): string | null {
  return pipeline.readiness?.extrasGroup ?? extraNameFromInstallHint(pipeline.installHint);
}

/** Only pipelines in the `selectable` card mode can be part of a job. */
export function isPipelineSelectable(pipeline: PipelineDescriptor): boolean {
  return pipelineCardMode(pipeline) === 'selectable';
}

const NOT_READY_FALLBACK: Record<Exclude<PipelineCardMode, 'selectable'>, string> = {
  install: "it isn't installed",
  installing: "it's still installing",
  restoring: "it's being restored after an update or restart",
  restart: 'it needs a server restart',
  setup: 'it needs setup',
  unavailable: "the server says it isn't available"
};

/**
 * Why a pipeline can't run right now, in one line: the server's first blocker
 * if it gave one ("can't reach the Ollama server"), else the card state.
 * Empty for a selectable pipeline.
 */
export function notReadyReason(pipeline: PipelineDescriptor): string {
  const mode = pipelineCardMode(pipeline);
  if (mode === 'selectable') return '';
  const blocker = pipeline.readiness?.blockers?.[0]?.message?.trim();
  return blocker ? blocker.replace(/\.$/, '') : NOT_READY_FALLBACK[mode];
}

/**
 * Splits a wished-for selection (defaults, a retried job's pipelines) into the
 * ids that can run now and notices for the ones left out, e.g.
 * "VLM Frame Annotation left out: Ollama isn't reachable". Ids the catalog
 * doesn't know are kept: the server validates them on submit.
 */
export function partitionSelection(
  ids: string[],
  pipelines: PipelineDescriptor[]
): { kept: string[]; leftOut: string[] } {
  const byId = new Map(pipelines.map((p) => [p.id, p]));
  const kept: string[] = [];
  const leftOut: string[] = [];
  ids.forEach((id) => {
    const pipeline = byId.get(id);
    if (!pipeline || isPipelineSelectable(pipeline)) kept.push(id);
    else leftOut.push(`${pipeline.name} left out: ${notReadyReason(pipeline)}.`);
  });
  return { kept, leftOut };
}
