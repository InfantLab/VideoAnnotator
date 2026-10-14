/**
 * Reads what made an output file (spec 017), so every overlay can name the
 * pipeline and version that drew it (constitution Principle VI).
 *
 * VideoAnnotator records it as a top-level `provenance` key in JSON, a
 * `NOTE videoannotator-provenance <json>` block in WebVTT, and a companion
 * `<file>.provenance.json` beside RTTM. Older files have none, or (COCO) only
 * the VideoAnnotator version; nothing is ever inferred beyond what a file says.
 */
import type { ProvenanceInfo, ProvenanceRecord } from '@/types/annotations';

export const COMPANION_SUFFIX = '.provenance.json';
const VTT_MARKER = 'NOTE videoannotator-provenance ';

const NONE: ProvenanceInfo = { kind: 'none' };

function asRecord(value: unknown): ProvenanceRecord | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Partial<ProvenanceRecord>;
  return typeof v.videoannotator_version === 'string' && v.pipeline && typeof v.pipeline.name === 'string'
    ? (value as ProvenanceRecord)
    : null;
}

/** From a parsed JSON output. */
export function provenanceFromJSON(data: unknown): ProvenanceInfo {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return NONE;
  const d = data as Record<string, unknown>;
  const record = asRecord(d.provenance);
  if (record) return { kind: 'recorded', record };
  const info = d.info as Record<string, unknown> | undefined;
  if (typeof info?.version === 'string') return { kind: 'partial', videoannotatorVersion: info.version };
  return NONE;
}

/** From a WebVTT file's text. */
export function provenanceFromWebVTT(text: string): ProvenanceInfo {
  const block = text.split(/\r?\n\r?\n/).find((b) => b.startsWith(VTT_MARKER));
  if (!block) return NONE;
  try {
    const record = asRecord(JSON.parse(block.slice(VTT_MARKER.length)));
    return record ? { kind: 'recorded', record } : NONE;
  } catch {
    return NONE;
  }
}

/** From a companion `.provenance.json` file's text. */
export function provenanceFromCompanion(text: string): ProvenanceInfo {
  try {
    const record = asRecord(JSON.parse(text));
    return record ? { kind: 'recorded', record } : NONE;
  } catch {
    return NONE;
  }
}

/** Reads a JSON or WebVTT output's own record; other formats use a companion. */
export async function provenanceOfFile(file: File): Promise<ProvenanceInfo> {
  const name = file.name.toLowerCase();
  try {
    if (name.endsWith('.json')) return provenanceFromJSON(JSON.parse(await file.text()));
    if (name.endsWith('.vtt')) return provenanceFromWebVTT(await file.text());
  } catch {
    // Unreadable here means unrecorded; the parser reports the file's own problem.
  }
  return NONE;
}

/** The one-line label shown with an overlay. */
export function attributionLabel(info: ProvenanceInfo | undefined, pipeline: string): string {
  switch (info?.kind) {
    case 'recorded':
      return `${info.record.pipeline.name} · VideoAnnotator ${info.record.videoannotator_version}`;
    case 'partial':
      return info.videoannotatorVersion
        ? `${pipeline} · VideoAnnotator ${info.videoannotatorVersion} · rest not recorded`
        : `${pipeline} · version not recorded`;
    case 'ground_truth':
      return `Ground truth · ${info.fileName}`;
    default:
      return `${pipeline} · version not recorded`;
  }
}
