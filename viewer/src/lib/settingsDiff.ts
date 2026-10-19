/**
 * What differs between two runs' settings, from their provenance (spec 028
 * R11; spec 029's "What changed?" reuses it). Nested settings are flattened
 * with `.`; a run without a provenance record says "not recorded".
 */
import { NOT_RECORDED } from '@/lib/table/rows';

export interface SettingDifference {
  key: string;
  a: string;
  b: string;
}

type Record_ = Record<string, unknown> | null | undefined;

const isDict = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

function flatten(value: unknown, prefix: string, out: Map<string, string>) {
  if (isDict(value) && Object.keys(value).length) {
    for (const [k, v] of Object.entries(value)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else if (prefix) {
    out.set(prefix, typeof value === 'string' ? value : JSON.stringify(value) ?? 'null');
  }
}

const settingsOf = (record: Record_): Map<string, string> | null => {
  if (!isDict(record)) return null;
  const out = new Map<string, string>();
  flatten(record.settings ?? {}, '', out);
  return out;
};

export function settingsDiff(a: Record_, b: Record_): SettingDifference[] {
  const left = settingsOf(a);
  const right = settingsOf(b);
  if (!left || !right) {
    return left === right ? [] : [{ key: 'settings', a: left ? 'recorded' : NOT_RECORDED, b: right ? 'recorded' : NOT_RECORDED }];
  }
  const keys = [...new Set([...left.keys(), ...right.keys()])].sort();
  return keys
    .filter((k) => left.get(k) !== right.get(k))
    .map((k) => ({ key: k, a: left.get(k) ?? NOT_RECORDED, b: right.get(k) ?? NOT_RECORDED }));
}
