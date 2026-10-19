/**
 * Which lanes and overlays a researcher shows, remembered per user in the
 * browser (spec 028 R13, data-model.md "Lane preferences"). By lane kind, not
 * run, so they carry across videos. A display preference, not research data:
 * when the browser can't store it, the defaults are used and nothing breaks.
 */
import { OLD_OVERLAY_KEYS } from '@/lib/laneKinds';
import { idbGet, idbSet } from '@/lib/persistence/idbKv';

export interface LanePrefs {
  version: 1;
  collapsed: Record<string, boolean>;
  overlay: Record<string, boolean>;
  timelineHeight: number;
  dataFilter: string;
  panelOpen: boolean;
}

export const TIMELINE_HEIGHT = { default: 200, min: 120 } as const;

export const defaultLanePrefs = (): LanePrefs => ({
  version: 1,
  collapsed: {},
  overlay: {},
  timelineHeight: TIMELINE_HEIGHT.default,
  dataFilter: 'all',
  panelOpen: true,
});

export const lanePrefsKey = (serverUrl?: string | null, userId?: string | null): string =>
  serverUrl ? `lanePrefs:${serverUrl}:${userId ?? 'anonymous'}` : 'lanePrefs:local';

/** Unknown kinds are shown, expanded, with their overlay on. */
export const isCollapsed = (prefs: LanePrefs, kind: string): boolean => prefs.collapsed[kind] ?? false;
export const overlayOn = (prefs: LanePrefs, key: string): boolean => prefs.overlay[key] ?? true;

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

function booleans(v: unknown): Record<string, boolean> {
  if (!isRecord(v)) return {};
  return Object.fromEntries(Object.entries(v).filter(([, b]) => typeof b === 'boolean')) as Record<string, boolean>;
}

/** Overlay switches as `OverlaySettings` named them before spec 028, by lane kind. */
export function overlayFromOldSettings(old: Record<string, unknown>): Record<string, boolean> {
  const mapped: Record<string, boolean> = {};
  for (const [oldKey, kind] of Object.entries(OLD_OVERLAY_KEYS)) {
    if (typeof old[oldKey] === 'boolean') mapped[kind] = old[oldKey] as boolean;
  }
  return mapped;
}

/** Whatever was stored, as preferences: unknown fields dropped, old overlay keys mapped once. */
export function normaliseLanePrefs(stored: unknown): LanePrefs {
  const prefs = defaultLanePrefs();
  if (!isRecord(stored)) return prefs;
  prefs.collapsed = booleans(stored.collapsed);
  const overlay = booleans(stored.overlay);
  const fromOld = overlayFromOldSettings(overlay);
  for (const [oldKey, kind] of Object.entries(OLD_OVERLAY_KEYS)) {
    if (kind !== oldKey) delete overlay[oldKey];
  }
  prefs.overlay = { ...fromOld, ...overlay };
  if (typeof stored.timelineHeight === 'number' && stored.timelineHeight >= TIMELINE_HEIGHT.min) {
    prefs.timelineHeight = stored.timelineHeight;
  }
  if (typeof stored.dataFilter === 'string') prefs.dataFilter = stored.dataFilter;
  if (typeof stored.panelOpen === 'boolean') prefs.panelOpen = stored.panelOpen;
  return prefs;
}

export async function loadLanePrefs(key: string): Promise<LanePrefs> {
  try {
    return normaliseLanePrefs(await idbGet<unknown>(key));
  } catch {
    return defaultLanePrefs();
  }
}

export async function saveLanePrefs(key: string, prefs: LanePrefs): Promise<void> {
  try {
    await idbSet(key, prefs);
  } catch {
    // Private windows and blocked storage: the page works, it just won't remember.
  }
}
