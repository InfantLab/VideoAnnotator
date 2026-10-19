import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  defaultLanePrefs,
  isCollapsed,
  lanePrefsKey,
  loadLanePrefs,
  overlayOn,
  saveLanePrefs,
} from '@/lib/lanePrefs';
import * as idbKv from '@/lib/persistence/idbKv';

afterEach(() => vi.restoreAllMocks());

describe('lane preferences', () => {
  it('round-trips through the browser store', async () => {
    const key = lanePrefsKey('http://lab:18011', 'u1');
    const prefs = { ...defaultLanePrefs(), collapsed: { scenes: true }, overlay: { faces: false }, timelineHeight: 260 };
    await saveLanePrefs(key, prefs);
    expect(await loadLanePrefs(key)).toEqual(prefs);
    expect(await loadLanePrefs(lanePrefsKey('http://lab:18011', 'u2'))).toEqual(defaultLanePrefs());
  });

  it('shows a kind it has never seen, expanded with its overlay on', () => {
    const prefs = defaultLanePrefs();
    expect(isCollapsed(prefs, 'other:blink/rate')).toBe(false);
    expect(overlayOn(prefs, 'other.overlay')).toBe(true);
    expect(overlayOn(prefs, 'faces.landmarks')).toBe(true);
    expect(overlayOn(prefs, 'faces.gaze')).toBe(false);
  });

  it('maps overlay switches saved under their old names onto lane kinds', async () => {
    await idbKv.idbSet('lanePrefs:local', { overlay: { pose: false, subtitles: true, emotions: false, faces: false } });
    const prefs = await loadLanePrefs('lanePrefs:local');
    expect(prefs.overlay).toEqual({ people: false, speech: true, emotion: false, faces: false });
  });

  it('falls back to the defaults when the browser can\'t store anything', async () => {
    vi.spyOn(idbKv, 'idbGet').mockRejectedValue(new Error('blocked'));
    vi.spyOn(idbKv, 'idbSet').mockRejectedValue(new Error('blocked'));
    await expect(saveLanePrefs('k', defaultLanePrefs())).resolves.toBeUndefined();
    expect(await loadLanePrefs('k')).toEqual(defaultLanePrefs());
  });

  it('drops what it doesn\'t understand', async () => {
    await idbKv.idbSet('bad', { collapsed: { scenes: 'yes' }, timelineHeight: 5, panelOpen: false });
    const prefs = await loadLanePrefs('bad');
    expect(prefs.collapsed).toEqual({});
    expect(prefs.timelineHeight).toBe(200);
    expect(prefs.panelOpen).toBe(false);
  });
});
