import { describe, expect, it } from 'vitest';

import { settingsDiff } from '@/lib/settingsDiff';

describe('settingsDiff', () => {
  it('lists only the settings that differ, nested keys flattened', () => {
    const a = { settings: { threshold: 0.5, model: 'm', detector: { backend: 'retinaface', size: 640 } } };
    const b = { settings: { threshold: 0.7, model: 'm', detector: { backend: 'retinaface', size: 1280 } } };
    expect(settingsDiff(a, b)).toEqual([
      { key: 'detector.size', a: '640', b: '1280' },
      { key: 'threshold', a: '0.5', b: '0.7' },
    ]);
  });

  it('says not recorded for a setting one run lacks', () => {
    expect(settingsDiff({ settings: { a: 1 } }, { settings: { b: true } })).toEqual([
      { key: 'a', a: '1', b: 'not recorded' },
      { key: 'b', a: 'not recorded', b: 'true' },
    ]);
  });

  it('says not recorded for a run without provenance', () => {
    expect(settingsDiff(null, { settings: { a: 1 } })).toEqual([{ key: 'settings', a: 'not recorded', b: 'recorded' }]);
    expect(settingsDiff(null, undefined)).toEqual([]);
  });

  it('finds nothing between equal settings', () => {
    expect(settingsDiff({ settings: { a: [1, 2] } }, { settings: { a: [1, 2] } })).toEqual([]);
  });
});
