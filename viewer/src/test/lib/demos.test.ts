/** The demos ship in public/demo/, and every file they name is there and readable. */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { detectFileType } from '@/lib/fileDetection';
import { DEMO_DATA_SETS, DEMOS, getDemoFetchErrorMessage, isDemoVideo } from '@/lib/demos';
import { checkDataIntegrity } from '@/utils/debugUtils';

const PUBLIC = path.resolve(__dirname, '../../../public');
const onDisk = (url: string) => path.join(PUBLIC, url.replace(/^\//, ''));

afterEach(() => vi.restoreAllMocks());

describe('demos', () => {
  it('lists every demo, the first being the default', () => {
    expect(DEMOS.map((d) => d.key).sort()).toEqual(Object.keys(DEMO_DATA_SETS).sort());
  });

  it.each(Object.entries(DEMO_DATA_SETS))('%s: every file is in public/demo', (_key, set) => {
    for (const url of [set.video, ...set.results]) {
      expect(url.startsWith('/demo/babyjokes/')).toBe(true);
      expect(existsSync(onDisk(url)), url).toBe(true);
    }
  });

  it.each(Object.entries(DEMO_DATA_SETS))('%s: every result is read as what it is', async (_key, set) => {
    const types = await Promise.all(
      set.results.map(async (url) => {
        const name = url.split('/').pop()!;
        return (await detectFileType(new File([readFileSync(onDisk(url), 'utf8')], name))).type;
      }),
    );
    expect(types).toEqual([
      'face_analysis',
      'openface3_faces',
      'person_tracking',
      'scene_detection',
      'speaker_diarization',
      'provenance',
      'speech_recognition',
    ]);
  });

  it('a demo clip can be named in a problem report; another video cannot', () => {
    expect(isDemoVideo('FVM73L.sess1.joke1.rep2.take1.Peekaboo.mp4')).toBe(true);
    expect(isDemoVideo('my_baby.mp4')).toBe(false);
  });

  it('says what to do when nothing answers', () => {
    expect(getDemoFetchErrorMessage(new TypeError('Failed to fetch'))).toMatch(/viewer is still running/);
  });

  it('the integrity check names a file that is missing', async () => {
    vi.spyOn(global, 'fetch').mockImplementation(async (input) =>
      new Response(null, { status: String(input).endsWith('.vtt') ? 404 : 200 }),
    );
    const result = await checkDataIntegrity('babyjokes-cat');
    expect(result.valid).toBe(false);
    expect(result.issues).toEqual([expect.stringMatching(/ThatsNotACat_speech_recognition\.vtt: 404$/)]);
  });
});
