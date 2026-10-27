/** A run is named by what it ran on, then what it found; no date in the name (UAT #41). */
import { describe, expect, it } from 'vitest';

import { runName, runSubject } from '@/lib/runName';

describe('run names', () => {
  it('says what the run was on', () => {
    expect(runSubject({ kind: 'videos', paths: ['/d/pk/expCP2.mov'], label: '1 video' }, ['/d/pk/expCP2.mov'], null)).toBe('expCP2');
    expect(runSubject({ kind: 'videos', paths: [], label: '2 videos' }, ['/d/pk/a.mp4', '/d/pk/b.mp4'], null)).toBe('pk');
    expect(runSubject({ kind: 'videos', paths: [], label: '2 videos' }, ['/x/a.mp4', '/y/b.mp4'], null)).toBe('2 videos');
    expect(runSubject({ kind: 'folder', path: '/data/site_a/', recursive: false }, null, null)).toBe('site_a');
    expect(runSubject(null, null, [new File(['a'], 'clip.mp4')])).toBe('clip');
  });

  it('then what it found: two by name, more as a count', () => {
    expect(runName('expCP2', ['PySceneDetect + CLIP'])).toBe('expCP2 · PySceneDetect + CLIP');
    expect(runName('pk', ['pyannote community-1', 'Whisper large-v3-turbo'])).toBe('pk · pyannote community-1, Whisper large-v3-turbo');
    expect(runName('pk', ['a', 'b', 'c'])).toBe('pk · 3 pipelines');
    expect(runName('pk', [])).toBe('pk');
  });
});
