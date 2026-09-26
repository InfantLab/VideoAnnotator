// The playback message names the codec from the file's own sample entry.
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { sniffVideoCodec } from '@/lib/videoCodec';

const stsdWith = (fourcc: string, padBefore = 0) => {
  const bytes = new Uint8Array(padBefore + 64);
  const at = (text: string, offset: number) => [...text].forEach((c, i) => (bytes[offset + i] = c.charCodeAt(0)));
  at('stsd', padBefore);
  at(fourcc, padBefore + 16); // after version/flags, entry count, entry size
  return bytes;
};

describe('sniffVideoCodec', () => {
  it('reads hev1 from the demo H.265 video', async () => {
    const file = readFileSync(path.resolve(__dirname, '../../../demo-assets/2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4'));
    expect(await sniffVideoCodec(new Blob([file]))).toEqual({ fourcc: 'hev1', label: 'H.265/HEVC' });
  });

  it('skips an audio sample description and finds the video one', async () => {
    const audioThenVideo = new Blob([stsdWith('mp4a'), stsdWith('avc1')]);
    expect(await sniffVideoCodec(audioThenVideo)).toEqual({ fourcc: 'avc1', label: 'H.264' });
  });

  it('returns null when nothing is recognised', async () => {
    expect(await sniffVideoCodec(new Blob([new Uint8Array(128)]))).toBeNull();
  });
});
