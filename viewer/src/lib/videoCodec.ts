/**
 * What a video file's container says its video codec is, so a playback
 * problem can name the codec from evidence rather than from the file name.
 *
 * Reads the MP4/MOV sample description (`stsd`) and returns its sample-entry
 * code, e.g. "hev1". The `moov` box holding it sits at the start of a
 * fast-start file and at the end otherwise, so only the head and tail are read.
 */

const WINDOW = 2 * 1024 * 1024;

const CODEC_LABELS: Record<string, string> = {
  hev1: 'H.265/HEVC',
  hvc1: 'H.265/HEVC',
  avc1: 'H.264',
  avc3: 'H.264',
  av01: 'AV1',
  vp09: 'VP9',
  mp4v: 'MPEG-4 Part 2',
  apcn: 'ProRes',
  apch: 'ProRes',
  apcs: 'ProRes',
  apco: 'ProRes',
  ap4h: 'ProRes',
};

export interface VideoCodecInfo {
  /** Sample-entry code from the file, e.g. "hev1". */
  fourcc: string;
  /** Human name, e.g. "H.265/HEVC". */
  label: string;
}

function findSampleEntry(bytes: Uint8Array): string | null {
  const ascii = (i: number) => String.fromCharCode(bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]);
  for (let i = 0; i + 4 <= bytes.length; i++) {
    // 's' 't' 's' 'd'
    if (bytes[i] !== 0x73 || bytes[i + 1] !== 0x74 || bytes[i + 2] !== 0x73 || bytes[i + 3] !== 0x64) continue;
    // stsd: version/flags (4), entry count (4), then each entry: size (4), type (4).
    // An audio stsd is found too, so keep looking until a known video code turns up.
    const type = i + 16 + 4 <= bytes.length ? ascii(i + 16) : '';
    if (CODEC_LABELS[type]) return type;
  }
  return null;
}

/** Blob bytes; FileReader where Blob.arrayBuffer is missing (older engines, jsdom). */
function readBytes(blob: Blob): Promise<Uint8Array> {
  if (typeof blob.arrayBuffer === 'function') return blob.arrayBuffer().then((b) => new Uint8Array(b));
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(blob);
  });
}

export async function sniffVideoCodec(file: Blob): Promise<VideoCodecInfo | null> {
  try {
    const parts = file.size <= 2 * WINDOW ? [file] : [file.slice(0, WINDOW), file.slice(file.size - WINDOW)];
    for (const part of parts) {
      const fourcc = findSampleEntry(await readBytes(part));
      if (fourcc) return { fourcc, label: CODEC_LABELS[fourcc] };
    }
  } catch {
    // Evidence only; playback doesn't depend on it.
  }
  return null;
}

/** What the browser claims about a codec. Claims can be wrong: Edge says "probably" to HEVC it then fails to decode. */
export function browserClaimFor(fourcc: string): string {
  const codecs: Record<string, string> = {
    hev1: 'hev1.1.6.L93.B0',
    hvc1: 'hvc1.1.6.L93.B0',
    avc1: 'avc1.64001E',
    avc3: 'avc3.64001E',
    av01: 'av01.0.04M.08',
    vp09: 'vp09.00.10.08',
  };
  const codec = codecs[fourcc];
  if (!codec || typeof document === 'undefined') return 'unknown';
  const answer = document.createElement('video').canPlayType(`video/mp4; codecs="${codec}"`);
  return answer || 'no';
}
