/**
 * The demo results, opened in the standalone Video page without a server
 * (spec 029 R14): their bundled files are fetched into Files and read as if
 * they had been dropped in.
 */
import type { LoadedFiles } from '@/components/FileUploader';
import { detectFileType } from '@/lib/fileDetection';
import { mergeAnnotationData } from '@/lib/parsers/merger';
import { DEMO_DATA_SETS, getDemoFetchErrorMessage } from '@/utils/debugUtils';

export type DemoKey = keyof typeof DEMO_DATA_SETS;

export const DEMO_KEY: DemoKey = 'peekaboo-rep3-v1.1.1';

export const DEMOS: { key: DemoKey; title: string; description: string }[] = [
  {
    key: 'peekaboo-rep3-v1.1.1',
    title: 'Peekaboo',
    description: 'A parent and baby playing peekaboo: speech, speakers, faces and people.',
  },
];

export function isDemoKey(value: string | null): value is DemoKey {
  return value !== null && value in DEMO_DATA_SETS;
}

async function fetchFile(url: string): Promise<File> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Couldn't fetch ${url} (${response.status})`);
  const blob = await response.blob();
  return new File([blob], url.split('/').pop() ?? 'file', { type: blob.type });
}

export async function loadDemoFiles(key: DemoKey): Promise<LoadedFiles> {
  const { video: videoUrl, audio: _audio, ...results } = DEMO_DATA_SETS[key];
  let fetched: File[];
  try {
    fetched = await Promise.all([videoUrl, ...Object.values(results)].map(fetchFile));
  } catch (error) {
    throw new Error(getDemoFetchErrorMessage(error));
  }
  const [video, ...files] = fetched;
  const detected = await Promise.all(fetched.map((file) => detectFileType(file)));
  const merged = await mergeAnnotationData(detected);
  return { video, results: files, data: merged.data };
}
