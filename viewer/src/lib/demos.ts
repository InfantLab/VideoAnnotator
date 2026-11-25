/**
 * The demo results, opened in the standalone Video page without a server
 * (spec 029 R14): their bundled files are fetched into Files and read as if
 * they had been dropped in.
 *
 * Five clips of one family from the BabyJokes dataset, one per game, with the
 * six default pipelines' results. They ship inside the viewer (public/demo/),
 * so a demo opens offline and on any build, at the cost of about 13 MB.
 */
import type { LoadedFiles } from '@/components/FileUploader';
import { detectFileType } from '@/lib/fileDetection';
import { mergeAnnotationData } from '@/lib/parsers/merger';

export const DEMO_DATASET = {
  name: 'BabyJokes',
  doi: '10.5281/zenodo.23913806',
  url: 'https://doi.org/10.5281/zenodo.23913806',
  licence: 'CC BY 4.0',
};

const BASE = `${import.meta.env.BASE_URL}demo/babyjokes/`;

const RESULTS = [
  '_face_detections.json',
  '_openface3_analysis.json',
  '_person_tracking.json',
  '_scene_detection.json',
  '_speaker_diarization.rttm',
  '_speaker_diarization.rttm.provenance.json',
  '_speech_recognition.vtt',
];

function clip(name: string): { video: string; results: string[] } {
  return { video: `${BASE}${name}.mp4`, results: RESULTS.map((suffix) => `${BASE}${name}${suffix}`) };
}

export const DEMO_DATA_SETS = {
  'babyjokes-peekaboo': clip('FVM73L.sess1.joke1.rep2.take1.Peekaboo'),
  'babyjokes-cat': clip('FVM73L.sess1.joke2.rep2.take1.ThatsNotACat'),
  'babyjokes-hat': clip('FVM73L.sess1.joke3.rep1.take1.ThatsNotAHat'),
  'babyjokes-nomnom': clip('FVM73L.sess1.joke4.rep1.take1.NomNomNom'),
  'babyjokes-paper': clip('FVM73L.sess1.joke5.rep3.take1.TearingPaper'),
} as const;

export type DemoKey = keyof typeof DEMO_DATA_SETS;

export const DEMO_KEY: DemoKey = 'babyjokes-peekaboo';

/** What the parent said after each game, from the dataset's ratings. */
export const DEMOS: { key: DemoKey; title: string; description: string }[] = [
  { key: 'babyjokes-peekaboo', title: 'Peekaboo', description: 'the baby laughed, as at every peekaboo' },
  { key: 'babyjokes-cat', title: "That's not a cat", description: 'the baby laughed, after not laughing the first time' },
  { key: 'babyjokes-hat', title: "That's not a hat", description: 'no laugh' },
  { key: 'babyjokes-nomnom', title: 'Nom nom nom', description: 'no laugh' },
  { key: 'babyjokes-paper', title: 'Tearing paper', description: 'the baby laughed, the third time only' },
];

/** A demo clip's own name, which can go in a problem report as it identifies nobody. */
export function isDemoVideo(name: string | null | undefined): boolean {
  if (!name) return false;
  return Object.values(DEMO_DATA_SETS).some((set) => set.video.split('/').pop() === name);
}

export function isDemoKey(value: string | null): value is DemoKey {
  return value !== null && value in DEMO_DATA_SETS;
}

// A plain fetch() rejects with a TypeError (not an HTTP error) when nothing
// answers, so that case gets a message a user can act on.
export function getDemoFetchErrorMessage(error: unknown): string {
  if (error instanceof TypeError) {
    return "The demo files couldn't be fetched. Check that the viewer is still running and try again.";
  }
  return error instanceof Error ? error.message : String(error);
}

async function fetchFile(url: string): Promise<File> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Couldn't fetch ${url} (${response.status})`);
  const blob = await response.blob();
  return new File([blob], url.split('/').pop() ?? 'file', { type: blob.type });
}

export async function loadDemoFiles(key: DemoKey): Promise<LoadedFiles> {
  const { video: videoUrl, results } = DEMO_DATA_SETS[key];
  let fetched: File[];
  try {
    fetched = await Promise.all([videoUrl, ...results].map(fetchFile));
  } catch (error) {
    throw new Error(getDemoFetchErrorMessage(error));
  }
  const [video, ...files] = fetched;
  const detected = await Promise.all(fetched.map((file) => detectFileType(file)));
  const merged = await mergeAnnotationData(detected);
  return { video, results: files, data: merged.data };
}
