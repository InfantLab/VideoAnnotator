import { DEMO_DATA_SETS } from '@/utils/debugUtils';
import type { StandardAnnotationData, VideoAnnotatorCompleteResults } from '@/types/annotations';
import {
  getJobDatasetIndex,
  setDatasetForJob,
  type JobDatasetIndexEntry
} from '@/lib/localLibrary/libraryStore';

export const DEMO_JOB_ID_PREFIX = 'demo:';

export type InstallDemoDatasetResult = {
  jobId: string;
  entry: JobDatasetIndexEntry;
};

/** Human-readable labels for demo datasets. */
export const DEMO_LABELS: Record<string, string> = {
  'peekaboo-rep3-v1.1.1': 'Peekaboo (rep 3)',
  'peekaboo-rep2-v1.1.1': 'Peekaboo (rep 2)',
  'tearingpaper-rep1-v1.1.1': 'Tearing Paper',
  'thatsnotahat-rep1-v1.1.1': "That's Not a Hat",
  'veatic-3-silent': 'VEATIC Silent Video',
};

export function getDemoLabel(jobId: string): string | null {
  if (!jobId.startsWith(DEMO_JOB_ID_PREFIX)) return null;
  const key = jobId.slice(DEMO_JOB_ID_PREFIX.length);
  return DEMO_LABELS[key] ?? key;
}

/** Check whether a jobId refers to a bundled demo. */
export function isDemoJobId(jobId: string): boolean {
  return jobId.startsWith(DEMO_JOB_ID_PREFIX);
}

/** Extract the demo key from a demo job ID. */
export function getDemoKey(jobId: string): keyof typeof DEMO_DATA_SETS | null {
  if (!isDemoJobId(jobId)) return null;
  const key = jobId.slice(DEMO_JOB_ID_PREFIX.length);
  return key in DEMO_DATA_SETS ? (key as keyof typeof DEMO_DATA_SETS) : null;
}

/**
 * Convert VideoAnnotator complete_results.json → StandardAnnotationData.
 *
 * This avoids the merger's extractVideoInfo (which creates a <video> element
 * and fails for h265 videos). The viewer will get real dimensions from its
 * own video element at playback time.
 */
function convertCompleteResults(
  raw: VideoAnnotatorCompleteResults,
  videoFileName: string,
): StandardAnnotationData {
  const pipelinesFound: string[] = [];

  const personTracking = raw.pipeline_results?.person?.results ?? [];
  const faceAnalysis = raw.pipeline_results?.face?.results ?? [];
  const sceneDetection = raw.pipeline_results?.scene?.results ?? [];

  if (personTracking.length > 0) pipelinesFound.push('person_tracking');
  if (faceAnalysis.length > 0) pipelinesFound.push('face_analysis');
  if (sceneDetection.length > 0) pipelinesFound.push('scene_detection');

  const data: StandardAnnotationData = {
    video_info: {
      filename: videoFileName,
      duration: 0,
      width: 0,
      height: 0,
      frame_rate: 30,
    },
    metadata: {
      created: new Date().toISOString(),
      version: '1.1.1',
      pipelines: pipelinesFound,
      source: 'videoannotator',
      processing_config: raw.config,
      processing_time: Object.values(raw.pipeline_results ?? {}).reduce(
        (sum, r) => sum + (r?.processing_time ?? 0),
        0,
      ),
      total_duration: raw.total_duration,
    },
  };

  if (personTracking.length > 0) data.person_tracking = personTracking;
  if (faceAnalysis.length > 0) data.face_analysis = faceAnalysis;
  if (sceneDetection.length > 0) data.scene_detection = sceneDetection;

  return data;
}

/**
 * Load a demo dataset directly from bundled assets (no FS needed).
 *
 * Fetches the video and complete_results.json and converts the
 * VideoAnnotator format to StandardAnnotationData inline (no merger
 * pipeline, no <video> element needed for metadata).
 */
export async function loadDemoFromAssets(jobId: string): Promise<{
  videoFile: File;
  annotationData: StandardAnnotationData;
}> {
  const demoKey = getDemoKey(jobId);
  if (!demoKey) throw new Error(`Unknown demo key in "${jobId}".`);

  const paths = DEMO_DATA_SETS[demoKey];

  // Fetch video and complete_results.json in parallel.
  const [videoRes, annotRes] = await Promise.all([
    fetch(paths.video),
    paths.complete_results ? fetch(paths.complete_results) : null,
  ]);

  if (!videoRes.ok) throw new Error(`Failed to fetch demo video (${videoRes.status}).`);

  const videoBlob = await videoRes.blob();
  const videoFileName = paths.video.split('/').pop() || 'demo.mp4';
  const videoFile = new File([videoBlob], videoFileName, { type: videoBlob.type || 'video/mp4' });

  let annotationData: StandardAnnotationData;

  if (annotRes && annotRes.ok) {
    const raw = (await annotRes.json()) as VideoAnnotatorCompleteResults;
    annotationData = convertCompleteResults(raw, videoFileName);
  } else {
    annotationData = {
      video_info: { filename: videoFileName, duration: 0, width: 0, height: 0, frame_rate: 30 },
      metadata: { created: new Date().toISOString(), version: '1.0.0', pipelines: [], source: 'demo' },
    };
  }

  return { videoFile, annotationData };
}

export type DemoFailure = { key: string; error: string };

export type InstallAllResult = {
  installed: number;
  skipped: number;
  failed: DemoFailure[];
};

/**
 * Register ALL bundled demo datasets in IndexedDB.
 *
 * This does NOT write files to disk — demos are loaded from bundled assets
 * at view time. This avoids File System Access API issues entirely.
 */
export async function installAllBundledDemos(
  onProgress?: (msg: string) => void
): Promise<InstallAllResult> {
  const existingIndex = await getJobDatasetIndex();
  const allKeys = Object.keys(DEMO_DATA_SETS) as Array<keyof typeof DEMO_DATA_SETS>;
  const keysToInstall = allKeys.filter(k => !existingIndex[`${DEMO_JOB_ID_PREFIX}${k}`]);
  const skipped = allKeys.length - keysToInstall.length;

  if (keysToInstall.length === 0) {
    return { installed: 0, skipped, failed: [] };
  }

  let installed = 0;
  const failed: DemoFailure[] = [];

  for (const key of keysToInstall) {
    const label = DEMO_LABELS[key] ?? key;
    onProgress?.(`Registering "${label}"…`);

    try {
      const jobId = `${DEMO_JOB_ID_PREFIX}${key}`;
      const paths = DEMO_DATA_SETS[key];
      const videoFileName = paths.video.split('/').pop() || 'demo.mp4';

      const entry: JobDatasetIndexEntry = {
        datasetId: crypto.randomUUID(),
        folderName: `demo_${key}`,
        createdAt: new Date().toISOString(),
        videoFileName,
      };

      await setDatasetForJob(jobId, entry);
      installed++;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`Failed to register demo "${key}":`, err);
      failed.push({ key, error: msg });
    }
  }

  return { installed, skipped, failed };
}
