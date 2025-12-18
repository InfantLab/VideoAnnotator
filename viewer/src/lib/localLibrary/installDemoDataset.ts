import { loadDemoAnnotations, loadDemoVideo } from '@/utils/debugUtils';
import {
  ensurePermission,
  resolveDatasetsDir,
  setDatasetForJob,
  type JobDatasetIndexEntry
} from '@/lib/localLibrary/libraryStore';

export const DEMO_JOB_ID_PREFIX = 'demo:';

export type InstallDemoDatasetResult = {
  jobId: string;
  entry: JobDatasetIndexEntry;
};

function toSafeFolderName(input: string): string {
  return input
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 100);
}

async function writeBlobToDir(dir: FileSystemDirectoryHandle, name: string, blob: Blob): Promise<void> {
  const fileHandle = await dir.getFileHandle(name, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(blob);
  await writable.close();
}

export async function installBundledDemoDataset(options: {
  rootDir: FileSystemDirectoryHandle;
  demoKey?: Parameters<typeof loadDemoVideo>[0];
}): Promise<InstallDemoDatasetResult> {
  const demoKey = options.demoKey ?? ('veatic-3-silent' as const);
  const jobId = `${DEMO_JOB_ID_PREFIX}${demoKey}`;

  const permitted = await ensurePermission(options.rootDir, 'readwrite');
  if (!permitted) {
    throw new Error('Permission denied for selected library folder.');
  }

  const datasetsDir = await resolveDatasetsDir(options.rootDir);
  const folderName = toSafeFolderName(`demo_${demoKey}`);
  const datasetDir = await datasetsDir.getDirectoryHandle(folderName, { create: true });

  // Load merged demo annotation + demo video from the app-hosted demo assets.
  // Note: this requires the demo assets to be available under /demo/* at runtime.
  const [videoFile, annotationData] = await Promise.all([
    loadDemoVideo(demoKey),
    loadDemoAnnotations(demoKey)
  ]);

  if (!videoFile || !annotationData) {
    throw new Error('Failed to load demo data from app assets.');
  }

  await writeBlobToDir(datasetDir, videoFile.name, videoFile);
  await writeBlobToDir(
    datasetDir,
    'annotations_merged.json',
    new Blob([JSON.stringify(annotationData, null, 2)], { type: 'application/json' })
  );

  const datasetId = crypto.randomUUID();
  const now = new Date().toISOString();
  const datasetManifest = {
    schema_version: '1',
    dataset_id: datasetId,
    created_at: now,
    updated_at: now,
    title: `Demo: ${demoKey}`,
    video: {
      original_filename: videoFile.name,
      size_bytes: videoFile.size,
      local: {
        relative_path: videoFile.name
      }
    },
    artifacts: [
      {
        kind: 'annotations_merged',
        local: { relative_path: 'annotations_merged.json' }
      }
    ],
    provenance: {
      source: 'demo',
      demo: {
        key: demoKey
      }
    }
  };

  await writeBlobToDir(
    datasetDir,
    'dataset.json',
    new Blob([JSON.stringify(datasetManifest, null, 2)], { type: 'application/json' })
  );

  const entry: JobDatasetIndexEntry = {
    datasetId,
    folderName,
    createdAt: now,
    videoFileName: videoFile.name
  };

  await setDatasetForJob(jobId, entry);
  return { jobId, entry };
}
