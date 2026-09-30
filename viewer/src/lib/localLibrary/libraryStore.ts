import { idbGet, idbSet } from '@/lib/persistence/idbKv';

const KEY_ROOT_DIR_HANDLE = 'localLibrary.rootDirHandle';
const KEY_JOB_INDEX = 'localLibrary.jobIndex';

export type JobDatasetIndexEntry = {
  datasetId: string;
  folderName: string;
  createdAt: string;
  videoFileName?: string;
};

export type JobDatasetIndex = Record<string, JobDatasetIndexEntry>;

export type LocalFsPermissionMode = 'read' | 'readwrite';

type PermissionQueryableHandle = FileSystemHandle & {
  queryPermission?: (descriptor: { mode: LocalFsPermissionMode }) => Promise<PermissionState>;
  requestPermission?: (descriptor: { mode: LocalFsPermissionMode }) => Promise<PermissionState>;
};

export async function getRootDirHandle(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const handle = await idbGet<FileSystemDirectoryHandle>(KEY_ROOT_DIR_HANDLE);
    return handle ?? null;
  } catch {
    return null;
  }
}

export async function setRootDirHandle(handle: FileSystemDirectoryHandle): Promise<void> {
  await idbSet(KEY_ROOT_DIR_HANDLE, handle);
}

export async function getJobDatasetIndex(): Promise<JobDatasetIndex> {
  return (await idbGet<JobDatasetIndex>(KEY_JOB_INDEX)) ?? {};
}

export async function setJobDatasetIndex(index: JobDatasetIndex): Promise<void> {
  await idbSet(KEY_JOB_INDEX, index);
}

export async function getDatasetForJob(jobId: string): Promise<JobDatasetIndexEntry | null> {
  const index = await getJobDatasetIndex();
  return index[jobId] ?? null;
}

export async function setDatasetForJob(jobId: string, entry: JobDatasetIndexEntry): Promise<void> {
  const index = await getJobDatasetIndex();
  index[jobId] = entry;
  await setJobDatasetIndex(index);
}

export async function ensurePermission(
  handle: FileSystemHandle,
  mode: LocalFsPermissionMode = 'readwrite'
): Promise<boolean> {
  // Some browsers may throw for these APIs even if types exist.
  try {
    const permissionHandle = handle as PermissionQueryableHandle;
    const query = await permissionHandle.queryPermission?.({ mode });
    if (query === 'granted') return true;

    const request = await permissionHandle.requestPermission?.({ mode });
    return request === 'granted';
  } catch {
    return false;
  }
}

export function getLegacyDatasetsDirName() {
  return 'videoannotator_datasets';
}

/**
 * Returns the directory where datasets are stored.
 *
 * New installs: datasets live directly under the selected library root.
 * Legacy installs: datasets live under <root>/videoannotator_datasets.
 */
export async function resolveDatasetsDir(
  rootDir: FileSystemDirectoryHandle
): Promise<FileSystemDirectoryHandle> {
  const legacyName = getLegacyDatasetsDirName();
  const legacy = await rootDir.getDirectoryHandle(legacyName, { create: false }).catch(() => null);
  return legacy ?? rootDir;
}

export function getDatasetFolderName(jobId: string) {
  // Conservative, filesystem-safe, stable.
  return `job_${jobId}`;
}
