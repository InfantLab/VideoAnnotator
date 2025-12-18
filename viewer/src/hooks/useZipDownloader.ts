import { useState, useCallback } from 'react';
import { StandardAnnotationData } from '@/types/annotations';
import { apiClient } from '@/api/client';
import * as zip from '@zip.js/zip.js';
import { detectFileType, mergeAnnotationData, DetectedFile } from '@/lib/parsers/merger';
import {
  ensurePermission,
  getDatasetFolderName,
  getDatasetForJob,
  getRootDirHandle,
  resolveDatasetsDir,
  setDatasetForJob,
  setRootDirHandle
} from '@/lib/localLibrary/libraryStore';

export type DownloadState = 'idle' | 'selecting_dir' | 'downloading' | 'unzipping' | 'ready' | 'error';

interface UseZipDownloaderResult {
  state: DownloadState;
  progress: number;
  error: string | null;
  videoFile: File | null;
  annotationData: StandardAnnotationData | null;
  startDownload: (jobId: string) => Promise<void>;
  reset: () => void;
}

export const useZipDownloader = (): UseZipDownloaderResult => {
  const [state, setState] = useState<DownloadState>('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [annotationData, setAnnotationData] = useState<StandardAnnotationData | null>(null);

  const reset = useCallback(() => {
    setState('idle');
    setProgress(0);
    setError(null);
    setVideoFile(null);
    setAnnotationData(null);
  }, []);

  const tryOpenLocalDataset = useCallback(async (jobId: string) => {
    const root = await getRootDirHandle();
    if (!root) return null;

    const hasPermission = await ensurePermission(root, 'readwrite');
    if (!hasPermission) return null;

    const entry = await getDatasetForJob(jobId);
    const datasetFolderName = entry?.folderName ?? getDatasetFolderName(jobId);

    try {
      const datasetsDir = await resolveDatasetsDir(root);
      const datasetDir = await datasetsDir.getDirectoryHandle(datasetFolderName, { create: false });

      const annotationsHandle = await datasetDir.getFileHandle('annotations_merged.json', { create: false }).catch(() => null);
      if (!annotationsHandle) return null;

      let videoFile: File | null = null;
      if (entry?.videoFileName) {
        const fileHandle = await datasetDir.getFileHandle(entry.videoFileName, { create: false }).catch(() => null);
        if (fileHandle) {
          videoFile = await fileHandle.getFile();
        }
      }

      if (!videoFile) {
        // If index entry was lost, fall back to dataset.json.
        const manifestHandle = await datasetDir.getFileHandle('dataset.json', { create: false }).catch(() => null);
        if (manifestHandle) {
          const manifestFile = await manifestHandle.getFile();
          const manifest = JSON.parse(await manifestFile.text()) as {
            video?: { local?: { relative_path?: unknown } };
          };
          const relPath = manifest.video?.local?.relative_path;
          if (typeof relPath === 'string' && relPath.length > 0) {
            const fileHandle = await datasetDir.getFileHandle(relPath, { create: false }).catch(() => null);
            if (fileHandle) {
              videoFile = await fileHandle.getFile();
            }
          }
        }
      }

      if (!videoFile) {
        return null;
      }

      const annotationsFile = await annotationsHandle.getFile();
      const annotationsText = await annotationsFile.text();
      const annotations = JSON.parse(annotationsText) as StandardAnnotationData;

      return { videoFile, annotationData: annotations };
    } catch {
      return null;
    }
  }, []);

  const pickOrReuseRootDir = useCallback(async (): Promise<FileSystemDirectoryHandle | null> => {
    const supportsFS = 'showDirectoryPicker' in window;
    if (!supportsFS) return null;

    const existing = await getRootDirHandle();
    if (existing) {
      const ok = await ensurePermission(existing, 'readwrite');
      if (ok) return existing;
    }

    try {
      // @ts-expect-error - File System Access API is not always present in TS lib.dom
      const handle: FileSystemDirectoryHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
      await setRootDirHandle(handle);
      return handle;
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return null;
      return null;
    }
  }, []);

  const writeFileToDir = useCallback(async (dir: FileSystemDirectoryHandle, name: string, blob: Blob) => {
    const fileHandle = await dir.getFileHandle(name, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();
  }, []);

  const ingestToLocalDataset = useCallback(
    async (jobId: string, rootDir: FileSystemDirectoryHandle, zipBlob: Blob, video: File, annotations: StandardAnnotationData) => {
      const datasetsDir = await resolveDatasetsDir(rootDir);
      const folderName = getDatasetFolderName(jobId);
      const datasetDir = await datasetsDir.getDirectoryHandle(folderName, { create: true });

      // Save original zip (transport) for debugging/auditing.
      await writeFileToDir(datasetDir, `job_${jobId}_artifacts.zip`, zipBlob);

      // Save canonical artifacts.
      await writeFileToDir(datasetDir, video.name, video);
      await writeFileToDir(datasetDir, 'annotations_merged.json', new Blob([JSON.stringify(annotations, null, 2)], { type: 'application/json' }));

      const jobMeta = await apiClient.getJob(jobId).catch(() => null);
      const config = apiClient.getConfig();

      const datasetId = crypto.randomUUID();
      const now = new Date().toISOString();
      const datasetManifest = {
        schema_version: '1',
        dataset_id: datasetId,
        created_at: now,
        updated_at: now,
        title: jobMeta?.video_filename ?? video.name,
        video: {
          original_filename: jobMeta?.video_filename ?? video.name,
          size_bytes: jobMeta?.video_size_bytes ?? video.size,
          duration_seconds: jobMeta?.video_duration_seconds ?? undefined,
          local: {
            relative_path: video.name
          }
        },
        artifacts: [
          {
            kind: 'annotations_merged',
            local: { relative_path: 'annotations_merged.json' }
          }
        ],
        provenance: {
          source: 'videoannotator',
          server_job: jobMeta
            ? {
                base_url: config.baseURL,
                job_id: jobId,
                job_created_at: jobMeta.created_at,
                job_completed_at: jobMeta.completed_at,
                selected_pipelines: jobMeta.selected_pipelines ?? [],
                status: jobMeta.status
              }
            : {
                base_url: config.baseURL,
                job_id: jobId
              }
        }
      };

      await writeFileToDir(datasetDir, 'dataset.json', new Blob([JSON.stringify(datasetManifest, null, 2)], { type: 'application/json' }));

      await setDatasetForJob(jobId, {
        datasetId,
        folderName,
        createdAt: now,
        videoFileName: video.name
      });
    },
    [writeFileToDir]
  );

  const startDownload = useCallback(async (jobId: string) => {
    console.log('Starting download for job:', jobId);
    setError(null);
    setProgress(0);

    // 0) Fast path: open local dataset if already ingested
    try {
      const local = await tryOpenLocalDataset(jobId);
      if (local) {
        setState('ready');
        setVideoFile(local.videoFile);
        setAnnotationData(local.annotationData);
        return;
      }
    } catch {
      // Ignore and continue to download path.
    }

    setState('selecting_dir');
    setError(null);
    setProgress(0);
    
    try {
      const supportsFS = 'showDirectoryPicker' in window;
      let rootDirHandle: FileSystemDirectoryHandle | null = null;
      if (supportsFS) {
        rootDirHandle = await pickOrReuseRootDir();
        if (!rootDirHandle) {
          // User canceled folder picker.
          setState('idle');
          return;
        }
      }

      // 1. Fetch the artifacts stream
      const response = await apiClient.getJobArtifacts(jobId);
      
      if (!response.body) {
        throw new Error('Response body is empty');
      }

      const contentLength = response.headers.get('Content-Length');
      const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;
      let loadedBytes = 0;

      // 2. Create a progress stream
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loadedBytes += chunk.length;
          if (totalBytes > 0) {
            setProgress((loadedBytes / totalBytes) * 100);
          }
          controller.enqueue(chunk);
        }
      });

      setState('downloading');

      let blob: Blob;

      if (rootDirHandle) {
        // Stream to memory first for unzip. We also persist the ZIP into the dataset folder after unzip.
        // (Teeing streams is possible but adds complexity; keep this robust first.)
        blob = await new Response(response.body.pipeThrough(progressStream)).blob();
      } else {
        // Fallback: Buffer in memory
        blob = await new Response(response.body.pipeThrough(progressStream)).blob();
        
        // If FS not supported, trigger standard download
        if (!supportsFS) {
           const url = URL.createObjectURL(blob);
           const a = document.createElement('a');
           a.href = url;
           a.download = `job_${jobId}_artifacts.zip`;
           a.click();
           URL.revokeObjectURL(url);
        }
      }
      
      // 4. Unzip logic
      setState('unzipping');
      
      const reader = new zip.BlobReader(blob);
      const zipReader = new zip.ZipReader(reader);
      const entries = await zipReader.getEntries();
      
      let foundVideo: File | null = null;
      let foundAnnotations: StandardAnnotationData | null = null;
      const candidateFiles: File[] = [];

      type ZipEntryLike = {
        filename: string;
        directory?: boolean;
        getData?: (writer: unknown) => Promise<Blob>;
      };

      for (const entry of entries) {
        const entryLike = entry as unknown as ZipEntryLike;
        if (entryLike.directory) continue;

        if (entryLike.filename.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
          const getData = entryLike.getData;
          const videoBlob = typeof getData === 'function' ? await getData(new zip.BlobWriter()) : null;
          if (videoBlob) {
            foundVideo = new File([videoBlob], entryLike.filename, { type: 'video/mp4' });
          }
        } else {
          // Extract other files for detection (JSON, VTT, RTTM)
          const getData = entryLike.getData;
          const fileBlob = typeof getData === 'function' ? await getData(new zip.BlobWriter()) : null;
          if (fileBlob) {
            candidateFiles.push(new File([fileBlob], entryLike.filename));
          }
        }
      }

      await zipReader.close();

      if (!foundVideo) {
        throw new Error('No video file found in artifacts ZIP');
      }

      // Detect and merge annotations
      if (candidateFiles.length > 0) {
        console.log('Detecting annotation files from ZIP:', candidateFiles.map(f => f.name));
        const detectedFiles: DetectedFile[] = [];
        
        for (const file of candidateFiles) {
          const detected = await detectFileType(file);
          if (detected.type !== 'unknown') {
            console.log(`Detected ${file.name} as ${detected.type}`);
            detectedFiles.push(detected);
          } else {
            // Fallback: Check if it's a legacy results.json (StandardAnnotationData)
            if (file.name === 'results.json') {
               try {
                 const text = await file.text();
                 const json = JSON.parse(text);
                 if (json.metadata && json.annotations) {
                   console.log('Detected legacy results.json');
                   foundAnnotations = json;
                 }
               } catch (e) {
                 console.warn('Failed to parse potential results.json', e);
               }
            }
          }
        }

        if (detectedFiles.length > 0) {
          // Add the video file to the detected files list so merger can use it for metadata
          if (foundVideo) {
             detectedFiles.push({
               file: foundVideo,
               type: 'video',
               confidence: 1.0
             });
          }

          const result = await mergeAnnotationData(detectedFiles);
          foundAnnotations = result.data;
        }
      }

      if (!foundAnnotations) {
        console.warn('No valid annotations found in artifacts ZIP');
        // We might want to allow viewing video without annotations, 
        // but for now let's assume annotations are expected or create empty structure
        const now = new Date().toISOString();
        foundAnnotations = {
          video_info: {
            filename: foundVideo.name,
            duration: 0,
            width: 0,
            height: 0,
            frame_rate: 30
          },
          metadata: {
            created: now,
            version: '1.0.0',
            pipelines: [],
            source: 'videoannotator'
          }
        };
      }

      setVideoFile(foundVideo);
      setAnnotationData(foundAnnotations);

      // Persist to local library if we have a chosen root folder.
      if (rootDirHandle) {
        try {
          await ingestToLocalDataset(jobId, rootDirHandle, blob, foundVideo, foundAnnotations);
        } catch (persistErr) {
          console.warn('Failed to persist dataset locally (viewer will still work this session):', persistErr);
        }
      }

      setState('ready');

    } catch (err) {
      console.error('Download failed:', err);
      setError(err instanceof Error ? err.message : 'Download failed');
      setState('error');
    }
  }, [ingestToLocalDataset, pickOrReuseRootDir, tryOpenLocalDataset]);

  return {
    state,
    progress,
    error,
    videoFile,
    annotationData,
    startDownload,
    reset
  };
};
