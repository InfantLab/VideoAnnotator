import { useState, useCallback } from 'react';
import { StandardAnnotationData } from '@/types/annotations';
import { apiClient } from '@/api/client';
import * as zip from '@zip.js/zip.js';
import { detectFileType, mergeAnnotationData, DetectedFile } from '@/lib/parsers/merger';

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

  const startDownload = useCallback(async (jobId: string) => {
    console.log('Starting download for job:', jobId);
    setState('selecting_dir');
    setError(null);
    setProgress(0);
    
    try {
      // Check for File System Access API support
      const supportsFS = 'showDirectoryPicker' in window;
      let dirHandle: any = null;
      let fileHandle: any = null;
      let writable: any = null;

      if (supportsFS) {
        try {
          // @ts-ignore - File System Access API types might be missing
          dirHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
          fileHandle = await dirHandle.getFileHandle(`job_${jobId}_artifacts.zip`, { create: true });
          writable = await fileHandle.createWritable();
        } catch (err) {
          // User cancelled or permission denied
          if (err instanceof Error && err.name === 'AbortError') {
            setState('idle');
            return;
          }
          console.warn('FS Access API failed, falling back to memory/blob', err);
          // Fallback to memory if FS fails (or we could error out if strict)
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

      if (writable) {
        // Stream directly to disk
        // We need to tee the stream: one branch to disk, one to blob for immediate viewing?
        // Actually, if we save to disk, we should read back from disk to avoid double memory usage.
        // But for now, let's keep it simple: 
        // If we have FS access, we pipe to disk AND we need the data for the viewer.
        // Reading back from the file handle is cleaner than teeing into memory.
        
        await response.body.pipeThrough(progressStream).pipeTo(writable);
        
        // Read back from disk
        const file = await fileHandle!.getFile();
        blob = file;
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

      for (const entry of entries) {
        if (entry.directory) continue;

        // @ts-ignore - zip.js types might be slightly off for Entry
        if (entry.filename.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
          // @ts-ignore
          const videoBlob = await entry.getData?.(new zip.BlobWriter());
          if (videoBlob) {
            foundVideo = new File([videoBlob], entry.filename, { type: 'video/mp4' });
          }
        } else {
          // Extract other files for detection (JSON, VTT, RTTM)
          // @ts-ignore
          const fileBlob = await entry.getData?.(new zip.BlobWriter());
          if (fileBlob) {
            candidateFiles.push(new File([fileBlob], entry.filename));
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
        foundAnnotations = {
          metadata: {
            created: new Date().toISOString(),
            version: '1.0.0',
            pipelines: [],
            source: 'videoannotator',
            // @ts-ignore - Partial metadata for fallback
            video_info: {
              width: 0,
              height: 0,
              fps: 30,
              duration: 0,
              frame_count: 0
            }
          },
          annotations: {
            pose: [],
            faces: [],
            scenes: [],
            speakers: [],
            subtitles: []
          }
        };
      }

      setVideoFile(foundVideo);
      setAnnotationData(foundAnnotations);
      setState('ready');

    } catch (err) {
      console.error('Download failed:', err);
      setError(err instanceof Error ? err.message : 'Download failed');
      setState('error');
    }
  }, []);

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
