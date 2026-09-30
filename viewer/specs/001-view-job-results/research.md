# Research: View Job Results

**Feature**: View Job Results (001-view-job-results)
**Date**: 2025-12-09

## 1. Client-side ZIP Library

**Decision**: `@zip.js/zip.js`

**Rationale**:
-   **Streaming Support**: Native support for Streams API is critical for handling large files (up to 2GB) without crashing the browser tab due to memory exhaustion.
-   **File System Access API Integration**: Can read/write directly to `FileSystemHandle` objects, minimizing memory footprint.
-   **Performance**: Efficient processing of large archives compared to `jszip` which often requires loading the entire file into memory.

**Alternatives Considered**:
-   **JSZip**: Popular and easy to use, but generally requires loading the entire file into memory or complex workarounds for large files. Not suitable for 2GB+ archives in a browser context.
-   **fflate**: Very fast and lightweight, but `zip.js` offers better high-level abstractions for the specific combination of Streams and File System Access API needed here.

## 2. File System Access API & Fallbacks

**Decision**: Use `window.showDirectoryPicker()` with a graceful fallback to standard browser download.

**Rationale**:
-   **Primary Flow (Chrome/Edge)**: `showDirectoryPicker` allows the user to select a folder. We can then stream the download directly to a file in this folder using `FileSystemWritableFileStream`. This avoids memory limits.
-   **Fallback Flow (Firefox/Safari)**: These browsers do not support `showDirectoryPicker`. The fallback is to trigger a standard browser download of the ZIP file. The user must then unzip it manually or we can try to unzip in-memory if the file is small enough (but for 2GB, manual unzip is safer). *Correction based on Spec*: The spec FR-005 explicitly says "download to default folder and require manual open".

**Implementation Pattern**:
```typescript
// Check support
const supportsFS = 'showDirectoryPicker' in window;

if (supportsFS) {
  const dirHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
  const fileHandle = await dirHandle.getFileHandle('job_artifacts.zip', { create: true });
  const writable = await fileHandle.createWritable();
  // Stream download to writable
  await response.body.pipeTo(writable);
} else {
  // Fallback: Standard download
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'job_artifacts.zip';
  a.click();
}
```

## 3. Routing & Component Structure

**Decision**: Create a new `JobResultsViewer` page component at `/view/:jobId`.

**Rationale**:
-   **Separation of Concerns**: The existing `VideoAnnotationViewer` is a presentation component. The new `JobResultsViewer` will handle the "smart" logic of fetching, downloading, and unzipping data before passing it to the presentation component.
-   **Routing**: Fits naturally into the existing `react-router-dom` setup in `App.tsx`.

**Structure**:
-   `src/pages/JobResultsViewer.tsx`:
    -   Extracts `jobId` from URL.
    -   Manages state: `idle` | `downloading` | `unzipping` | `ready` | `error`.
    -   Renders `VideoAnnotationViewer` when `ready`.
    -   Renders `DownloadProgress` component when `downloading`.

## 4. Progress Tracking

**Decision**: Use a `TransformStream` to intercept the fetch stream for progress calculation.

**Rationale**:
-   `fetch` API doesn't provide native progress events.
-   A `TransformStream` allows us to count bytes as they pass through the stream to the disk writer, updating a progress state without buffering the data.

**Implementation Pattern**:
```typescript
const progressStream = new TransformStream({
  transform(chunk, controller) {
    loaded += chunk.length;
    onProgress(loaded / total);
    controller.enqueue(chunk);
  }
});
await response.body.pipeThrough(progressStream).pipeTo(writable);
```
