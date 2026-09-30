# Tasks: View Job Results

**Feature Branch**: `001-view-job-results`
**Status**: Planning

## Dependencies

- **Phase 1 (Setup)**: No dependencies.
- **Phase 2 (Foundational)**: Depends on Phase 1.
- **Phase 3 (US1)**: Depends on Phase 2.
- **Phase 4 (US2)**: Depends on Phase 3.
- **Phase 5 (Polish)**: Depends on Phase 4.

## Phase 1: Setup

**Goal**: Initialize project dependencies and structure.

- [x] T001 Install @zip.js/zip.js dependency in package.json
- [x] T002 Create directory structure for new components in src/pages and src/hooks

## Phase 2: Foundational

**Goal**: Update API client and create shared utilities.

- [x] T003 Update API client to include artifacts endpoint in src/api/client.ts
- [x] T004 Create DownloadProgress component in src/components/DownloadProgress.tsx
- [x] T005 Create useZipDownloader hook shell in src/hooks/useZipDownloader.ts

## Phase 3: User Story 1 - Launch Viewer from Job List

**Goal**: Enable users to launch the viewer and download/view results (Core Flow).

**Independent Test**:
1. Navigate to Jobs list.
2. Click "View Results" on a completed job.
3. Verify viewer opens and loads video/annotations.

- [x] T006 [US1] Add "View Results" button to Job List in src/pages/CreateJobs.tsx
- [x] T007 [US1] Add "View Results" button to Job Detail in src/pages/CreateJobDetail.tsx
- [x] T008 [US1] Create JobResultsViewer page component in src/pages/JobResultsViewer.tsx
- [x] T009 [US1] Add route for /view/:jobId in src/App.tsx
- [x] T010 [US1] Implement basic download logic in src/hooks/useZipDownloader.ts
- [x] T011 [US1] Implement unzip logic using zip.js in src/hooks/useZipDownloader.ts
- [x] T012 [US1] Integrate VideoAnnotationViewer in src/pages/JobResultsViewer.tsx

## Phase 4: User Story 2 - Local Data Storage

**Goal**: Ensure robust local storage with File System Access API and fallbacks.

**Independent Test**:
1. Click "View Results".
2. Select local directory when prompted.
3. Verify files are saved to disk.

- [x] T013 [US2] Implement File System Access API integration in src/hooks/useZipDownloader.ts
- [x] T014 [US2] Implement fallback for non-supported browsers in src/hooks/useZipDownloader.ts
- [x] T015 [US2] Add error handling for permission denial in src/pages/JobResultsViewer.tsx
- [x] T016 [US2] Add progress tracking integration in src/pages/JobResultsViewer.tsx

## Phase 5: Polish & Cross-Cutting Concerns

**Goal**: Finalize UX and handle edge cases.

- [x] T017 Add error boundary for viewer in src/components/ErrorBoundary.tsx
- [x] T018 Optimize large file handling (stream piping) in src/hooks/useZipDownloader.ts
- [x] T019 Verify accessibility of new buttons and progress indicators

## Implementation Strategy

- **MVP**: Focus on US1 with basic download (even if just to memory for small files initially) to prove the flow.
- **Incremental**: Add FS Access API (US2) to handle the 2GB requirement robustly.
- **Parallel**: UI components (Buttons, Progress) can be built while the Hook logic is being developed.
