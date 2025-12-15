# Feature Specification: View Job Results

**Feature Branch**: `001-view-job-results`
**Created**: 2025-12-09
**Status**: Draft
**Input**: User description: "view-results"

## User Scenarios & Testing

### User Story 1 - Launch Viewer from Job List (Priority: P1)

As a user, I want to launch the Video Annotation Viewer directly from a completed job in the pipeline list, so that I can inspect the results immediately without manually managing files.

**Why this priority**: Connects the job processing workflow with the visualization tool, streamlining the user experience.

**Independent Test**:
1.  Navigate to the Jobs list.
2.  Find a "completed" job.
3.  Click the "View Results" button.
4.  Verify the Viewer opens and loads the correct video and annotations.

**Acceptance Scenarios**:

1.  **Given** a job is "completed", **When** I view the job list, **Then** a "View Results" button is available.
2.  **Given** I click "View Results", **When** the viewer loads, **Then** it initiates the data retrieval process.
3.  **Given** the data is retrieved, **When** the viewer is ready, **Then** the video plays and annotations are displayed.

### User Story 2 - Local Data Storage (Priority: P2)

As a user, I want to specify a local folder to store the downloaded job artifacts, so that I can access them later and the viewer can read them from my local disk.

**Why this priority**: Ensures data persistence and handles the requirement to "download data... to use locally".

**Independent Test**:
1.  Click "View Results".
2.  When prompted, select a local directory.
3.  Verify the job artifacts (ZIP content) are saved to that directory.
4.  Verify the Viewer reads the video and JSON files from that directory.

**Acceptance Scenarios**:

1.  **Given** I initiate the view process, **When** the browser supports it, **Then** I am prompted to select a destination folder.
2.  **Given** a folder is selected, **When** the download completes, **Then** the files are saved to disk.
3.  **Given** the files are saved, **When** the viewer renders, **Then** it uses the local file handles.

### Edge Cases

-   **Network Failure**: If the download is interrupted, the system MUST notify the user and allow retrying.
-   **Corrupt/Invalid Artifacts**: If the downloaded ZIP is corrupt or missing essential files (video/annotations), the Viewer MUST display an error message.
-   **Permission Denied**: If the user denies permission to the File System Access API, the system MUST fall back to the default download behavior (FR-005).
-   **Large Files**: The system MUST handle large video files (up to 2GB) without crashing the browser tab during the unzip/load process.

## Requirements

### Functional Requirements

-   **FR-001**: The Job List and Job Detail pages MUST display a "View Results" button for jobs with "completed" status.
-   **FR-002**: The application MUST provide a routing mechanism (e.g., `/view/:jobId`) to initialize the viewer with a job context.
-   **FR-003**: The system MUST fetch job artifacts from the server API (`/api/v1/jobs/{job_id}/artifacts`).
-   **FR-004**: The system MUST attempt to save the downloaded artifacts to a user-selected local directory using the File System Access API.
-   **FR-005**: The system MUST handle the case where the File System Access API is not supported by downloading the artifacts ZIP to the browser's default download location and instructing the user to manually open the file in the Viewer.
-   **FR-006**: The Viewer MUST support loading data from the downloaded file handles/blobs.
-   **FR-007**: The system MUST display download progress to the user.
-   **FR-008**: The system MUST unzip the downloaded artifacts client-side to access the individual video and annotation files.

### Key Entities

-   **Job Artifacts**: The ZIP file containing `video.mp4`, `results.json`, and other output files.
-   **Local Directory Handle**: The reference to the user-selected folder for storing results.

### Success Criteria

-   **Performance**: Users can launch the viewer and start downloading artifacts within 2 seconds of clicking "View Results".
-   **Reliability**: 100% of valid job artifact ZIPs are correctly unzipped and loaded into the viewer.
-   **Usability**: Users on supported browsers (Chrome/Edge) can save results to a specific folder without leaving the application flow.
-   **Resilience**: The system gracefully handles network errors and invalid files with clear user feedback.

## Assumptions

-   The `/api/v1/jobs/{job_id}/artifacts` endpoint returns a ZIP file containing the video file and all annotation JSONs.
-   The user is primarily using a browser with File System Access API support (Chrome/Edge) for the intended "smooth" experience.
-   The client-side unzipping library (e.g., `jszip` or `fflate`) can handle the artifact ZIP size.

