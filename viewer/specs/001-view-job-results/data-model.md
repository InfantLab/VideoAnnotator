# Data Model: View Job Results

**Feature**: View Job Results (001-view-job-results)

## Entities

### JobArtifacts
Represents the structure of the downloaded ZIP file content.

| Field | Type | Description |
|---|---|---|
| `video.mp4` | File (Video) | The source video file. |
| `results.json` | JSON | Combined pipeline results (annotations). |
| `scene_detection.json` | JSON | (Optional) Scene detection specific output. |
| `person_tracking.json` | JSON | (Optional) Person tracking specific output. |

### JobViewerState
Represents the UI state of the viewer page.

| State | Description |
|---|---|
| `IDLE` | Initial state, waiting for user action (or auto-start). |
| `SELECTING_DIR` | User is choosing a destination folder (FS Access API). |
| `DOWNLOADING` | Artifacts are being fetched and saved to disk. |
| `UNZIPPING` | Artifacts are being extracted (if necessary/possible). |
| `READY` | Data is loaded and `VideoAnnotationViewer` is active. |
| `ERROR` | An error occurred (network, permission, disk). |

### DownloadProgress
| Field | Type | Description |
|---|---|---|
| `bytesLoaded` | number | Bytes received so far. |
| `totalBytes` | number | Total bytes expected (Content-Length). |
| `percentage` | number | Calculated percentage (0-100). |

## Relationships

-   `JobResultsViewer` manages `JobViewerState`.
-   `JobResultsViewer` consumes `JobArtifacts` (as File handles) and passes them to `VideoAnnotationViewer`.

## Validation Rules

-   **Video File**: Must be present in the ZIP (or accessible in the folder).
-   **Results JSON**: Must be valid JSON and conform to `StandardAnnotationData` schema.
