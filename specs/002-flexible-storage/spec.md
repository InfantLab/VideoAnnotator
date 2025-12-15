# Feature Specification: Flexible Storage & Artifact Access

**Feature Branch**: `002-flexible-storage`
**Created**: 2025-12-08
**Status**: Draft
**Input**: User description: "Now that the completed pipelines are working between server and client. we want to find the best way for the client to be able to easily view the videos and theit annotations together. I think this might involve a rethink of how data is stored. At present I believe our pipeline process copies videos into the server container in a Storage/jobs/<jobid> folder and stores annotations in same location. this is currently not accessible to client directly. As a first step we need to allow download of all annotations. In long run I don't think the server container is the best location to accumulate all this data. So we need to be able to support being able to specify a stand-alone storage location wiith some flexibility. Initially we want to be able to support a storage location We should investigate the best way to do this (what do other similar annotation tools do). As a first remote location. How easy is it support accessing a onedrive instance which the client could also access?"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Download Job Annotations (Priority: P1)

As a researcher using the client, I want to download all generated annotations for a specific job so that I can view them locally alongside the video or use them in other tools.

**Why this priority**: This addresses the immediate pain point of data being "locked" in the server container. It provides immediate value without requiring a full storage re-architecture.

**Independent Test**: Can be fully tested by running a pipeline, then using the CLI/API to download the resulting annotations and verifying the files exist locally.

**Acceptance Scenarios**:

1. **Given** a completed job with ID `job-123` containing annotations (e.g., `.json`, `.srt`), **When** I run `videoannotator job download-annotations job-123`, **Then** the system downloads a ZIP file or folder containing all annotation files to my current directory.
2. **Given** a job ID that does not exist, **When** I attempt to download annotations, **Then** the system returns a clear "Job not found" error.

---

### User Story 2 - Configurable Local Storage Root (Priority: P2)

As a system administrator or advanced user, I want to configure the root directory where VideoAnnotator stores job data so that I can use a specific disk partition, external drive, or mounted network share (like a mounted OneDrive folder).

**Why this priority**: This enables the "stand-alone storage location" requirement by allowing the user to point the system to a location outside the container's ephemeral storage (via volume mounts).

**Independent Test**: Can be tested by changing the config, running a new job, and verifying data appears in the new location.

**Acceptance Scenarios**:

1. **Given** a configuration file specifying `storage.root_path = /mnt/external_drive`, **When** I submit a new job, **Then** the video and generated annotations are stored under `/mnt/external_drive/jobs/<jobid>`.

---

### User Story 3 - Pluggable Storage Architecture (Priority: P3)

As a developer, I want the storage logic to be abstracted behind an interface so that we can easily add support for cloud providers (S3, OneDrive API) in the future without rewriting the core pipeline logic.

**Why this priority**: Sets the foundation for the "long run" goal of flexible remote storage.

**Independent Test**: Code review verification that file I/O is performed through an abstraction layer, not direct `open()` calls on hardcoded paths.

**Acceptance Scenarios**:

1. **Given** the codebase, **When** I inspect the job management logic, **Then** the system architecture supports swapping the storage backend without changing core logic.

### Edge Cases

- What happens when the configured storage location is full?
- How does the system handle permissions errors when writing to the storage location?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide an API endpoint to retrieve all annotation artifacts for a given job ID (e.g., as a ZIP archive).
- **FR-002**: The CLI MUST provide a command `videoannotator job download-annotations <job_id> [output_path]` to save annotations locally.
- **FR-003**: The system MUST allow configuration of the base storage directory via the main configuration file.
- **FR-004**: The system MUST use a storage abstraction layer for all job artifact I/O operations.
- **FR-005**: The default storage implementation MUST support the local filesystem.
- **FR-006**: The system MUST rely on OS-level mounting for remote storage access in this iteration; native cloud provider integration is out of scope.
- **FR-007**: When the storage location is changed, the system MUST only access jobs in the currently configured location; access to jobs in previous locations is not required.

### Key Entities

- **JobArtifact**: Represents a file generated by a pipeline (video, annotation, report).
- **StorageProvider**: Abstract interface for file operations (save, load, list, get_url).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can download all annotations for a job via CLI in a single command.
- **SC-002**: Changing the storage root in config directs all new job data to the new path.
- **SC-003**: The system can successfully read/write job data to a mounted external volume (simulating a "stand-alone storage location").
