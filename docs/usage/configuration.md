# Configuration Guide

**Version**: v1.3.0+
**Status**: Production Ready

VideoAnnotator supports a flexible configuration system using YAML files and environment variables.

## Configuration File (`configs/default.yaml`)

The primary configuration file is located at `configs/default.yaml`. This file defines the default settings for the application, including storage, pipelines, and API behavior.

### Storage Configuration

The `storage` section controls where job artifacts (videos, annotations, reports) are stored.

```yaml
storage:
  # Root directory for persistent job storage
  # Can be absolute or relative to the workspace root
  root_path: "./storage/jobs"

  # Storage provider type (currently only 'local' is supported)
  provider: "local"
```

### Overriding Configuration

You can override configuration values using environment variables. Environment variables take precedence over the YAML configuration.

| YAML Key | Environment Variable | Description |
|----------|----------------------|-------------|
| `storage.root_path` | `STORAGE_ROOT` | Root directory for job storage |

## Storage Structure

When using the local storage provider, the directory structure is organized by job ID:

```
storage/jobs/
├── <job_id_1>/
│   ├── video.mp4          # Original video file
│   ├── results.json       # Pipeline results
│   ├── annotations.vtt    # Generated annotations
│   └── processing.log     # Job processing log
├── <job_id_2>/
│   └── ...
```

## Advanced Configuration

For advanced use cases, you can create custom configuration files (e.g., `configs/production.yaml`) and load them by modifying the application entry point or using future CLI flags (planned for v1.4.0).
