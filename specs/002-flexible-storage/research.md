# Research: Flexible Storage & Artifact Access

**Feature**: Flexible Storage & Artifact Access
**Date**: 2025-12-08

## 1. Storage Abstraction

### Decision
Implement a custom `StorageProvider` Abstract Base Class (ABC) in `src/videoannotator/storage/providers/base.py`.

### Rationale
- **Simplicity**: A custom ABC is lightweight and tailored exactly to our needs (save, load, list, get_url).
- **Control**: We avoid the heavy dependency of `fsspec` for now, but can wrap `fsspec` in a provider later if needed.
- **Extensibility**: Easy to implement `LocalStorageProvider` now and `S3StorageProvider` later.

### Alternatives Considered
- **fsspec**: Standard in data science, but adds dependency weight. Better to wrap it later than depend on it directly now.
- **pathlib.Path**: Good for local, but doesn't abstract cloud storage well (S3 paths aren't real paths).

## 2. FastAPI File Streaming (ZIP Download)

### Decision
Use a generator function with `zipfile` writing to an in-memory buffer (or `zipstream-ng` if added) to stream the ZIP archive. For MVP, we will use a **temporary file** approach if streaming proves too complex without extra dependencies, but aim for streaming.

*Update*: We will use a generator that yields chunks of files. Since standard `zipfile` requires seeking to write the central directory, true streaming without temp files is hard with stdlib.
**Refined Decision**: Create a temporary ZIP file in `/tmp`, stream it, then delete it. This is robust and uses standard libraries.

### Rationale
- **Robustness**: Standard `zipfile` is reliable.
- **Simplicity**: "True" streaming of ZIPs is complex (requires Zip64 or data descriptors). Temp file is a safe MVP trade-off.

### Alternatives Considered
- **zipstream-ng**: Adds dependency.
- **On-the-fly compression**: Complex to implement correctly with `zipfile` stdlib.

## 3. Config Integration

### Decision
Add a `storage` section to the main YAML configuration.

### Structure
```yaml
storage:
  root_path: "./storage/jobs"  # Default
  # Future: provider: "s3", bucket: "..."
```

### Rationale
- **Consistency**: Fits the existing YAML structure.
- **Clarity**: Explicit `root_path` is easy to understand.
