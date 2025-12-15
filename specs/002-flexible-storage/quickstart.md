# Quickstart: Flexible Storage

## Configuration

To change the storage location, edit your configuration file (e.g., `configs/default.yaml`):

```yaml
storage:
  root_path: "/mnt/external_drive/videoannotator_jobs"
```

## Downloading Annotations

### CLI

Download all annotations for a job to your current directory:

```bash
videoannotator job download-annotations <job_id>
```

Specify an output directory:

```bash
videoannotator job download-annotations <job_id> --output ./my_results/
```

### API

Download via curl:

```bash
curl -O -J http://localhost:8000/api/v1/jobs/<job_id>/artifacts
```
