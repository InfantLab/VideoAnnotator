# Accessing Results

This page describes where VideoAnnotator stores outputs and how to retrieve them using the CLI or HTTP API.

## Where results are stored

### Docker Compose (recommended)

The default `docker-compose.yml` mounts these host folders into the container:

- `./data` → `/app/data` (input videos)
- `./output` → `/app/output` (job artifacts and exported results)
- `./logs` → `/app/logs` (server and job logs)
- `./database` → `/app/database` (SQLite database)

In practice:

- Look for generated files under `./output/` on your host machine.
- If you need to inspect container paths directly: `docker compose exec videoannotator ls -la /app/output`.

### Local install

When running locally, the output directory is controlled by your configuration and/or storage settings. If you are using the default local setup, results typically land under `./output/` or `./storage/` depending on configuration.

## Retrieve results with the CLI

All examples assume the API server is running on `http://localhost:18011`.

### Check job status

```bash
uv run videoannotator job status <job_id>
```

### Get job results (JSON)

```bash
uv run videoannotator job results <job_id>
```

### Download all job artifacts (ZIP)

```bash
uv run videoannotator job download-annotations <job_id> --output ./results/
```

### If you are running via Docker Compose

```bash
docker compose exec videoannotator va job results <job_id>
docker compose exec videoannotator va job download-annotations <job_id> --output /app/output
```

## Retrieve results with the HTTP API

```bash
export API_KEY="va_api_your_key_here"

# Get job status
curl -H "Authorization: Bearer $API_KEY" \
	"http://localhost:18011/api/v1/jobs/{job_id}"

# Get job results
curl -H "Authorization: Bearer $API_KEY" \
	"http://localhost:18011/api/v1/jobs/{job_id}/results"

# Download all artifacts (ZIP)
curl -H "Authorization: Bearer $API_KEY" \
	"http://localhost:18011/api/v1/jobs/{job_id}/artifacts" \
	--output "job_{job_id}_artifacts.zip"
```
