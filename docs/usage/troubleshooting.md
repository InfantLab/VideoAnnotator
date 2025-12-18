# Troubleshooting

This page covers common runtime issues when using VideoAnnotator via Docker Compose (recommended) or locally via `uv`.

## Quick Checks

1. **Server reachable**:

```bash
curl -sS http://localhost:18011/health
```

2. **Docker Compose service running** (Compose workflow):

```bash
docker compose ps
docker compose logs -n 200 videoannotator
```

3. **API docs page**:

- http://localhost:18011/docs

## Common Issues

### Cannot connect to the API (connection refused)

**Compose**:

```bash
docker compose up -d --build
docker compose logs -n 200 videoannotator
```

**GPU Compose** (avoid running both CPU and GPU services on the same host port):

```bash
docker compose --profile gpu up -d --build videoannotator-gpu
docker compose logs -n 200 videoannotator-gpu
```

**Local (advanced)**:

```bash
uv sync
uv run videoannotator
```

### Port 18011 already in use

**Compose**:

```bash
docker compose down
```

If something else is listening on 18011:

```bash
lsof -i :18011
```

**Local** (use a different port):

```bash
uv run videoannotator server --port 18012
```

### 401 Unauthorized (missing/invalid API key)

VideoAnnotator requires an API token by default.

**Generate a token**:

- Local: `uv run videoannotator generate-token`
- Compose: `docker compose exec videoannotator newtoken`

**Use it in requests**:

```bash
export API_KEY="va_api_xxx..."

curl -sS http://localhost:18011/api/v1/jobs \
	-H "X-API-Key: $API_KEY"
```

For local-only testing, you can run in dev mode (disables auth, allows all CORS origins):

```bash
uv run videoannotator --dev
```

### Browser CORS errors

- If you are using a local dev client, prefer `--dev` for development.
- For non-dev mode, configure allowed origins via `CORS_ORIGINS`.

See the authentication and CORS docs:

- [docs/security/authentication.md](../security/authentication.md)
- [docs/security/cors.md](../security/cors.md)

## More Help

For installation issues (FFmpeg, Python, GPU/CUDA, etc.), see:

- [docs/installation/troubleshooting.md](../installation/troubleshooting.md)
