# Image Contract

- `ghcr.io/infantlab/videoannotator:<version>` and `:latest`: the **slim** app image (core, no
  pipelines). What the launcher runs. Pipelines are installed from the viewer as needed and restored
  after every update or recreation (research.md R7). Built and pushed by CI on release tags (and
  `latest` from the default branch), with `GITHUB_TOKEN`. Docker Hub stays a mirror if its secrets
  exist. An every-pipeline variant (`EXTRAS=all`) stays buildable for labs; it is not published for
  researchers.
- OCI-compliant, `linux/amd64`. Pullable unchanged by Docker and Podman by its fully qualified name.
- Runs as root inside; listens on 18011. Volumes: `/app/models` (weights), `/app/database` and
  `/app/storage` (activity and settings), unchanged; new `/app/cache` (download cache, with
  `UV_CACHE_DIR=/app/cache/uv`). `/app/launcher/requests` is created empty for the launcher's
  requests folder.
