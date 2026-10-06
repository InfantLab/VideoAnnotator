# Docker Contract: Videos and Results Where You Expect Them

The documented Docker setup, as a researcher configures it (FR-011, FR-012, FR-031). It applies to
the compose services `videoannotator-prod` and `videoannotator-gpu`, and to the `docker run`
commands in `docs/installation/INSTALLATION.md`.

## What the researcher sets

| Variable (host) | Default | Meaning |
|---|---|---|
| `VIDEOS_DIR` | `./videos` | Your video folder. Mounted read-only; VideoAnnotator never changes it. |
| `RESULTS_DIR` | `~/VideoAnnotator` | Where results go, on your machine. |

```bash
VIDEOS_DIR=~/Studies RESULTS_DIR=~/VideoAnnotator docker compose up videoannotator-prod
```

## What the setup does

```yaml
ports:
  - "127.0.0.1:18011:18011"          # this machine only (was "18011:18011")
volumes:
  - ${VIDEOS_DIR:-./videos}:/videos:ro
  - ${RESULTS_DIR:-~/VideoAnnotator}:/results
  # existing volumes unchanged: models, database, storage, ./data:/app/data:ro, ./output
environment:
  - VIDEOANNOTATOR_INGEST_ROOTS=/videos
  - VIDEOANNOTATOR_RESULTS_DIR=/results
  - VIDEOANNOTATOR_PUBLISHED_LOCALLY=1
  - VIDEOANNOTATOR_HOST_PATHS=/videos=${VIDEOS_DIR:-./videos};/results=${RESULTS_DIR:-~/VideoAnnotator}
```

`docker run` equivalent:

```bash
docker run --rm -p 127.0.0.1:18011:18011 \
  -v "$HOME/Studies":/videos:ro -v "$HOME/VideoAnnotator":/results \
  -e VIDEOANNOTATOR_INGEST_ROOTS=/videos -e VIDEOANNOTATOR_RESULTS_DIR=/results \
  -e VIDEOANNOTATOR_PUBLISHED_LOCALLY=1 \
  -e VIDEOANNOTATOR_HOST_PATHS="/videos=$HOME/Studies;/results=$HOME/VideoAnnotator" \
  … image
```

## Guarantees and their limits

- With the port published on `127.0.0.1`, only this machine can reach the server. So
  `VIDEOANNOTATOR_PUBLISHED_LOCALLY=1` is true, and every caller counts as the same machine.
- **If you publish the port more widely** (e.g. `18011:18011` to share with a colleague), you MUST
  remove `VIDEOANNOTATOR_PUBLISHED_LOCALLY`. Otherwise anyone who can reach the port can read files
  under `/videos`. The server warns about this at startup whenever the variable is set.
- **Docker can't open folders on your computer**, so `can_open_folders` is false. The viewer shows
  the host path from `VIDEOANNOTATOR_HOST_PATHS`, with a copy button.
- **No video folder** (`/videos` missing or empty): `can_read_in_place` is false, with a `reason`
  saying how to set `VIDEOS_DIR`, and upload keeps working (FR-012).
- **Results directory ownership** (Linux): the container runs as root, so files in `RESULTS_DIR`
  are owned by root on the host. This is documented, with the `--user "$(id -u):$(id -g)"` option.
  Changing the image's user is out of scope.
