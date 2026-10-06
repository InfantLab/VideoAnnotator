# Image Contract

- `ghcr.io/infantlab/videoannotator:<version>-all` and `:latest-all`: every pipeline; what the
  launcher runs. `:<version>` and `:latest`: slim, for labs. Built and pushed by CI on release tags
  (and `latest*` from the default branch), with `GITHUB_TOKEN`. Docker Hub stays a mirror if its
  secrets exist.
- OCI-compliant, `linux/amd64`. Pullable unchanged by Docker and Podman by its fully qualified name.
- Runs as root inside; listens on 18011; volumes `/app/models`, `/app/database`, `/app/storage`
  (unchanged); `/app/launcher/requests` is created empty for the launcher's requests folder.
