#!/usr/bin/env bash
# Warn when the dev container's workspace is a Windows folder bind-mounted
# through Docker Desktop. Every file access then crosses the Windows<->WSL file
# bridge and is scanned by Defender, which makes tests and model loading slow
# (docs/installation/troubleshooting.md). Warns only; never fails.
#
# Usage: check_workspace_mount.sh [workspace_dir] [mounts_file]
# Set VIDEOANNOTATOR_SKIP_MOUNT_CHECK=1 to silence it.

workspace="${1:-$PWD}"
mounts="${2:-/proc/mounts}"

[ "${VIDEOANNOTATOR_SKIP_MOUNT_CHECK:-0}" = "1" ] && exit 0
[ -r "$mounts" ] || exit 0

# The mount the workspace lives on: the longest mount point that prefixes it.
# Docker Desktop shows a Windows drive as fstype 9p with "aname=drvfs" in its
# options (e.g. "C:\134 /workspaces/X 9p rw,aname=drvfs;path=C:\;..."); a named
# volume or a folder inside the WSL distro shows as ext4.
read -r fstype options < <(
    awk -v ws="$workspace" '
        { mp = $2; gsub(/\\040/, " ", mp) }
        (ws == mp || index(ws, mp == "/" ? "/" : mp "/") == 1) && length(mp) > best {
            best = length(mp); fs = $3; opts = $4
        }
        END { if (best) print fs, opts }
    ' "$mounts"
)

if [ "$fstype" = "9p" ] && [[ "$options" == *aname=drvfs* ]]; then
    cat >&2 <<'EOF'

  ======================================================================
  WARNING: this workspace is a Windows folder mounted into the container.

  Tests, imports and model loading are slow here: every file read
  crosses the Windows/WSL file bridge and Defender scans each one.

  For speed, use "Dev Containers: Clone Repository in Container Volume..."
  instead. See docs/installation/INSTALLATION.md, "Dev Container (VS Code)".

  (Set VIDEOANNOTATOR_SKIP_MOUNT_CHECK=1 to hide this message.)
  ======================================================================

EOF
fi
exit 0
