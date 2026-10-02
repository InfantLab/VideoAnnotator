"""scripts/check_workspace_mount.sh warns only for a Windows-drive workspace.

The Windows line is copied from /proc/mounts in a dev container opened with
"Reopen in Container" on Docker Desktop for Windows (2026-10-02); the ext4
lines are named volumes on the same host.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_workspace_mount.sh"
WS = "/workspaces/VideoAnnotator"

WINDOWS_BIND = (
    r"C:\134 /workspaces/VideoAnnotator 9p rw,noatime,aname=drvfs;path=C:\;"
    r"uid=0;gid=0;metadata;symlinkroot=/mnt/host/,cache=5,access=client,"
    "msize=65536,trans=fd,rfd=5,wfd=5 0 0"
)
VOLUME = "/dev/sdd /workspaces ext4 rw,relatime 0 0"
ROOT = "overlay / overlay rw,relatime 0 0"

# The script runs inside the Linux dev container. On Windows `bash` may be WSL's
# launcher, which fails when no distro is installed (seen on GitHub's runners).
pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or sys.platform == "win32",
    reason="needs a POSIX bash",
)


def run(tmp_path, mounts, workspace=WS, **env):
    table = tmp_path / "mounts"
    table.write_text("\n".join(mounts) + "\n")
    result = subprocess.run(
        ["bash", str(SCRIPT), workspace, str(table)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env},
    )
    assert result.returncode == 0
    return result.stderr


def test_warns_for_a_windows_drive_bind_mount(tmp_path):
    assert "Clone Repository in Container Volume" in run(tmp_path, [ROOT, WINDOWS_BIND])


def test_quiet_in_a_container_volume(tmp_path):
    assert run(tmp_path, [ROOT, VOLUME]) == ""


def test_quiet_for_a_named_volume_inside_a_windows_workspace(tmp_path):
    node_modules = f"/dev/sdd {WS}/viewer/node_modules ext4 rw,relatime 0 0"
    assert (
        run(tmp_path, [ROOT, WINDOWS_BIND, node_modules], f"{WS}/viewer/node_modules")
        == ""
    )


def test_matches_whole_path_components(tmp_path):
    assert run(tmp_path, [ROOT, WINDOWS_BIND], workspace=WS + "2") == ""


def test_quiet_for_linux_and_macos_hosts(tmp_path):
    linux = f"/dev/nvme0n1p2 {WS} ext4 rw,relatime 0 0"
    macos = f"fakeowner {WS} fakeowner rw,nosuid,nodev,relatime 0 0"
    assert run(tmp_path, [ROOT, linux]) == ""
    assert run(tmp_path, [ROOT, macos]) == ""


def test_can_be_silenced(tmp_path):
    assert (
        run(tmp_path, [ROOT, WINDOWS_BIND], VIDEOANNOTATOR_SKIP_MOUNT_CHECK="1") == ""
    )
