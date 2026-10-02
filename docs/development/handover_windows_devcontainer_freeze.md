# Handover: Windows devcontainer freezes the host during test runs

From: host-side investigation session (Claude Code on Windows, 2026-10-02)
To: the agent working in the VideoAnnotator repo
Owner: Caspar

## The problem in one paragraph

On 2026-10-02 Caspar's Windows laptop (31 GB RAM, 24 threads, Docker Desktop with WSL2) froze completely
while the VideoAnnotator devcontainer was running the regression tests. Even Task Manager wouldn't open, and
it needed a hard power-off. Defender's scan engine (`MsMpEng.exe`) was at the top of the process list. The
cause is how the devcontainer gets at the code. VS Code's default "Reopen in Container" bind-mounts the Windows
folder (`C:\Users\caspar\code\VideoAnnotator`) into the Linux VM, and `.venv` (~66k files), `models/` (16 GB)
and all test output live inside that folder. Every import, weight load and file write crosses the
Windows↔WSL file bridge, and Defender scans each file in real time on the Windows side. That, plus the VM's
own memory use, starved the host. **We want no VideoAnnotator user on Windows to hit this**, not just Caspar.

## Evidence (from host logs, all local BST times)

| Time | Event |
|---|---|
| 08:44:44 | Docker VM kernel: `watchdog: BUG: soft lockup - CPU#22 stuck for 25s` in `page_reporting_process`, the WSL routine that hands freed memory back to Windows |
| 08:45:39 | VM network adapter re-attached (VM networking reset) |
| 08:50:16 | Display-off timeout (5 min on AC). Modern Standby, but the machine was still running |
| 08:57 | VM `systemd-journald: Time jumped backwards`. The host had stopped scheduling the VM |
| 09:06:17 | Last Windows event-log heartbeat |
| 09:14–09:20 | Host scheduled task refused (0x800710E0); last Docker log lines. The machine was thrashing, not dead |
| 09:34 | Forced reboot (Kernel-Power 41, power button) |

- The WSL VM cap was ~15.6 GB (`hv_balloon: Max. dynamic memory size: 15932 MB`). That's WSL's default of half
  the RAM, because the host has no `memory=` set in `.wslconfig`.
- Windows' resource-exhaustion detector (event 2004) did **not** fire, and there were no OOM kills in the VM.
  So "host starved and thrashing" is better supported than "RAM simply ran out".
- There have been 7 dirty shutdowns since 2026-09-03. Not yet checked which of them were devcontainer days.
- Unrelated noise to ignore: fTPM/BitLocker errors (event 24641). That's a known firmware issue on this machine.

**Status of the diagnosis: strongly indicated, not proven.** The bind-mount + Defender explanation fits all the
evidence, but nobody has yet run the same suite from a container-volume clone and seen it stay healthy.
Proving that is part of the job (step 4).

## Update 2026-10-02: no tests were running (container-side evidence)

The agent in the container went looking for the test command, to make step 4 like-for-like, and
found there wasn't one. All times below are BST (UTC+1).

- **Nothing ran.** The Claude session's last command was at 22:06 on 10-01; it resumed at 10:02,
  after the reboot. No file in the workspace was written between those times, and a pytest run
  always writes `.pytest_cache`, `__pycache__` and `logs/`. pytest's temp directories hold only
  post-reboot runs. No other agent or terminal logged anything in the container.
- **The laptop was asleep.** The Claude Code extension's log in the container
  (`/root/.vscode-server/data/logs/20261001T095631/exthost1/Anthropic.claude-code/Claude VSCode.log`)
  records "clock jumps": spans where the VM did not run at all. They were 1.6 h from about 01:30,
  2.3 h from 03:15, and 3.0 h from 05:35, with brief wakes between.
- **It froze on waking.** At 08:39 the VM resumed. The extension's event loop was blocked for
  42 s, then 27 s, and the `ptyHost` stopped answering heartbeats. At 08:44 the VM kernel's soft
  lockup in `page_reporting_process` followed (the host log above). That is the routine returning
  freed memory to Windows, holding a VM of up to ~15.6 GB.

**Revised diagnosis: resume from sleep with a large, idle Docker VM**, not file-bridge traffic
from a test run. Defender at the top of the list fits a machine catching up after waking. This is
not proven either. There are public reports of WSL2 hangs after sleep with Docker Desktop, with
Resource Saver mode named as one cause
([docker/for-win#14656](https://github.com/docker/for-win/issues/14656),
[microsoft/WSL#9429](https://github.com/microsoft/WSL/issues/9429)); none was found naming
`page_reporting_process`.

**What the branch keeps and what changed.** The named volumes for `.venv` and models, the
Windows-drive warning, and the volume-clone and memory-cap docs are kept: the bridge is slow
whatever caused the freeze, and a smaller VM has less memory to hand back on resume. The docs no
longer say a test run froze the machine; they add "stop the container before the machine sleeps"
and a hint about Resource Saver.

**Next evidence (Caspar, on the host).** This replaces step 4's test-suite comparison, which has
no command to repeat.
1. Is Docker Desktop's Resource Saver on (Settings → Resources)? Which Docker Desktop and WSL
   versions (`wsl --version`)?
2. Did the 7 dirty shutdowns since 2026-09-03 follow a night asleep with the container running?
   Kernel-Power 41 times against the Docker VM's last log lines on those days would tell.
3. A controlled repeat: container up (bind mount, as on 10-02), let the laptop sleep, wake it, and
   watch for 15 minutes. Then the same with the `.wslconfig` cap, and/or Resource Saver off.

## What the repo did before this branch

- `.devcontainer/devcontainer.json` has no `workspaceMount`, so it uses the default bind mount of the
  host folder. It already uses named volumes for `/root/.claude`, `gh` config and `viewer/node_modules`.
  `.venv` and `models/` are **not** volumes.
- `VIDEOANNOTATOR_MODELS_DIR=${containerWorkspaceFolder}/models` puts models on the bind mount.
- `UV_LINK_MODE=copy` means `uv sync` writes full copies of every package file into `.venv` on the bind mount.
- `docker-compose.yml` already uses a named `models` volume ("model weights persist across rebuilds (spec 016)").
  The devcontainer should match it.
- `docs/installation/INSTALLATION.md` § "Dev Container (VS Code)" (line ~327) is one line: "use Reopen in
  Container". There's no Windows warning, and no troubleshooting entry.

## What to do

The aim is to make the safe path the default, so a Windows user who does the obvious thing doesn't freeze
their machine.

1. **Move the heavy traffic onto the Linux side in `devcontainer.json`.** This is the change that protects everyone.
   - Add a named volume for `.venv`, e.g.
     `source=videoannotator-venv,target=${containerWorkspaceFolder}/.venv,type=volume`.
   - Add a named volume for models, matching compose, e.g. `videoannotator-models` at the models dir.
   - Consider the same for other heavy write paths the tests use (`storage/`, `test_storage/`, `temp/`,
     `logs/`, `batch_results/`, coverage output). Check which ones the test suite actually writes to
     before adding mounts. Don't guess.
   - Named volumes outlive rebuilds and new containers. They're lost only when explicitly deleted
     (`docker volume rm`, or Docker Desktop's purge/reset). Note: on current Docker, `docker system prune -a --volumes`
     only removes *anonymous* volumes. Caspar's host runs that nightly and his named volumes have survived it.
     Verify this claim against the Docker docs before writing it into user docs.
   - Check `postCreateCommand` (`uv sync --inexact`) still works when `.venv` starts empty in a fresh volume,
     and that volume ownership is fine (the container runs as root).

2. **Warn at container start if the workspace is on a Windows drive.** Add a small script, run from
   `postStartCommand`, that detects a host-bind-mounted workspace on Docker Desktop for Windows and prints a clear
   warning: slow tests, and a risk of freezing the machine. Point it at the docs fix. Work out the detection
   empirically. Check the workspace's filesystem type via `stat -f -c %T` or `/proc/mounts` in this bind-mounted
   container, and again in a volume clone. Don't assume the fs-type string. Warn only; never block startup.

3. **Docs.**
   - `INSTALLATION.md` dev container section: on Windows, recommend **"Dev Containers: Clone Repository in
     Container Volume…"** over cloning to `C:\` and reopening. Say why in one or two sentences.
   - Recommend a host WSL memory cap in `%USERPROFILE%\.wslconfig`, e.g. `[wsl2]` / `memory=12GB` / `swap=8GB`,
     then `wsl --shutdown`. The repo can't set this for users.
   - Mention a Defender exclusion or a Windows Dev Drive for the code folder as optional extras. These are
     user/admin decisions; just document them.
   - A `troubleshooting.md` entry under a symptom-first heading, like "Machine freezes / MsMpEng high during tests
     on Windows". Include how models persist, how to wipe them (`docker volume rm videoannotator-models`), and a
     one-line backup command.
   - CHANGELOG entry.

4. **Prove it (Caspar runs this on the host, the agent prepares it).** Do this after the changes are pushed.
   Caspar does a fresh "Clone Repository in Container Volume" of the branch and runs the same regression suite
   that froze the machine, with Task Manager open. If the host stays responsive, record the result in the
   troubleshooting entry as the evidence. If it still freezes, the diagnosis is wrong: stop and report back
   before doing more.
   The agent should write down the exact test command that was running (ask Caspar if it's not in shell history), so
   the comparison is like-for-like.

## Constraints and cautions

- **Don't run the heavy regression suite from the current bind-mounted container.** That's what froze the host.
  Unit tests for the new script are fine.
- The current working tree has uncommitted changes that aren't Caspar's request to commit or discard
  (`.gitignore`, `.vscode/settings.json` modified; `.claude/settings.json` and a stray `bun` file untracked).
  Ask before touching them. A volume clone won't include them.
- Work on a branch off `1.6-dev`. Commits only when Caspar asks. This handover file itself needs committing
  (or its content carried over) if the work continues in a volume clone.
- Host-side things are out of this agent's reach from inside the container and stay with Caspar: `.wslconfig`,
  Defender, Docker Desktop settings, and the nightly `DockerAutoPrune` task.
- Any GitHub issue or public post: show Caspar the draft first.

## Done looks like

- `devcontainer.json` keeps `.venv` and models (and any other proven-heavy paths) in named volumes.
- The container prints a clear warning when opened from a Windows-drive bind mount.
- The docs steer Windows users to the volume clone and the WSL memory cap, with a troubleshooting entry.
- Caspar's verification run is recorded, pass or fail.
- `docs/development/roadmap_v1.6.0.md` (or the relevant roadmap) notes the change.
