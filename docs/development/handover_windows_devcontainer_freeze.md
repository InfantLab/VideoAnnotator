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
| 09:06:17 | Event log's last "dirty shutdown" heartbeat stamp (a periodic stamp, not the last event) |
| 09:14:40 | Host exits Modern Standby: the last host event logged. Scheduled task refused (0x800710E0); last Docker log lines by 09:20. Caspar woke it into a hung machine |
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

## Update 2026-10-02 (later): host-side answers (diagnosis superseded by Update 3)

Answers to the three "Next evidence" questions, from the host session.

**1. Settings and versions.** Docker Desktop's **Resource Saver is off** (`"UseResourceSaver": false` in
`%APPDATA%\Docker\settings-store.json`), so the docker/for-win#14656 cause doesn't apply here.
- Docker Desktop 4.93.0, engine 29.8.1.
- WSL 2.6.3.0, kernel 6.6.87.2-1, `networkingMode=mirrored`, no memory cap.
- Windows 10.0.26340, a **Dev-channel Insider build**.
- Hardware: ASUS ProArt PX13 (HN7306WV, Ryzen AI 9 HX 370), MediaTek MT7925 Wi-Fi. The firmware offers only
  S0 Modern Standby: no S3 sleep.

**2. The 7 dirty shutdowns.** Every one happened at a Modern Standby transition: the last host events before
each crash are standby enter/exit, and boot type 0x0 follows (a cold boot). These are **the same incidents as a
Modern Standby crash series Caspar has been investigating on this laptop since 3 Sep**, independently of
VideoAnnotator. It's recorded in his own notes, outside this repo. In that series the machine dies in or
around standby; the fallout includes fTPM/BitLocker corruption. It's still unsolved, and the BIOS and the
AMD/NVIDIA drivers are already current.

- The Docker/WSL VM was up in 6 of the 7. **But it was also up before 499 of the 502 standby entries in the
  last 31 days.** It's always on (Docker Desktop autostarts), so "VM running" is the base rate, and the
  correlation doesn't implicate Docker or the devcontainer.
- Crash rate: 7 in ~502 standby entries, roughly 1 in 70.

**What this means for the diagnosis.** The most likely story is that today's freeze is **episode 7 of a
host-side Modern Standby fault**, not something VideoAnnotator caused. The container evidence still stands:
event-loop stalls at 08:39, then the `page_reporting_process` soft lockup at 08:44. That shows the VM
suffering on resume, but it can't distinguish "the VM caused the hang" from "the VM was a victim of a host
that was already failing". Not proven either way.

**What this means for the branch.**
- Keep the volumes, the Windows-drive warning and the docs. They're justified by **performance alone**,
  because the file bridge is slow regardless.
- **The repo must not claim these changes prevent freezes.** Word the start-up warning and the docs around
  speed: "file access across the Windows drive is slow; use a volume clone". Mention possible system
  instability only as "reported", if at all.
- Keep "stop the container before the machine sleeps" out of user docs unless step 3 below supports it.
  There's no evidence for it yet.
- Don't open a public issue blaming Docker or WSL on this evidence.

**3. The controlled repeat: why a single sleep/wake won't tell us anything.** At a ~1-in-70 crash rate, one
clean sleep/wake proves nothing. The test that could separate host from VM is time-based. Caspar runs with
Docker Desktop fully quit (no autostart) for two weeks, and compares that with the prior rate of ~1.6
crashes a week.
- A crash with Docker quit clears Docker.
- Zero crashes in two weeks has roughly a 4% chance under the old rate, so it would point at the VM.
- One crash-free week is weak evidence (about 20% chance by luck).

This is Caspar's host decision, outside this agent's scope. Until it's done, **the repo work shouldn't wait on
it**: ship the performance changes with neutral wording.

**Step 4 below is superseded.** There's no test run to repeat. Replace it with a simple performance check: time
`uv run pytest tests/unit -q` once in the bind-mounted container and once in a volume clone, and record both
times in the troubleshooting entry. That's the evidence the docs change actually rests on.

## Update 3 (2026-10-02, afternoon): it was a memory-exhaustion hang, not the standby series

Caspar corrected Update 2. He saw **~99% memory use** with the machine awake and unusable, and pressed the power
button himself. The earlier crashes in his standby series died unattended. The logs agree that this one was
different:
- Kernel-Power 41 has a power-button timestamp set this time.
- The machine was awake, and logging hangs, for ~55 minutes before the reset.

The settings, versions and base-rate facts in Update 2 still stand. Its conclusion ("episode 7 of the standby
series") does not.

**Corrected timeline (BST).**

| Time | Event |
|---|---|
| 10-01 21:46 | Windows Resource Exhaustion Detector fired: low memory **the evening before**, with the same apps open |
| 10-02 ~05:36 | Overnight standby fell through to **hibernate** (system clock 04:36Z → 07:38Z at resume) |
| 08:38:39 | **Resume from hibernate** (Kernel-Boot 27, boot type 0x2). The full memory image, including the WSL VM, was read back from disk |
| 08:39 | Container: Claude extension event loop blocked 42 s, then 27 s; ptyHost heartbeats lost |
| 08:42:25 | Winlogon 1002: **Explorer (the shell) crashed** and was restarted |
| 08:44:44 | VM kernel soft lockup in `page_reporting_process`: the VM giving memory back to Windows |
| 08:50 → 09:14 | Screen-off standby, then Caspar woke it |
| 09:14 onward | Lock screen hung (WER MoAppHang, LockApp). Task Manager unreachable. ~99% memory |
| 09:34 | Power-button reset |
| after reboot | Two WER `LiveKernelEvent 0x193` reports (`dxgkrnl!ProcessDeadlockThread`, a graphics-kernel deadlock live dump). When they were captured is unknown (the dump folder needs admin). Could be symptom or cause; unresolved |

**Where the memory goes on this host.** Snapshot taken 105 minutes after the reboot, same everyday apps, one
container running:

| | Private memory |
|---|---|
| All processes (456 of them) | 35.3 GB |
| Physical RAM in use | 25.4 of 31.1 GB (5.4 GB available) |
| Commit charge | 48 of 77 GB |

The biggest holders:

| Process | Private memory |
|---|---|
| Edge (37 processes) | 4.9 GB |
| **WSL VM (`vmmemWSL`)** | **4.7 GB**, can grow to its 15.6 GB cap |
| MariaDB `mysqld` (4 GB buffer pool, mostly paged out) | 4.3 GB |
| VS Code | 2.7 GB |
| Claude | 2.2 GB |
| WebView2 | 2.1 GB |
| OneDrive | 2.7 GB |
| Defender (`MsMpEng`) | **0.4 GB**. So Defender was likely busy on CPU after the resume, not the memory hog |

**Working diagnosis (supported, not proven).** The host runs at ~80% of RAM with everyday apps. The WSL VM is the
only thing that can grow by ~11 GB, up to its uncapped-default 15.6 GB, because Linux page cache and process
memory count as VM memory until the VM hands them back. The soft lockup was in exactly the routine that hands
memory back. Resuming from hibernate with a large VM image adds a burst of pressure. Together that's enough to
reach 99%. Nobody recorded per-process memory during the hang, so the VM's actual size at 09:14 is unknown.

**What changes for the branch.**
- **The WSL memory cap moves from "nice to have" to the main user-facing advice.** It bounds the one component
  that can balloon. It belongs in the docs and the troubleshooting entry, with a symptom-first heading like
  "Windows becomes unresponsive / memory near 100% with the dev container running".
  - Size the recommended cap from evidence, not a guess. **Measure the container's peak memory during the
    full test suite and during a real-models pipeline run** (`docker stats`, or cgroup `memory.peak` inside
    the container), then recommend peak plus headroom.
  - Also check whether WSL's `autoMemoryReclaim` setting (`[experimental]` in `.wslconfig`; values
    `disabled` / `gradual` / `dropCache`) is worth recommending. Verify it against current Microsoft docs first.
- The volume and bind-mount changes are still justified on performance grounds. Keep the speed-only wording.
- The two-week "Docker quit" test in Update 2 is about the old standby series. It doesn't apply to this incident.
  Drop it from this repo's scope.

**Host-side follow-ups (Caspar, not this agent).** Set the cap on his own machine. Run a lightweight memory logger
(top processes by private memory every few minutes) so the next incident names its culprit. Decide whether
MariaDB needs to run at startup.

## Update 4: more inputs from the host side

1. **The repo can cap the container itself. This is the strongest protection it controls.** `.wslconfig` is
   per-machine and most users will never set it. A container memory limit in `devcontainer.json` `runArgs`
   ships with the repo. Make it overridable, with a default sized from the peak you measure (Update 3):
   `"--memory=${localEnv:VIDEOANNOTATOR_DEV_MEMORY:12g}"`, plus a matching `--memory-swap`. The devcontainer
   spec supports `${localEnv:VAR:default}`; check the exact syntax.
   - Linux page cache counts inside the container's limit and is reclaimed within it. So the cap bounds how far
     container activity can grow the VM. It doesn't bound other WSL distros.
   - The trade-off: a cap set too low gets real-model loads killed by the out-of-memory killer. Document the
     symptom ("Killed", exit code 137) and the override in the troubleshooting entry.
   - Unlike `.wslconfig`, this needs no `wsl --shutdown`, and it only affects this project.

2. **Size advice by the host's headroom, not just the container's needs.** On Caspar's machine, everyday apps
   (Edge, VS Code, Claude, OneDrive, WebView2 apps, Granola…) take ~20 GB of 31 GB before any container runs. A
   typical 16 GB laptop has far less spare. The docs should say this plainly: "on a 16 GB machine, keep the
   container under ~6–8 GB and close other heavy apps for real-model runs". Base the numbers on your
   measurements.

3. **On the WSL2 backend, Docker Desktop's memory limit lives in `.wslconfig`.** Docker Desktop's Settings →
   Resources has no memory slider when it uses WSL2; Windows manages it. Users look for the slider and
   don't find it, so say this in the docs. Check against current Docker docs.

4. **GPU memory may sit outside both caps (unverified).** With `--gpus all`, the host showed 1.9 GB and 2.5 GB of
   "shared GPU memory" in use. Shared GPU memory is system RAM. Whether CUDA allocations from inside WSL
   count against the VM or cgroup limit, or come straight out of Windows RAM, is unknown here. Worth a check
   during a real-models run: watch the container's memory while Caspar watches Task Manager → GPU → shared
   memory. If it's outside the caps, that's a docs line too.

5. **Check the default test run excludes real-model and GPU tests.** The markers `real_models`, `gpu` and `slow`
   exist. If a bare `pytest` pulls them in, a new contributor's first run is the heaviest possible one.
   Consider deselecting them by default and documenting the opt-in.

6. **Not for the repo (host-only, already handled or Caspar's).** A forgotten MariaDB service was reserving 4 GB;
   now disabled. Caspar's host also has the WSL cap, a memory logger, and the nightly `docker system prune -a`
   (which deletes stopped images, so expect rebuilds after a night off). Keep these out of user docs, except
   the general point in item 2.

## Update 5 (agent, 2026-10-02): what's done, and how to measure

**Checked against current docs.**
- **devcontainer:** `${localEnv:VAR:default}` works in any property, `runArgs` included. The JSON
  reference lists `localEnv` for "Any".
- **WSL (Microsoft's `wsl-config` page, updated 2026-09):** `memory` defaults to 50% of RAM and `swap`
  to 25%. Changes apply only once WSL has stopped (`wsl --shutdown`), and Microsoft now points users
  to the WSL Settings app. `autoMemoryReclaim` already defaults to `dropCache` (cached memory
  reclaimed immediately), so recommending it would change nothing; it's left out.
- **Docker Desktop:** "In WSL 2 mode, configure memory, CPU, and swap limits on the WSL 2 utility VM",
  i.e. `.wslconfig`.

**Done (item 1 of Update 4).** `runArgs` gains `--memory=${localEnv:VIDEOANNOTATOR_DEV_MEMORY:12g}`.
- 12g is provisional, to be replaced by peak plus headroom once measured.
- `--memory-swap` is deliberately not set. A fixed value below an overridden `--memory` would stop the
  container from starting, and Docker's default without it is twice `--memory`.
- The start-up warning now talks about speed only.

**Seen in this container (bind mount, VM uncapped).**
- The VM still reports `MemTotal` 15.9 GB, so no `.wslconfig` cap is active yet; it needs
  `wsl --shutdown`.
- The container's cgroup has no limit, and its peak since 08:40 is 3.7 GB.
- VS Code server plus Claude sit at a ~3.5 GB baseline.
- `pytest --collect-only` alone takes **3–4.5 minutes** here (six runs, 179–266 s).
- Of 1,375 tests: 1 is marked `real_models`, 0 `gpu`, 0 `pipeline`, 69 `slow` and 12 `performance`.
  A bare `pytest` runs all of them. CI's main job deselects `real_models` and `performance`.
- Most markers are added by name in `tests/conftest.py`, so unmarked tests may still load real weights.

**Measurements, in this order.**
1. **Caspar:** open a fresh **Clone Repository in Container Volume** of `windows-devcontainer-freeze`, which
   picks up the 12g container cap. No `.wslconfig` cap for now: Caspar wants more data before making a
   machine-wide change. So the container cap is what bounds these runs, and the docs must not lead
   with the WSL cap until the data supports it.
2. **Agent, in that container**, sampling the cgroup every second (`memory.current`, plus `anon` and
   `file` from `memory.stat`, plus `memory.events` for any limit hits):
   - (a) `pytest --collect-only -q` and `pytest tests/unit -q`: wall time, compared with the bind mount;
   - (b) the full default `pytest`: peak memory and the tests that drive it;
   - (c) one real-models job on the demo video with every pipeline: peak memory.
3. **Caspar, during (c):** watch Task Manager → Performance → GPU → "Shared GPU memory". If it rises
   without the container's memory rising, CUDA allocations sit outside both caps (Update 4, item 4).
4. **Then:** the default cap becomes peak (c) plus headroom. The docs get the 16 GB-machine guidance and
   the OOM symptom (`Killed`, exit 137) with the override. Whether a bare `pytest` should deselect heavy
   tests depends on (b).

## Update 6 (agent, 2026-10-02, ~13:00): measurements in the volume clone

**Setup.**
- Fresh Clone Repository in Container Volume of `windows-devcontainer-freeze` at `084cef4`.
- Workspace on ext4 (no 9p); cgroup `memory.max` = 12 GiB; VM `MemTotal` 15.9 GB (no `.wslconfig` cap).
- RTX 4060 Laptop, 8 GB VRAM; Python 3.13.
- The fresh `.venv` volume held only the core install (71 packages, no torch), because `postCreateCommand`
  runs `uv sync` without extras. I installed `uv sync --inexact --extra all` (torch 2.11.0+cu126, CUDA
  available). The models volume had survived, so no weights were downloaded.
- Sampled once a second: `memory.current`, `anon`/`file` from `memory.stat`, `memory.events`, and
  `nvidia-smi` used VRAM.
- `anon` is the number that matters below. `file` (page cache) grows to fill whatever the cap allows and
  is reclaimed inside the cgroup, so `memory.current` sits at about 12 GiB most of the time and says
  nothing about need.

**Results.** In every phase, `oom_kill` stayed at 0. The `max` events in the last column are cache
reclaim, not kills.

| Phase | Wall time | Peak `anon` (whole container) | Peak VRAM | `max` events |
|---|---|---|---|---|
| Baseline (VS Code server + Claude, idle) | – | 1.6–2.1 GiB | 0 | – |
| `uv sync --extra all` | 320 s | 2.0 GiB (**`file` 9.8 GiB**) | 0 | 38,036 |
| (a) `pytest --collect-only -q`, core env only | 0.9–3 s (11 import errors) | – | – | – |
| (a) `pytest --collect-only -q`, all extras, ×3 | 27 s cold, then 9–11 s (pytest reports 6 s) | 2.7 GiB | 0 | – |
| (a) `pytest tests/unit -q` | 23 s (806 passed, 1 failed, see below) | 2.7 GiB | 0 | – |
| (b) full default `pytest` (1,375 tests) | **96 s**: 1,342 passed, 33 skipped | **5.4 GiB** (pytest RSS 4.1 GiB) | 0.8 GiB | 28,446 |
| (c) API job, demo video (10 s), 6 pipelines | 45 s | **6.45 GiB** (server RSS 5.6 GiB) | 1.0 GiB | 15,055 |

**What it shows.**
- **Speed.** Collection drops from 179–266 s on the bind mount to 9–27 s on the volume, about 10–25×
  faster. The whole default suite runs in 96 s. The volume clone advice stands on speed alone.
- **The install alone fills the cap with page cache.** Installing the extras pushed `file` to 9.8 GiB
  while `anon` stayed at 2 GiB. Without a container cap, that cache sits in the WSL VM until
  the VM hands it back. That is the growth mechanism Update 3 suspected, now seen directly. With
  the cap, reclaim kept it at 12 GiB with no kills.
- **(b) The default `pytest` is not heavy.**
  - Peak `anon` was 5.4 GiB including the ~2 GiB editor baseline.
  - The biggest single step was +2.15 GiB at `tests/pipelines/test_face_analysis.py::TestDeepFaceAnalysis::test_deepface_error_handling`,
    where TensorFlow/DeepFace loads. Whisper tests come next (`test_speech_audio_processing`, about 9 s).
  - Memory ratchets up and isn't freed: pytest holds about 4.1 GiB RSS to the end.
  - Deselecting `real_models`/`gpu`/`slow` by default isn't needed for memory. It would buy little time,
    since the whole run is 96 s.
- **(c) A real-models job peaks at about 6.5 GiB of container `anon`**, about 4.7 GiB above baseline.
  - The pipelines run in order and each one keeps its models loaded, so memory climbs stepwise:
    job start plus DeepFace/TensorFlow +2.7, OpenFace3 +0.7, YOLO +0.1, scene/CLIP +0.6, pyannote/whisper
    +0.2 GiB.
  - The server still held 5.6 GiB RSS after the job ended.
  - VRAM never passed 1 GiB of the 8 GB card. Whether "Shared GPU memory" in Task Manager moved was
    **not observed** (step 3 wasn't coordinated with Caspar), so Update 4, item 4 is still open.
- **Cap sizing.**
  - Measured need: 6.5 GiB peak (c), 5.4 GiB (b), each including about 2 GiB for the editor.
  - With headroom, **8g** would cover both. It is also exactly WSL's default VM size on a 16 GB machine
    (50% of RAM), so there the VM default already binds before a 12g container cap does.
  - The 12g cap only bites on hosts of 24 GB or more, like Caspar's 32 GB.
  - Caveat: the demo video is 10 s long. A long video, a larger Whisper model or concurrent jobs may need
    more. **Recommendation:** keep 12g until one long-video job has been measured, then set peak plus
    about 1.5 GiB (likely 8g or 10g).

**Side findings.**
- `tests/unit/test_check_workspace_mount.py::test_warns_for_a_windows_drive_bind_mount` failed.
  - Cause: when 8e457c4 reworded the start-up warning, it wrapped the phrase "Clone Repository in Container
    Volume" across two lines.
  - Fixed in `scripts/check_workspace_mount.sh` (uncommitted). The full run afterwards passes.
- `face_analysis` (DeepFace) completed with **0 annotations** on the demo video, while OpenFace3 found faces
  in it. Its output file isn't written, and there was no error. Expected: the demo clip has no faces
  DeepFace detects (`tests/fixtures/viewer_contract/README.md`), in v1.5.0 too.
- `videoannotator process` is a stub (`TODO`, `src/videoannotator/cli.py:209`), so (c) used
  `videoannotator server` with `AUTH_REQUIRED=false` and a scratch DB/storage. `--dev` alone still
  demanded an API key on `POST /api/v1/jobs/`.
- A fresh volume clone gets no torch or extras by default. That's by design (004), but the dev-container
  docs should give the `uv sync --inexact --extra all` step, and say it takes about 5 minutes and fills
  the page cache.

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

4. **(Superseded. See "host-side answers" above: use the unit-test timing comparison instead.)** **Prove it (Caspar runs this on the host, the agent prepares it).** Do this after the changes are pushed.
   Caspar does a fresh "Clone Repository in Container Volume" of the branch and runs the same regression suite
   that froze the machine, with Task Manager open. If the host stays responsive, record the result in the
   troubleshooting entry as the evidence. If it still freezes, the diagnosis is wrong: stop and report back
   before doing more.
   The agent should write down the exact test command that was running (ask Caspar if it's not in shell history), so
   the comparison is like-for-like.

## Constraints and cautions

- No test run caused the freeze (see the updates above), but the heavy regression suite is still slow over the
  bind mount. Prefer running it from a volume clone. Unit tests are fine anywhere.
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

**Fix (2026-10-02): `--memory` hard-coded to `12g`.** In a clone-in-volume container the
`${localEnv:VIDEOANNOTATOR_DEV_MEMORY:12g}` default was not applied: with the variable unset on the host it
resolved to an empty string, and `docker run` failed with `invalid argument "" for "-m, --memory" flag`.
The `VIDEOANNOTATOR_DEV_MEMORY` override is dropped. To change the cap, edit `runArgs` directly.
