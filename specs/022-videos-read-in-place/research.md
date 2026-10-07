# Research: Videos and Results Where You Expect Them

Decisions behind [plan.md](plan.md). Each records what was chosen, why, and what else was
considered. Code references are as of commit `84bd53b`.

## R1. How a job gets its results folder

**Decision**: Set the existing `BatchJob.output_dir` (already a `jobs.output_dir` column) to
`<results root>/<run folder>/<video folder>` when the job is created, on every creation path:
upload, folder ingest, dataset run (stored copies), rerun and the CLI's `process`. Keep
`storage_path` for VideoAnnotator's own bookkeeping, and for the uploaded copy of a video on the
upload route.

**Rationale**: The runner already writes every pipeline's output to `job.output_dir`, falling back
to `storage_path` only when it is unset (`batch/job_execution.py:88-94`). Result lookup reads
`output_dir` first too (`batch/result_files.py: job_folder`). So results move by setting one field
at creation, with no change to any pipeline (Principle II). Jobs created before this feature have
`output_dir` unset, keep the fallback, and stay where they are (FR-018).

**Alternatives considered**:
- *Copy results out after each job finishes*: two copies of every result, and FR-023 (results appear
  as each video finishes) would depend on a second step that can fail separately.
- *Symlink from the results folder into the job folder*: fragile on Windows, and the results folder
  would not stand on its own.
- *Move `storage_path` itself into the results folder*: the upload route's video copy would then
  land in the results folder (against FR-020), and job deletion's `name == job_id` safety check
  would no longer hold.

## R2. Run and video folder names

**Decision**:
- **Run folder**: `<run name> (<YYYY-MM-DD>)`.
  - The run name is the batch name if there is one. Otherwise the dataset name, then the source
    folder's name, and for a single unbatched job, the video's stem.
  - A clash adds ` 2`, ` 3`, … before the closing bracket: `BabyJokes (2026-10-06 2)`.
- **Video folder**: the video's stem. If two videos in one run share a stem, the folder takes the
  video's relative path inside the source folder, with separators replaced by `__`:
  `site_a__child01`.
- **Sanitising**: names drop characters that are invalid on Windows (`<>:"/\|?*` and control
  characters), and trailing dots and spaces. They avoid reserved device names (`CON`, `NUL`, …), and
  each path component is capped at 120 characters.
- **Uniqueness**: decided by creating the directory exclusively (`mkdir` without `exist_ok`), so two
  runs started at the same moment can't take the same name.

**Rationale**: researchers recognise the run name and the date. The video stem matches the
`<stem>_<suffix>` file names pipelines already write, so a video folder reads
`child01/child01_face_detections.json`, unchanged from today's file names (Principle V: output
files are not renamed). Creating the folder exclusively makes "never overwritten" (FR-024, SC-009)
a property of the filesystem rather than of a race-prone check.

**Alternatives considered**:
- *Job ID in the folder name*: unique, but unrecognisable, which is exactly today's problem.
- *Timestamps down to the second*: unique but noisy. The date plus a counter reads better.
- *Shorter result file names inside video folders* (`face_detections.json`): nicer, but it renames
  files that downstream scripts already match on. Deferred. The spec's
  layout shows today's names.

## R3. The run record (`run.json`)

**Decision**: Write `run.json` into the run folder when the run is created. It lists:
- the run name, ID and creation time,
- the VideoAnnotator version,
- the pipelines and their configuration,
- each video's source location (path for in-place, "uploaded: <original filename>" for uploads),
  and the video folder it maps to.

As each job finishes, update its entry with status, finish time, result files, and the model
references and versions that spec 017's provenance already records per pipeline result. Rewrite the
file atomically: write a temporary file in the same folder, then rename it over `run.json`.

**Rationale**: FR-022 asks for a folder that makes sense without VideoAnnotator. Spec 017 already
puts full provenance inside each output file (or its `.provenance.json` companion), so `run.json` is
an index and summary, not a second source of truth. Atomic replacement keeps it readable while jobs
finish concurrently.

**Alternatives considered**: CSV (no nesting for per-pipeline settings); writing only at the end (a
long run would have no record until it finishes, and a crash would leave none).

## R4. Default results root and how to change it

**Decision**:
- **Default**: `~/VideoAnnotator`, i.e. `Path.home() / "VideoAnnotator"`. That is
  `C:\Users\<name>\VideoAnnotator` on Windows.
- **Override**: the `VIDEOANNOTATOR_RESULTS_DIR` environment variable, set in the shell or in
  `.env`, which `config_env` already loads. There is no config-file key: `config_env` reads only
  the environment and `.env`.
- **Visibility**: `videoannotator server` prints the root at startup, as it already does for the
  storage root, and the viewer's Settings page shows it with how to change it.
- **Changing it**: affects new runs only. Old runs keep their recorded `output_dir` (FR-025).

**Rationale**: home is where a single researcher looks first and is on every OS. It is visible,
unlike the per-user data directory where job folders now live (`f71d012`). An environment variable
plus config key matches how `STORAGE_ROOT` and `VIDEOANNOTATOR_INGEST_ROOTS` are set. Changing it
from the viewer is not required by the spec (Story 8 says "set once") and would need a settings
write path the server doesn't have.

**Alternatives considered**: `~/Documents/VideoAnnotator`. Documents is OneDrive-synced by default
on many Windows installs, which would upload sensitive results to a cloud service without the
researcher deciding to (Principle I's spirit), and its location varies by locale and OS.

## R5. "Same machine" on a plain install

**Decision**: A caller is on the same machine when its address is loopback. That is the existing
`require_local_caller` rule in `api/v1/ingest.py`. A new endpoint,
`GET /api/v1/ingest/access`, tells the viewer, per caller:
- `same_machine`,
- whether the caller may read in place (same machine and administrator),
- the allowed folders,
- the results root,
- whether the server can open folders.

The viewer chooses the wizard's default from this.

**Rationale**: the server alone can see how a request arrived (spec Assumptions). The answer is per
caller, so it can't go in the cached `/health` server info. Reusing the existing rule keeps one
definition.

**Alternatives considered**: the viewer comparing its own URL host to `127.0.0.1`. That is wrong
behind any proxy, and it is something the browser claims rather than the server checks.

## R6. "Same machine" under Docker

**Decision**:
- **Bind to this machine only**: the documented Docker setups (compose and `docker run`) publish the
  port on the host's loopback only, `127.0.0.1:18011:18011`. Nothing else on the network can reach
  the server at all.
- **Tell the server**: they also set `VIDEOANNOTATOR_PUBLISHED_LOCALLY=1`. With it, the server
  treats every caller as the same machine, because only the host can reach it.
- **Say so at startup**: the server prints a warning that names this assumption, and says to unset
  the variable if the port is published more widely.
- **Default off**: on a plain install the variable is unset, and the loopback rule (R5) applies.

**Rationale**: inside the container, a request from the host's browser arrives from the Docker
bridge gateway (e.g. `172.17.0.1`). On Docker Desktop (macOS and Windows) it arrives from the VM's
gateway. Neither is loopback, so today's check refuses it. Trusting the gateway address alone is
not safe: on Docker Desktop, LAN traffic can arrive from the same address. Binding to the host's
loopback is the real guarantee, and the variable states that guarantee explicitly. Binding to
loopback is also the safer default in itself: today's compose publishes on all interfaces, so the
API (and, with auth off, everything) is reachable from the local network.

**Alternatives considered**:
- *Trust the bridge gateway IP*: unsafe on Docker Desktop, as above.
- *`network_mode: host`*: Linux only.
- *A shared secret only the host's browser holds*: the API key already plays that role and doesn't
  establish locality.

## R7. Allowed folders under Docker

**Decision**:
- **Videos**: compose mounts `${VIDEOS_DIR:-./videos}` read-only at `/videos` and sets
  `VIDEOANNOTATOR_INGEST_ROOTS=/videos`.
- **Results**: it mounts `${RESULTS_DIR:-~/VideoAnnotator}` read-write at `/results` and sets
  `VIDEOANNOTATOR_RESULTS_DIR=/results`.
- **Host paths for display**: `VIDEOANNOTATOR_HOST_PATHS=/videos=<host path>;/results=<host path>`,
  which compose fills from the same variables. The server rewrites the container path prefix when
  it shows a location.
- **No video folder mounted**: when `/videos` is missing or empty, the access endpoint reports no
  usable folder, and the viewer shows the setup instructions (FR-012).

**Rationale**: today the container runs as root, so the default root is `/root`, while compose
mounts data at `/app/data`. In-place reading is refused for every useful path. One named variable
per folder is the "single, clearly named place" of FR-011. The viewer must show the host's path,
because the researcher can't find `/results/...` on their own disk (SC-007).

**Alternatives considered**: keeping `./data` at `/app/data`. That name doesn't say "your videos";
`./data` stays mounted for compatibility (FR-018) but is not the documented route.

## R8. Picking individual videos

**Decision**: Extend the ingest request (`POST /api/v1/ingest`) with an optional `files` field: a
list of paths relative to `path`. When given, only those files become jobs. Each is resolved and
checked against the allowed folders individually, and duplicates are removed by resolved path.
When absent, the whole folder is used, as today.

**Rationale**: additive and backward compatible (FR-018, Principle V). It reuses one request for
the whole run, and the existing per-file `skipped` reporting.

**Alternatives considered**: one ingest call per file. That makes many batches, or needs a batch ID
threaded through calls, and loses the single-step property (FR-006).

## R9. Datasets saved from a selection

**Decision**: Add `server_selection` (a boolean, default false) to saved datasets. When true, the
manifest is the selection: using the dataset runs exactly the manifest's files under
`server_folder`, and new files in the folder are not "differences". A moved or missing manifest file
is reported as missing (FR-015, FR-017). The column is added with the existing `ALTER TABLE`
pattern in `database/migrations.py`.

**Rationale**: a whole-folder dataset treats the folder as the truth (new files are drift). A
selection dataset treats the manifest as the truth. One flag distinguishes them without changing
existing datasets' behaviour.

**Alternatives considered**: inferring a selection when the manifest is smaller than the folder.
Ambiguous: a folder dataset whose folder grew looks the same.

## R10. Missing videos

**Decision**:
- **When a job starts**: the runner checks that `video_path` exists. If it doesn't, the job fails
  with "Video not found: <path> (moved or deleted since the job was created)", and the batch
  continues (FR-016).
- **Before a rerun or dataset use**: rerun and batch rerun already skip missing videos with a
  reason (`RERUN_VIDEO_MISSING`). The viewer calls a new dry run first (`?check=true` on batch
  rerun), which lists the missing videos and creates nothing. It shows them before submitting
  (FR-015).
- **Relocating**: batch rerun accepts `relocate_folder` (a folder inside the allowed folders) and
  `recursive`. For each job whose video is missing, the server looks in that folder for a file
  with the same name and size (`source.size_bytes` in `run.json`). Matches rerun from the new path;
  unmatched videos stay in `skipped`. With `check=true` it only reports the matches. Matching on
  name and size, not name alone, means a different take with the same name is never picked up
  silently (FR-015's "point to the videos' new location").
- **Folder and selection datasets**: these already scan and show differences
  (`DatasetDriftDialog`), so missing files are listed there.
- **Results stay viewable**: the job results endpoint keeps serving outputs regardless of the video.
  The viewer shows "Video not found at <path>" in place of playback (FR-014).

**Rationale**: each check sits where the video is first needed, and every message names the
location (FR-013).

## R11. Opening a folder on this computer

**Decision**: `POST /api/v1/results/open` with `{"path": ...}`, allowed only for same-machine
callers and only for paths under the results root. It opens the folder with the OS file manager
(`os.startfile` on Windows, `open` on macOS, `xdg-open` on Linux). It returns
`OPEN_FOLDER_UNSUPPORTED` when there is no desktop to open it on: headless Linux, under Docker, or
when the opener fails. The access endpoint's `can_open_folders` lets the viewer show the button
only where it can work. The location is always shown with a copy button.

**Rationale**: a browser page can't open a local folder, so the server, on the same machine, has to.
Restricting it to the results root means it can't be used to probe the filesystem.

**Alternatives considered**: `file://` links. Browsers block these from http pages.

## R12. Run download

**Decision**: `GET /api/v1/batches/{batch_id}/results.zip` streams a zip of the run folder:
`run.json` and every video folder, never videos (FR-028). The job artifacts zip
(`GET /api/v1/jobs/{id}/artifacts`) gains `include_video` (default false). Today's behaviour
(always including the video) changes, so this is noted in the CHANGELOG as a user-facing change.
`include_video=true` restores it (FR-029).

For jobs from before this feature, which have no run folder, the run zip assembles the same layout
from each job's folder.

**Rationale**: one download per run (SC-008), and the same layout as on disk, so scripts work on
either.

**Constitution note**: changing a download's default contents is not an output-schema change (the
pipeline files are unchanged), but it is a user-facing default. It is announced in the CHANGELOG,
with an opt-in to the old behaviour, which is the spirit of Principle V.

## R13. The viewer's "My folders" picker

**Decision**: Rework `ServerFolderPicker` into the "My folders" tab:
- It lists the subfolders and the videos of the current folder, from the existing `browse` and
  `scan` endpoints.
- Each video has a checkbox; there is "Select all", and an "Include subfolders" toggle.
- Durations load lazily: shown when the scan already has them, otherwise left blank. The listing
  never waits on them.
- Submission sends `path` plus `files` (R8).

The video source becomes two tabs (My folders, Saved datasets) when `same_machine`. Otherwise it
is upload plus Saved datasets. Upload on a same-machine connection is the "Videos on another
computer? Upload them" link.

**Rationale**: it reuses the picker and endpoints already shipped in spec 008, adding only file
selection.

## R14. CLI `process`

**Decision**: `videoannotator process <video>` reads the video in place (no hard link or copy into
the job folder) and writes results to the results folder, run-named after the video. `--output`
keeps its meaning (copy the result files to that folder too).

**Rationale**: the same "no copies, findable results" rule on every route. The CLI was the last
path still placing videos in the job folder (`batch/local_job.py:_place_video`).
