# Walkthrough before rc1: what to check

For Caspar, in the existing dev container, with no fresh install. Model weights are already in
`/app/models`, so nothing below needs a download, except where marked 🌐. Note anything odd
under each section, or at the bottom.

## 0. Start (2 min)

The container was built before `STORAGE_ROOT` was added to `devcontainer.json`, so set it by hand
until the next rebuild:

```bash
export STORAGE_ROOT=/workspaces/VideoAnnotator/storage/jobs
export HF_HUB_OFFLINE=1   # slow wifi: stop Hugging Face checking for updates; weights are cached
bash scripts/dev.sh       # API on 18011 plus hot-reload viewer on 19011
```

- [ ] The server log prints `[INFO] Jobs: /workspaces/VideoAnnotator/storage/jobs`
- [ ] Open http://127.0.0.1:19011 (or the bundled one at http://127.0.0.1:18011/viewer). Connected
      at top right. If not: `uv run videoannotator generate-token --port 18011` and open its link
- [ ] Ollama is running on the laptop (for section 6)

## 1. Home (new)

- [ ] The status line shows server version, "N of 7 pipelines ready" and saved results
- [ ] **Open latest results** opens your newest finished job
- [ ] Recent runs: a finished job offers View and ↻ Run again; a failed one shows its error and
      **Fix & rerun**
- [ ] Datasets and Starred prompts panels list yours; ▶ on a dataset starts the wizard with it
- [ ] The "Getting set up" checklist shows only if a step is undone, and **Hide** keeps it hidden
- [ ] Nav reads New job · Jobs · Results · Datasets · Prompts; the logo goes Home
- [ ] Narrow the window to phone width: nothing scrolls sideways (Jobs' table may, by 11 px)

## 2. Run a job (the main path)

- [ ] New job → upload a short clip → pick Scene Detection plus Person Tracking → Configure →
      Submit. It lands on the run page with "Preparing…"
- [ ] Jobs list: queued jobs say "Nth in queue" when more than one is waiting
- [ ] It completes. Job page: Run it again / Edit and run again / Use these settings / Save as
      preset
- [ ] **Download Results**: the zip holds the video and each pipeline's file
- [ ] Try the other two ways in: "Folder on the server" (scan a folder), and "Saved dataset"

## 3. Review in the viewer

- [ ] Open in Viewer → "Choose a folder" or "View without saving". Does asking first feel
      right? (an open question)
- [ ] Overlays draw; each is labelled with the pipeline and version that made it, and the
      provenance details open on demand
- [ ] Timeline: scrub, play, frame step
- [ ] The annotation panel lists pipelines the job didn't run, marked "(No data)". Keep that or
      hide them?
- [ ] Results page (was Library): your saved results; **Install the demos** works offline and
      the demos open

## 4. Run it again (spec 019)

- [ ] **Run again** on a finished job makes a new job with the same settings
- [ ] **Edit and run again**: the wizard opens with that video and settings; change one setting
      and submit
- [ ] On a batch: rerun the whole batch
- [ ] Save as preset, then use the preset in a new job

## 5. Datasets (spec 018)

- [ ] Make a dataset from a server folder; it lists videos with paths
- [ ] Start a job from it. The run page says "dataset" followed by its ID, not its name (worth fixing?)
- [ ] Add or remove a video in the folder, then reopen: drift is reported

## 6. VLM work (specs 020 and 021; needs Ollama)

- [ ] A VLM job with a prompt and gemma4:e4b completes; labels show on the timeline
- [ ] Prompts page: the prompt is there; star it, rename it, tick two to see a word diff
- [ ] **Open in workbench**: preview frames from a video with two prompts × two models
- [ ] Compare (from a VLM job page): pick a second VLM job on the same video, then check the
      agreement summary, seeking to a disagreement, and the CSV export. Optional: load an ELAN
      `.eaf` as ground truth

## 7. Provenance (spec 017)

- [ ] Open a fresh output JSON: a top-level `provenance` block with version, model, weights
      digest and settings
- [ ] WebVTT starts with a `NOTE videoannotator-provenance` line; RTTM has a
      `.provenance.json` beside it
- [ ] An old result file (from before) still opens in the viewer, labelled "version not
      recorded"

## 8. Upgrade behaviour (today's changes)

- [ ] Jobs made before today still list, open, download and delete (their folders stayed where
      they were)
- [ ] Settings shows an **Admin** badge for your key. A pipeline you haven't installed offers **Install** (don't
      press it on the train 🌐)

## 9. Command line (quick)

- [ ] `uv run videoannotator process <clip> --pipelines scene_detection` writes results and
      records the job (it shows in the viewer)
- [ ] `uv run videoannotator job rerun <id>` and `uv run videoannotator prompts list`

## 10. Docs (when you have wifi 🌐)

- [ ] README reads right for a researcher; the install steps match your memory of each OS
- [ ] The docs site (GitHub Pages) appears after the next merge to master
- [ ] Offline instead: `uv run --no-sync mkdocs serve`, then http://127.0.0.1:8000

## Notes

-
