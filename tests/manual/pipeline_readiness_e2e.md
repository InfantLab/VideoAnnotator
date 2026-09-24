# Manual test: pipelines from the viewer, no terminal (spec 011 SC-001)

Run before tagging a release, against the viewer bundled in `src/videoannotator/viewer_static/`.
The only terminal commands allowed are the ones in **Setup**; after that, everything happens in the
browser. Record the date, platform and result at the bottom.

## Setup

1. A **core-only** environment: a fresh venv or container with `videoannotator` installed and no
   extras. In a source checkout: `uv venv .venv-e2e && UV_PROJECT_ENVIRONMENT=.venv-e2e uv sync`.
   Check: `python -c "import torch"` fails.
2. For the diarization part, set `HF_AUTH_TOKEN` in the server's environment (the container env, or
   `.env` for docker compose) with a token whose account has accepted the licences of
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and
   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
   Do part A first **without** it to see the setup message, then restart with it for part C.
3. Start the server with `videoannotator server` (one worker, no `--reload`; otherwise the restart
   step reports that it can't restart itself, which is correct but not this test).
4. Create an admin key: `videoannotator generate-token --admin`, and open the printed
   "Connect the viewer with one click" link.
5. Have a short real video (10–30 s with speech and a face) on the machine running the browser.

## A. Discover

Open **Create Job**, add the video, go to **Select Pipelines**.

- [ ] Every shipped pipeline is listed. No "Stub" pipeline.
- [ ] Face, audio, scene, person pipelines show **Not installed**, with "Installs the *group*
      group (approx. N MB)". The three audio pipelines each say the group also enables the others.
- [ ] VLM Frame Annotation shows **Not installed** (llm group), or **Needs setup** with
      "Ollama isn't reachable" if the llm extra is already there and Ollama isn't running.
- [ ] Nothing on the page asks you to run a command.

## B. Install and use `face_analysis`

- [ ] Click **Install** on Face Analysis (DeepFace). The card shows progress; it takes minutes.
- [ ] Reload the page mid-install: the card still shows **Installing** (picked up from the server).
- [ ] When it finishes, **one** of:
  - it turns selectable on its own ("Installed. Loading…" then a checkbox): *live activation*; or
  - a **Server restart needed** banner appears. Click **Restart server**, see
    "Restarting server…", and within about 30 s the page updates and Face Analysis is selectable.
    If a job is running you're asked first; "Restart anyway" interrupts it.
- [ ] Select Face Analysis, finish the wizard, submit. The job completes and results open in the viewer.

## C. `speaker_diarization`

- [ ] Install the **audio** group from Speaker Diarization's card. Afterwards (restart if asked):
  - without `HF_AUTH_TOKEN`: the card shows **Needs setup**, "Set HF_AUTH_TOKEN in the server's
    environment…", a **Get a token** link, and no field to type a token into;
  - with it: the card is selectable, with licence notes linking to both model pages, and a note
    about the first-run model download.
- [ ] Run a job with Speaker Diarization. It completes. (A licence not accepted on Hugging Face
      shows up here as a job failure that says so: licences can't be checked locally.)

## D. Edge cases (quick)

- [ ] Start the server with `--reload`, click **Restart server** (after any install that asks
      for it): the banner shows the server's own instructions for restarting manually.
- [ ] Non-admin key: cards show their state, but Install/Restart are replaced by an explanation.
- [ ] With Ollama stopped, VLM shows **Needs setup**. Start `ollama serve` (and pull a model),
      click **Check again**: the card becomes selectable.

## Results

| Date | Platform | Viewer build | Result | Notes |
|------|----------|--------------|--------|-------|
|      |          |              |        |       |
