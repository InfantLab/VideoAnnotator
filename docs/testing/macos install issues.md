# 🐛 Video Annotator – macOS Testing Bug Report

This document consolidates user testing feedback from a macOS environment for the [Video Annotator](https://github.com/InfantLab/VideoAnnotator) project.  
Each section below describes a reproducible issue, observed behaviour, and suggested next steps.

---

## 1. ⚠️ Permission Issues – Shell Config & `.config` Directory

**Summary:**  
Installer cannot write to `~/.zshrc` or create `~/.config` due to root ownership, preventing path setup and config creation.

**Steps to Reproduce:**
1. Run installer or launch backend from terminal.
2. Attempt to add `uv` to PATH or save edits to `.zshrc`.
3. Observe write failures or missing `.config` directory.

**Expected:**  
Installer automatically creates or edits these files with correct permissions.

**Actual:**  
- `~/.zshrc` and/or `~/.config` are owned by `root`.
- PATH not updated and installer fails silently.

**Suggested Fix:**
```bash
sudo chown -R $USER ~/.zshrc
chmod u+w ~/.zshrc
sudo chown -R $USER ~/.config
```
- Add permission check + warning in installer.
- Provide fallback to `.zprofile` if `.zshrc` missing.

**Screenshot Placeholder:**  
![Permission error](<add-screenshot-here>)

---

## 2. 📦 NPM / Node.js Installation & Script Errors

**Summary:**  
On fresh macOS setups, `npm` is not installed and the README suggests `npm start`, which is not a valid script.

**Steps to Reproduce:**
1. Clone the repo.
2. Navigate to `video-annotation-viewer` and run `npm start`.
3. Observe: `zsh: command not found: npm` or “Missing script: start”.

**Expected:**  
- Clear pre-check for Node.js / NPM presence.  
- Correct instructions for starting the frontend.

**Actual:**  
- `npm` not installed.  
- Wrong script name (`npm run dev` is correct).

**Suggested Fix:**
- Add version check in installer:
```bash
node -v && npm -v
```
- Update docs:
```bash
cd video-annotation-viewer
npm install
npm run dev
```

**Screenshot Placeholder:**  
![NPM error](<add-screenshot-here>)

---

## 3. 🪛 Missing Python Dependencies

**Summary:**  
Backend fails due to missing modules (`lap`, `librosa`, `pycocotools`, `webvtt-py`, `pyannote.core`, `praatio`, `openai-whisper`).

**Steps to Reproduce:**
1. Install requirements with current instructions.
2. Run API server.
3. Observe `ModuleNotFoundError` messages.

**Expected:**  
All core dependencies are installed automatically.

**Actual:**  
Import errors occur during runtime.

**Suggested Fix:**
- Add missing dependencies to `requirements.txt` or a `requirements-extras.txt`.
- Provide a post-install validation step.

**Workaround Used:**
```bash
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install lap librosa pycocotools webvtt-py pyannote.core praatio openai-whisper
```

**Screenshot Placeholder:**  
![Missing dependency](<add-screenshot-here>)

---

## 4. 🔊 Speaker Diarization Requires Hugging Face Token

**Summary:**  
Audio pipeline fails to initialize without a Hugging Face auth token and terms acceptance for PyAnnote models.

**Steps to Reproduce:**
1. Enable diarization in the pipeline.
2. Run audio processing.
3. Error: “HuggingFace token required for PyAnnote models”.

**Expected:**  
- Clear prompt to set token or fallback behaviour if missing.

**Actual:**  
- Pipeline fails immediately without guidance.

**Suggested Fix:**
- Add `.env` support for `HF_AUTH_TOKEN`.
- Check for token on startup and print instructions.

**Workaround:**
```bash
echo 'export HF_AUTH_TOKEN="your_token_here"' >> ~/.bash_profile
source ~/.bash_profile
```

**Screenshot Placeholder:**  
![HF error](<add-screenshot-here>)

---

## 5. 🧠 Speech Recognition Segfault (Apple Silicon)

**Summary:**  
Speech recognition crashes with OpenMP errors on M1/M2 Macs.

**Error:**
```
OMP: Error #179: Function pthread_mutex_init failed
zsh: segmentation fault  python api_server.py
Speech recognition failed: No threading layer could be loaded.
```

**Steps to Reproduce:**
1. Run speech pipeline on Apple Silicon.
2. Observe OpenMP crash and segmentation fault.

**Expected:**  
- Compatible threading backend or graceful failure.

**Actual:**  
- Crash and abort due to missing `intel-openmp`.

**Suggested Fix:**
- Detect architecture and suggest installing Apple-compatible OpenMP:
```bash
brew install libomp
```
- Add pre-run environment validation.

**Screenshot Placeholder:**  
![OpenMP error](<add-screenshot-here>)

---

## 6. 📤 API Results Not Downloaded

**Summary:**  
Video processing completes, but no downloadable file is returned to the browser. Results appear only on the server filesystem.

**Steps to Reproduce:**
1. Upload and process a video.
2. Wait for pipeline completion.
3. No download prompt or returned file.

**Expected:**  
Results should be downloadable or a link should be displayed in the UI.

**Actual:**  
Processing completes successfully, but the frontend never receives the output.

**Suggested Fix:**
- Ensure API endpoint returns correct headers:
  - `Content-Type: application/octet-stream`
  - `Content-Disposition: attachment; filename="results.json"`
- Add download link or retry option to UI.

**Screenshot Placeholder:**  
![No download](<add-screenshot-here>)

---

## ✅ Summary of Action Items

| # | Issue | Priority | Status |
|---|---|---|---|
| 1 | Shell permission errors | 🔥 Critical | Needs installer fix |
| 2 | Node/NPM installation & script mismatch | ⚠️ High | Update docs |
| 3 | Missing Python dependencies | ⚠️ High | Update requirements |
| 4 | Diarization token requirement | ⚠️ High | Add auth check |
| 5 | Speech recognition segfault (M1/M2) | ⚠️ High | Add arch-specific install |
| 6 | API not returning results | ⚠️ Medium | Backend fix required |

