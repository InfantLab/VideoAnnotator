---
name: "Backend: missing Python dependencies (lap, librosa, etc.)"
about: "API server fails due to missing modules: lap, librosa, pycocotools, webvtt-py, pyannote.core, praatio, openai-whisper."
labels: "bug, backend, dependencies"
assignees: ""
---

### Summary
Backend fails to start/execute due to missing Python modules (e.g., `lap`, `librosa`) and other optional components.

### Steps to Reproduce
1. Create venv and install with current instructions.
2. Run API server.
3. Observe `ModuleNotFoundError` for various packages.

### Expected
- All required modules installed automatically for core functionality.
- Optional features documented and gated behind flags.

### Actual
- Import errors at runtime for multiple packages.

### Environment
- OS: macOS
- Python: (paste `python -V`)
- Package manager: uv/pip/conda (paste)
- Video Annotator commit: (paste)

### Suggested Fix
- Update `requirements.txt` and/or extras to include missing modules, e.g.:
  ```txt
  lap
  librosa
  pycocotools
  webvtt-py
  pyannote.core
  praatio
  openai-whisper
  ```
- Provide `requirements-extras.txt` for optional pipelines.
- Add a startup self-check that reports missing dependencies with install hints.

### Workaround Used
```bash
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install lap librosa pycocotools webvtt-py pyannote.core praatio openai-whisper
```

### Screenshots / Logs
**Screenshot:**  
![ModuleNotFoundError lap](<add-screenshot-here>)

**Traceback:**
```text
ModuleNotFoundError: No module named 'lap'
ModuleNotFoundError: No module named 'librosa'
```
