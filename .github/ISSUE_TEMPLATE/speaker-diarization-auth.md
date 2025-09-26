---
name: "Audio: speaker diarization requires Hugging Face auth token"
about: "PyAnnote models require terms acceptance and HF token; pipeline fails without it."
labels: "bug, audio, dependencies"
assignees: ""
---

### Summary
Speaker diarization fails to initialize due to missing Hugging Face authentication for pyannote models.

### Steps to Reproduce
1. Enable diarization in audio pipeline.
2. Run processing.
3. Observe error about missing HF token / terms acceptance.

### Expected
- Clear prompt for HF token or a graceful fallback (diarization disabled).

### Actual
- Pipeline fails during initialization.

### Environment
- OS: macOS
- Python: (paste)
- Video Annotator commit: (paste)

### Suggested Fix
- Support `.env` with `HF_AUTH_TOKEN`.
- On startup, detect missing token and print clear guidance:
  1) Create a Hugging Face account.
  2) Accept pyannote terms.
  3) Generate a token and set env variable.
- Provide offline/no-auth fallback where possible.

### Workaround Used
```bash
# After creating token & accepting terms
echo 'export HF_AUTH_TOKEN="xxxxxxxxx"' >> ~/.bash_profile
source ~/.bash_profile
```

### Screenshots / Logs
**Screenshot:**  
![HF token error](<add-screenshot-here>)

**Logs:**
```text
ERROR: Failed to initialize speaker_diarization: HuggingFace token required for PyAnnote models
```
