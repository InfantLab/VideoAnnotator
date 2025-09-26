---
name: "Audio: speech recognition segfault / OpenMP error (Apple Silicon)"
about: "Intel OpenMP not available on Apple Silicon; OpenMP error leads to segfault in speech pipeline."
labels: "bug, audio, macOS, Apple-Silicon"
assignees: ""
---

### Summary
Speech recognition crashes with OpenMP error and segmentation fault on Apple Silicon.

### Steps to Reproduce
1. Run speech recognition pipeline on macOS (M1/M2).
2. Observe OpenMP error and segfault.

### Expected
- Apple Silicon-compatible threading backend / graceful fallback.

### Actual
- Immediate failure with OpenMP error and segfault.

### Environment
- OS: macOS (Apple Silicon)
- Python: (paste)
- Video Annotator commit: (paste)

### Suggested Fix
- Detect architecture and suggest/install Apple-compatible OpenMP (`libomp`):
  ```bash
  brew install libomp
  ```
- Consider linking against Accelerate/vecLib or alternative threading backends.
- Add defensive checks to avoid hard crash.

### Screenshots / Logs
**Screenshot:**  
![OpenMP error](<add-screenshot-here>)

**Logs:**
```text
OMP: Error #179: Function pthread_mutex_init failed
zsh: segmentation fault  python api_server.py
Speech recognition failed: No threading layer could be loaded.
```
