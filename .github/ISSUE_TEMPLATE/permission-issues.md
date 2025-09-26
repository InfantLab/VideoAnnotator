---
name: "Permission issues — shell config & ~/.config (macOS)"
about: "Installer can't write to .zshrc or ~/.config due to ownership/permissions problems."
labels: "bug, macOS, setup"
assignees: ""
---

### Summary
Installer fails to add `uv` to PATH and cannot create/write required config files on macOS (root-owned `~/.zshrc` and/or `~/.config`).

### Steps to Reproduce
1. Run the installer / launch backend on macOS.
2. Attempt to edit `~/.zshrc` to add PATH.
3. Observe write failures and/or missing `~/.config` with incorrect ownership.

### Expected
- Installer creates/updates shell config and `~/.config` without manual intervention.

### Actual
- `~/.zshrc` and/or `~/.config` owned by `root`; write attempts fail.
- PATH not updated (e.g., `uv` not found).

### Environment
- OS: macOS (Apple Silicon or Intel)
- Shell: zsh
- Video Annotator commit: (paste)
- Python: (paste)

### Suggested Fix
- Detect and warn if `~/.zshrc` or `~/.config` ownership != `$USER`.
- Offer auto-fix (with prompt) or print commands:
  ```bash
  sudo chown -R $USER ~/.zshrc
  chmod u+w ~/.zshrc
  sudo chown -R $USER ~/.config
  ```
- Write to `.zprofile` if `.zshrc` is missing (login shells).

### Screenshots / Logs
**Screenshot:**  
![Permission error](<add-screenshot-here>)

**Logs:**
```bash
ls -l ~/.zshrc
ls -ld ~/.config
```

### Additional Context
Root ownership likely came from running parts of setup with `sudo`. Consider documenting a strict “no sudo” requirement for user-level setup.
