---
name: "Frontend: Node/NPM missing and incorrect start script"
about: "Viewer fails on fresh macOS – npm not found; docs suggest `npm start` but project uses `npm run dev`."
labels: "bug, documentation, frontend"
assignees: ""
---

### Summary
On fresh macOS, `npm` is missing and the README suggests `npm start` while the actual script is `npm run dev` inside `video-annotation-viewer`.

### Steps to Reproduce
1. Clone repo.
2. In `video-annotation-viewer`, run `npm start`.
3. Observe: `zsh: command not found: npm` or `npm ERR! Missing script: "start"`.

### Expected
- Pre-flight check for Node/NPM.
- Correct docs/instructions for starting the dev server.

### Actual
- No Node present by default; wrong script in docs.

### Environment
- OS: macOS
- Node: (paste `node -v`)
- NPM: (paste `npm -v`)
- Video Annotator commit: (paste)

### Suggested Fix
- Add Node version check to setup: `node -v && npm -v`.
- Update docs:
  ```bash
  cd video-annotation-viewer
  npm install
  npm run dev
  ```
- Optionally recommend Homebrew install:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  brew install node
  ```

### Screenshots / Logs
**Screenshot:**  
![npm missing](<add-screenshot-here>)

**Logs:**
```bash
npm run
npm start
```

### Additional Context
Consider pinning a minimal Node version (e.g., `>= 20`) and surfacing it in the root README.
