---
name: "Frontend/Backend: API completes but no downloadable results returned"
about: "Processing finishes, UI shows success, but browser does not receive a downloadable file response."
labels: "bug, api, frontend"
assignees: ""
---

### Summary
Processing completes, but the UI never receives/downloads the result file. Outputs appear only on the server filesystem.

### Steps to Reproduce
1. Process a video via the UI.
2. Wait for completion.
3. No download is offered; no file in browser.

### Expected
- API responds with proper headers and body to trigger a download or presents a link in UI.

### Actual
- Results saved locally on server only; client doesn't get the file.

### Environment
- OS: macOS (client)
- Browser: (paste)
- Video Annotator commit: (paste)
- Backend logs: (attach)

### Suggested Fix
- Ensure the endpoint returns `Content-Disposition` and `Content-Type` correctly.
- Verify CORS and reverse proxy settings (if any).
- In UI, show an explicit download link with retry if the initial response is delayed.

### Screenshots / Logs
**Screenshot:**  
![No download returned](<add-screenshot-here>)

**Network Tab / Logs:**
```text
<attach failed request / response headers>
```

### Additional Context
If results are large, consider streaming or a separate “jobs” endpoint with polling and a final “download” link.
