# Issue: Missing Video File in Job Artifacts ZIP

**Status:** Open  
**Priority:** High  
**Date:** 2025-12-15  
**Component:** API / Job Artifacts

## Description

**Endpoint:** `GET /api/v1/jobs/{job_id}/artifacts`

**Issue:** The artifacts ZIP downloaded from this endpoint does not contain the source video file (e.g., `.mp4`, `.mov`, `.avi`), or it is in an unexpected format.

## Observed Behavior
The frontend `JobResultsViewer` fails with "No video file found in artifacts ZIP" because it cannot find a video file in the downloaded ZIP.

## Impact
Users cannot view job results in the web UI because the viewer requires the video file to be present in the artifacts ZIP to initialize the video player.

## Expected Behavior
The artifacts ZIP should contain the source video file so the frontend can play it back alongside the annotations.

## Questions for Server Team
1. Is the video file intentionally excluded from the artifacts ZIP?
2. If so, how should the frontend retrieve the video for playback? (Is there a separate endpoint?)
3. If the video is included, what is the file extension? (Frontend currently checks for `.mp4`, `.mov`, `.avi`).
