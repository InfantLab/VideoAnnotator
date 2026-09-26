# Handoff to video-annotation-viewer: HEVC videos show the "can't play this codec" message

**For**: the video-annotation-viewer agent. **Wanted for**: the viewer bundle that ships in
VideoAnnotator v1.5.0 (`src/videoannotator/viewer_static/`), before the release e2e run
(`tests/manual/v1.5.0_release_e2e.md`, step 10).

## The report

With the current bundle (commit `22ac9ca`, built from the v0.7.x viewer), opening results for these
videos shows:

> This browser can't play this video's codec (H.265/HEVC). Annotations are still available below.
> To see the video, re-encode it to H.264 …

The user is certain **the same files were annotated and displayed properly in the viewer before**.
H.264 `.mov` files play fine. Find out why these videos no longer display, and fix it if the
viewer causes it. If the browser really can't decode them, the message has to say so accurately.

## What the backend side has established

- **The files really are HEVC.** Example: `2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4`, 844 KB,
  10.6 s: HEVC Main, level 3.0, 640×480, yuv420p, **`hev1`** sample entry (not `hvc1`), libx265
  (Lavc61.13); AAC-LC audio; brand `isom`. Odd frame rate `37750/2629` (≈14.36 fps), probably
  variable frame rate. The other `*_h265.mp4` files in that corpus are the same.
- **The bytes the viewer gets are the original file.** No transcoding anywhere. The viewer fetches
  `GET /api/v1/jobs/{id}/artifacts` (zip, source video included), and that endpoint hasn't changed
  since 2025-12-15.
- **Old and new bundles load video identically**: `new File([blob], filename, {type: "video/mp4"})`
  → `URL.createObjectURL` → `<video src>`. The only video-related change in `22ac9ca` is the new
  detection:
  - `onLoadedMetadata`: `videoWidth === 0` → show the message;
  - `onError`: `MEDIA_ERR_SRC_NOT_SUPPORTED` or `MEDIA_ERR_DECODE` → show the message;
  - "(H.265/HEVC)" rather than "(often H.265/HEVC)" when the **filename** matches
    `/(h\.?265|hevc|x265)/i`. That's why this report names HEVC outright: it's the filename, not a
    codec check.
- **Contrary evidence**: the spec 011 e2e run, before the message existed, recorded
  `2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4` as "annotations loaded but a black video area"
  (`viewer-handoff-e2e-findings.md`, §4). So at least once already, the browser didn't show it.

## Questions to answer

1. **Is it the browser?** On the user's setup (Windows host, Chrome or Edge, viewer served from the
   dev container at `http://localhost:18011/viewer`): does the same `.mp4` play when dragged
   straight into a tab? What does `chrome://gpu` → *Video Acceleration Information* list for
   HEVC? Does it change with hardware acceleration off? Firefox?
2. **Is the detection giving false positives?** Check each:
   - a late `error` event from a **previous, revoked** blob URL arriving after the state was reset
     for the new file. The effect sets `C(false)` and then revokes the old URL on cleanup; the flag
     is never cleared again;
   - `videoWidth === 0` at `loadedmetadata` for a stream that decodes fine a moment later
     (possible with `hev1`, where the parameter sets are in-band);
   - the message never clearing once frames do arrive (`loadeddata`/`playing` with
     `videoWidth > 0`);
   - React re-renders creating a new `File`/object URL for the same video.
3. **Was there ever another playback path?** Search the viewer's git history for anything that gave
   `<video>` a different source: a server URL rather than a blob, a different MIME type, the
   standalone (drop files in) mode, the Vite dev server. The user may have seen these play through
   one of those.
4. **Does `hev1` vs `hvc1` matter here?** Test a lossless remux,
   `ffmpeg -i in.mp4 -c copy -tag:v hvc1 out.mp4`, in the same browser.

For the diagnosis, log on error: `video.error.code` and `video.error.message` (Chrome's message names
the demuxer or decoder failure); `video.canPlayType('video/mp4; codecs="hev1.1.6.L90.B0"')` and the
`hvc1` equivalent; and `navigator.mediaCapabilities.decodingInfo(...)` for the same codec. A bare
`hvc1` isn't a valid codec string (Chrome returns `""` even where HEVC plays).

## What to fix for v1.5.0

- **If the viewer causes it** (false positive, stale error, regression): fix it, so these files
  play wherever the browser can decode HEVC.
- **In any case**, make the detection trustworthy:
  - show the message only for the current source, and clear it once frames decode;
  - name the codec only from evidence. The filename heuristic can say "the file name suggests
    H.265"; better still, use what `canPlayType` or `mediaCapabilities` report;
  - include the browser's own error message in a collapsible "details" line, so the next report
    is diagnosable from a screenshot.
- **If the browser genuinely can't decode it**, give the fixes in order of effort: Chrome or Edge
  with hardware acceleration (Windows: "HEVC Video Extensions"); a lossless `hvc1` remux if that
  turns out to help; re-encode to H.264.

**Backend help on offer**, if it makes the viewer's job simpler: VideoAnnotator can probe the codec
at upload and return it with the job (`codec`, `codec_tag`, `profile`, `width`, `height`,
`fps`), so the viewer knows before playback. An H.264 preview copy made at upload is on the v1.6.0
roadmap, not v1.5.0. Say which you want in the handback.

## Handback

- The cause, with evidence (question 1–4 answers).
- A rebuilt bundle for `src/videoannotator/viewer_static/`.
- Done when: on the user's machine, the `*_h265.mp4` videos either **play**, or show a message that
  matches what the browser actually reported; H.264 `.mov`/`.mp4` still play; switching between
  videos in the results view never leaves a stale message behind.

---

## Handback (2026-09-26, video-annotation-viewer)

**Cause: the browser, not the viewer.** Tested on the reference machine (Windows 11; HEVC Video
Extensions 2.5.33 installed; default browser Edge) with the demo file
`2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4`, loaded the way the viewer loads it (`File` → object
URL → `<video>`), and then through the real viewer:

1. **Is it the browser?** Yes.
   - **Chrome 153** decodes it: `hev1` and an `hvc1` remux both play (32 frames in 2 s, no error).
     The viewer shows the video and no message.
   - **Edge 155**, headless and in a normal window, says it can (`canPlayType` "probably",
     `mediaCapabilities` supported/smooth) and then fails: `MEDIA_ERR_DECODE`,
     `PipelineStatus::PIPELINE_ERROR_DISCONNECTED`, 0 frames. H.264 plays in Edge. Your default
     browser is Edge, so `localhost:18011/viewer` most likely opens there; the "black video area" in
     the first e2e run fits.
   - **Chrome with GPU decode off** (`--disable-accelerated-video-decode`): audio plays, `videoWidth`
     stays 0, no frames. So Chrome's HEVC depends on hardware decoding.
2. **False positives?** One real one found and fixed, though it wasn't the cause here: an element
   rendered with `src=""` (before the object URL exists) fires `MEDIA_ERR_SRC_NOT_SUPPORTED`,
   "Empty src attribute", when the URL arrives more than a few ms later. It didn't trigger in these
   runs. Also fixed: the flag is now keyed to the current object URL (no stale or revoked-URL
   events), `videoWidth === 0` is checked at `loadeddata` not `loadedmetadata`, and the message
   clears once frames decode (on `timeupdate` with the clock moving, width > 0 and no error; not on
   `playing`, which Edge fires even after its decode error).
3. **Another playback path?** No. Results, the library and file-drop mode all use the same
   object-URL `<video>`.
4. **`hev1` vs `hvc1`?** No difference in either browser, so a remux isn't offered as a fix.

**The message now** names the codec from the file's own MP4 sample entry (`hev1` → H.265/HEVC), not
the file name. When the browser claims support and fails anyway it says so. Fixes in order: Chrome
with hardware acceleration on (Edge can fail even with the HEVC extension installed), then re-encode
to H.264. A Details line gives the browser's own error, the codec found, what the browser claims,
and the user agent. Checked in the real viewer: Chrome HEVC plays with no message; Edge HEVC shows
it; H.264 plays in both; switching from the HEVC file to the H.264 one in Edge clears the message.

**Backend codec probe**: not needed for v1.5.0, since the viewer reads the codec from the file. An
H.264 preview copy (v1.6.0) is still the real fix for Edge users.

Viewer commit and bundle: see the VideoAnnotator commit that refreshes `viewer_static/`.
