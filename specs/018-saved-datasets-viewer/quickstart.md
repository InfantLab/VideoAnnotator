# Quickstart: saved datasets

1. Start the dev stack: `bash scripts/dev.sh`, open http://127.0.0.1:19011.
2. New job → Choose videos → pick a folder of videos → **Save as dataset** "Pilot".
3. Run the job; its page shows it came from "Pilot".
4. Datasets page: "Pilot" lists its videos, owner, created/last used. Rename it.
5. New job → **Use a saved dataset** → "Pilot": the same videos are selected (re-pick the folder
   if the browser asks). Rename one video file on disk first to see the differences dialog.
6. Export "Pilot" → import the file → a copy "Pilot (imported)" appears.
7. `uv run videoannotator dataset list` shows both.
