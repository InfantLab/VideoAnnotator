# Quickstart: checking provenance end to end

```bash
# 1. Run a job with the five baseline pipelines (no server needed)
uv run videoannotator process viewer/demo-assets/2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4 \
  --pipelines face_openface3_embedding,person_tracking,scene_detection,speaker_diarization,speech_recognition \
  --output /tmp/prov

# 2. Every JSON output has a record
for f in /tmp/prov/*.json; do python -c "import json,sys; p=json.load(open(sys.argv[1]))['provenance']; print(sys.argv[1], p['pipeline']['name'], p['videoannotator_version'], [m['revision'] for m in p['models']])" "$f"; done

# 3. The transcript carries it in a NOTE; the RTTM in a companion file
grep -m1 '^NOTE videoannotator-provenance' /tmp/prov/*_speech_recognition.vtt
cat /tmp/prov/*_speaker_diarization.rttm.provenance.json

# 4. Standard readers still work
python -c "from pycocotools.coco import COCO; import glob; [COCO(f) for f in glob.glob('/tmp/prov/*_person_tracking.json')]"

# 5. The job record has the same record per pipeline
uv run videoannotator job results <job id>   # or GET /api/v1/jobs/<id>/results

# 6. Viewer: open the job; each overlay shows "<pipeline> · VideoAnnotator <version>";
#    the info button shows the full record. Open tests/fixtures/viewer_contract/legacy/ files:
#    overlays say "version not recorded".
```
