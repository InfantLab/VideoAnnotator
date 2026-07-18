# Quickstart: Verifying the Extras-Based Install

Manual/CI verification steps mapped to spec 004's acceptance scenarios.

## 1. Slim single-family install (User Story 1)

```bash
python -m venv /tmp/va-scene && source /tmp/va-scene/bin/activate
pip install videoannotator[scene]
pip show torch open-clip-torch scenedetect  # present — scene needs them (research.md §1)
pip show pyannote.audio ultralytics deepface  # MUST be absent
videoannotator pipelines list                 # only scene_detection shown
videoannotator job submit sample.mp4 --pipelines scene_detection   # succeeds
videoannotator job submit sample.mp4 --pipelines face_analysis     # actionable error, no traceback
```

## 2. Additive install (User Story 1, scenario 4)

```bash
pip install videoannotator[face]
videoannotator pipelines list   # scene_detection AND face_analysis now shown; nothing reset
```

## 3. Full-parity install (User Story 2)

```bash
pip install videoannotator[all]
pytest tests/ -k acceptance   # v1.4.4 fixtures pass byte-identical / within documented tolerance
```

## 4. LAION migration message (User Story 2, scenario 2)

```bash
pip install videoannotator[face]   # face, not face-laion
videoannotator job submit sample.mp4 --config old_v144_config_with_laion.yaml
# expect: "no longer installed by default as of v1.5.0" message naming `face-laion`, not a raw ImportError
```

## 5. NumPy 2.x

```bash
pip install videoannotator[all] "numpy>=2.0"
pytest tests/ -x
```

## 6. Forward-compatibility gate (User Story 3 / SC-007)

```bash
# Author a throwaway stub pipeline YAML with requires_extras: [] and a module_path
# pointing at a NotImplementedError stub; confirm registry loads it without code changes.
cp tests/fixtures/stub_pipeline.yaml src/videoannotator/registry/metadata/ # or a test-only registry dir
python -c "from videoannotator.registry import get_registry; get_registry().load(force=True); print('ok')"
```
