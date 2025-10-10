from src.registry import constants
from src.registry.pipeline_registry import get_registry


def test_registry_loads_with_extended_fields():
    reg = get_registry()
    reg.load(force=True)
    metas = {m.name: m for m in reg.list()}
    # Ensure new face pipelines present
    assert "face_openface3_embedding" in metas
    assert "face_laion_clip" in metas

    face_meta = metas["face_laion_clip"]
    assert "emotion-recognition" in face_meta.tasks
    assert face_meta.pipeline_family == "face"
    assert face_meta.variant == "laion-clip-face"
    # Outputs still intact
    assert any(o.format == "JSON" for o in face_meta.outputs)


def test_vocabulary_is_superset():
    reg = get_registry()
    reg.load(force=True)
    for m in reg.list():
        # Soft check: values should be strings and (if present) in vocabulary sets
        for t in m.tasks:
            assert isinstance(t, str)
            assert t in constants.TASKS
        for c in m.capabilities:
            assert c in constants.CAPABILITIES
        for mod in m.modalities:
            assert mod in constants.MODALITIES
        for b in m.backends:
            assert b in constants.BACKENDS
        if m.stability:
            assert m.stability in constants.STABILITY
