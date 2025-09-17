from src.registry.pipeline_registry import get_registry

def test_voice_emotion_pipeline_present():
    reg = get_registry()
    reg.load(force=True)
    vm = {m.name: m for m in reg.list()}["voice_emotion_baseline"]
    assert "emotion-recognition" in vm.tasks
    assert "speech-transcription" in vm.tasks
    assert vm.pipeline_family == "audio"
    assert any(o.format == "JSON" for o in vm.outputs)


def test_interaction_analysis_task_vocab_exists():
    # Ensure future planning task exists in vocab for forward compat
    from src.registry.constants import TASKS
    assert "interaction-analysis" in TASKS
