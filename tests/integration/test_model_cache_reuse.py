"""Integration test: the extras-aware registry/loader never touches the
user's model cache (HF_HOME / TORCH_HOME) — a pre-existing cache directory
is left exactly as-is, not cleared or re-populated, when pipelines are
loaded (User Story 2 / FR-015-equivalent guarantee, plan.md's Storage note).

This doesn't need real models: it plants a marker file standing in for a
cached model, points HF_HOME/TORCH_HOME at that directory, runs the full
registry+loader load path, and asserts the marker file (and its mtime) is
untouched.
"""

import os

from videoannotator.registry.pipeline_loader import get_pipeline_loader
from videoannotator.registry.pipeline_registry import get_registry


class TestModelCacheReuse:
    def test_registry_load_does_not_touch_hf_cache(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf_cache"
        hf_home.mkdir()
        cached_model = hf_home / "models--fake--cached-model" / "snapshot.bin"
        cached_model.parent.mkdir(parents=True)
        cached_model.write_bytes(b"pretend this is a downloaded model weight")
        mtime_before = cached_model.stat().st_mtime

        monkeypatch.setenv("HF_HOME", str(hf_home))

        registry = get_registry()
        registry.load(force=True)

        assert cached_model.exists()
        assert cached_model.stat().st_mtime == mtime_before
        assert cached_model.read_bytes() == b"pretend this is a downloaded model weight"
        assert os.environ["HF_HOME"] == str(hf_home)

    def test_pipeline_loader_does_not_touch_torch_cache(self, tmp_path, monkeypatch):
        torch_home = tmp_path / "torch_cache"
        torch_home.mkdir()
        cached_weight = torch_home / "hub" / "checkpoints" / "yolo11n.pt"
        cached_weight.parent.mkdir(parents=True)
        cached_weight.write_bytes(b"pretend this is a cached torch checkpoint")
        mtime_before = cached_weight.stat().st_mtime

        monkeypatch.setenv("TORCH_HOME", str(torch_home))

        loader = get_pipeline_loader()
        loader.load_all_pipelines()

        assert cached_weight.exists()
        assert cached_weight.stat().st_mtime == mtime_before
        assert os.environ["TORCH_HOME"] == str(torch_home)
