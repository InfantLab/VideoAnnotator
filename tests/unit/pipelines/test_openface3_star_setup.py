"""OpenFace's STAR config adds root-logger handlers (and a TensorBoard dir
under a hard-coded upstream path) on every LandmarkDetector; the pipeline
must skip that so server log lines don't multiply per job."""

import logging

import pytest

pytest.importorskip("openface", reason="requires the `face-openface3` extra")

from videoannotator.pipelines.face_analysis.openface3_pipeline import (
    _without_star_training_setup,
)


def test_handlers_and_level_added_inside_are_undone():
    root = logging.getLogger()
    before_handlers, before_level = list(root.handlers), root.level
    added = logging.StreamHandler()

    with _without_star_training_setup():
        root.addHandler(added)
        root.setLevel(logging.NOTSET)

    assert root.handlers == before_handlers
    assert root.level == before_level


def test_star_training_setup_is_skipped_then_restored():
    from openface.STAR.conf.base import Base

    original = Base.init_instance
    config = Base("alignment")
    with _without_star_training_setup():
        config.init_instance()
    assert config.logger is None and config.writer is None
    assert Base.init_instance is original


def test_existing_handlers_are_kept():
    root = logging.getLogger()
    existing = logging.NullHandler()
    root.addHandler(existing)
    try:
        with _without_star_training_setup():
            pass
        assert existing in root.handlers
    finally:
        root.removeHandler(existing)
