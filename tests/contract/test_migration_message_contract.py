"""Contract test: the migration-message variant for pipelines demoted out of
the v1.4.4 default install (LAION face/voice, OpenFace3).

Contract: specs/004-extras-based-install/contracts/unavailable-pipeline-error.md
  "Migration case (pipeline demoted from v1.4.4 default install)"
Data:     specs/004-extras-based-install/data-model.md
  "Migration message record"
"""

import pytest

from videoannotator.registry.pipeline_loader import install_hint, migration_note


class TestMigrationMessageContract:
    @pytest.mark.parametrize(
        "pipeline_name,expected_extra",
        [
            ("face_laion_clip", "face-laion"),
            ("laion_voice", "audio-laion"),
            ("face_openface3_embedding", "face-openface3"),
        ],
    )
    def test_demoted_pipelines_get_a_migration_note(
        self, pipeline_name, expected_extra
    ):
        note = migration_note(pipeline_name)

        assert note is not None
        assert "no longer installed by default" in note
        assert expected_extra in note
        assert "v1.5.0" in note

    @pytest.mark.parametrize(
        "pipeline_name",
        [
            "face_analysis",
            "scene_detection",
            "person_tracking",
            "audio_processing",
            "speech_recognition",
            "speaker_diarization",
            "never_existed_pipeline",
        ],
    )
    def test_non_demoted_pipelines_get_no_migration_note(self, pipeline_name):
        assert migration_note(pipeline_name) is None

    def test_migration_note_is_distinguishable_from_generic_message(self):
        """A demoted pipeline's note must not read like the plain
        'not a recognized pipeline' case — it must explain the pipeline
        *used to* work and name the extras group to restore it."""
        note = migration_note("face_laion_clip")
        generic_unavailable_message = (
            "Pipeline 'some_pipeline' is not available in this install."
        )

        assert note != generic_unavailable_message
        assert "face-laion" in note
        assert install_hint(["face-laion"]) == "pip install videoannotator[face-laion]"

    def test_full_message_shape_matches_contract_example(self):
        """Assemble the same message shape the API/CLI error path builds
        (exceptions.PipelineUnavailableException) and check it matches the
        three required parts from the contract: naming the pipeline, the
        'no longer default' explanation, and the exact install command."""
        pipeline_name = "face_laion_clip"
        requires_extras = ["face-laion"]

        note = migration_note(pipeline_name)
        message = f"Pipeline '{pipeline_name}' is not available in this install."
        if note:
            message = f"{message} {note}"
        hint = install_hint(requires_extras)

        assert message.startswith(
            "Pipeline 'face_laion_clip' is not available in this install."
        )
        assert "no longer installed by default" in message
        assert "v1.5.0" in message
        assert "face-laion" in message
        assert hint == "pip install videoannotator[face-laion]"
