"""Contract test: VideoAnnotator's real output vs. Video Annotation Viewer's parsers.

This does not import or execute VAV's code (it's a separate TypeScript repo — that
half of the contract is exercised in VAV's own CI against fixtures this test
generates). Instead it models VAV's actual Zod validation schemas in Python,
sourced directly from InfantLab/video-annotation-viewer:

- COCOPersonAnnotationSchema: src/lib/validation.ts (~line 20)
- WebVTT/RTTM syntax:         src/lib/parsers/{webvtt,rttm}.ts

Fixtures are produced via VideoAnnotator's real exporters
(videoannotator.exporters.native_formats), following the exact code paths
used by the person-tracking pipeline (with and without identity labeling
enabled — see src/videoannotator/pipelines/person_tracking/person_pipeline.py
around lines 456-501), not hand-rolled JSON. If VideoAnnotator's output shape
drifts from what VAV requires, this test fails on the VideoAnnotator side
before a researcher ever sees a broken viewer.

History: the first run of this test (before the person_pipeline.py fix in the
same change) caught a real, live gap — without identity labeling configured,
person_id/person_label/label_confidence/labeling_method were omitted entirely,
which VAV's schema requires as non-optional. The pipeline now always populates
them with fallback defaults ("track_<id>"/"unknown"/0.0/"none"); this test
locks that fix in.
"""

import json

import pytest

from videoannotator.exporters.native_formats import (
    create_coco_image_entry,
    create_coco_keypoints_annotation,
    export_coco_json,
    export_rttm_diarization,
    export_webvtt_captions,
)

# Mirrors COCOPersonAnnotationSchema in video-annotation-viewer's src/lib/validation.ts.
# Each entry: (field, required, expected_python_types).
VAV_COCO_PERSON_CONTRACT = [
    ("id", True, (int,)),
    ("image_id", True, (str,)),  # Zod: z.string() — NOT the "integer" VideoAnnotator's
    # own get_schema() docstring used to claim; runtime value is an f-string. Fixed
    # alongside this test (person_pipeline.py get_schema()).
    ("category_id", True, (int,)),
    ("keypoints", True, (list,)),  # z.array(z.number()).length(51)
    ("num_keypoints", True, (int,)),
    ("bbox", True, (list, tuple)),
    ("area", True, (int, float)),
    ("iscrowd", True, (int,)),
    ("score", True, (int, float)),
    ("track_id", False, (int,)),  # z.number().optional() — NOT z.nullable(); an
    # explicit `null` fails Zod validation, only a missing key is tolerated.
    ("timestamp", True, (int, float)),
    ("frame_number", True, (int, float)),
    ("person_id", True, (str,)),
    ("person_label", True, (str,)),
    ("label_confidence", True, (int, float)),
    ("labeling_method", True, (str,)),
]


def _build_annotation(*, with_identity: bool, track_id: int) -> dict:
    """Build one annotation exactly as person_pipeline.py's fixed code does.

    Mirrors PersonTrackingPipeline._process_frame's keypoints branch (lines
    456-468) followed by the identity-labeling block (lines ~483-501), which
    now always sets person_id/person_label/label_confidence/labeling_method —
    falling back to "track_<id>"/"unknown"/0.0/"none" when identity_manager is
    absent or has no resolved label for this track, rather than omitting them.
    """
    annotation = create_coco_keypoints_annotation(
        annotation_id=1,
        image_id="video123_frame_42",
        category_id=1,
        keypoints=[float(i % 10) for i in range(51)],
        bbox=[10.0, 20.0, 30.0, 40.0],
        num_keypoints=17,
        score=0.91,
        track_id=track_id,
        timestamp=1.4,
        frame_number=42,
    )
    if with_identity:
        annotation["person_id"] = "person_video123_001"
        annotation["person_label"] = "child"
        annotation["label_confidence"] = 0.82
        annotation["labeling_method"] = "automatic_size_based"
    else:
        # Fallback defaults applied by person_pipeline.py when identity_manager
        # is None or has no label for this track yet.
        annotation["person_id"] = f"track_{track_id}"
        annotation["person_label"] = "unknown"
        annotation["label_confidence"] = 0.0
        annotation["labeling_method"] = "none"
    return annotation


def _check_against_vav_contract(annotation: dict) -> list[str]:
    """Return a list of violations of VAV's COCOPersonAnnotationSchema, if any."""
    violations = []
    for field, required, expected_types in VAV_COCO_PERSON_CONTRACT:
        if field not in annotation:
            if required:
                violations.append(f"missing required field '{field}'")
            continue
        value = annotation[field]
        if value is None:
            # Zod .optional() (not .nullable()) rejects an explicit null.
            violations.append(f"field '{field}' is null, which Zod .optional() rejects")
            continue
        if not isinstance(value, expected_types):
            violations.append(
                f"field '{field}' is {type(value).__name__}, expected one of {expected_types}"
            )
    return violations


def test_coco_person_annotation_matches_vav_contract_with_identity_labeling(tmp_path):
    """Identity labeling enabled and a label resolved for this track."""
    annotation = _build_annotation(with_identity=True, track_id=7)
    image = create_coco_image_entry(
        image_id="video123_frame_42", width=1920, height=1080, file_name="frame_42.jpg"
    )
    out_path = tmp_path / "person_tracking.json"
    export_coco_json([annotation], [image], str(out_path))

    saved = json.loads(out_path.read_text())
    violations = _check_against_vav_contract(saved["annotations"][0])

    assert not violations, (
        "VideoAnnotator's COCO output no longer matches Video Annotation Viewer's "
        f"COCOPersonAnnotationSchema: {violations}"
    )


def test_coco_person_annotation_without_identity_labeling_matches_vav_contract(
    tmp_path,
):
    """Regression test: identity labeling disabled (or track unlabeled) still
    produces schema-compliant output, thanks to the fallback defaults in
    person_pipeline.py. This used to fail before that fix.
    """
    annotation = _build_annotation(with_identity=False, track_id=3)
    image = create_coco_image_entry(
        image_id="video123_frame_42", width=1920, height=1080, file_name="frame_42.jpg"
    )
    out_path = tmp_path / "person_tracking.json"
    export_coco_json([annotation], [image], str(out_path))

    saved = json.loads(out_path.read_text())
    violations = _check_against_vav_contract(saved["annotations"][0])

    assert not violations, (
        "VideoAnnotator's COCO output without identity labeling no longer matches "
        f"Video Annotation Viewer's COCOPersonAnnotationSchema: {violations}"
    )


def test_webvtt_export_is_well_formed(tmp_path):
    """Syntactic check matching VAV's src/lib/parsers/webvtt.ts cue-block parsing."""
    out_path = tmp_path / "speech.vtt"
    export_webvtt_captions(
        [
            {"start": 1.0, "end": 3.5, "text": "Hello, how are you doing today?"},
            {"start": 4.0, "end": 6.2, "text": "I'm doing great, thanks for asking."},
        ],
        str(out_path),
    )
    content = out_path.read_text()

    assert content.startswith("WEBVTT")
    # VAV's parseTimestamp expects exactly HH:MM:SS.mmm (3 colon-separated parts).
    for line in content.splitlines():
        if "-->" in line:
            start, _, end = line.partition("-->")
            for ts in (start.strip(), end.strip().split(" ")[0]):
                parts = ts.split(":")
                assert len(parts) == 3, f"timestamp '{ts}' is not HH:MM:SS.mmm"


def test_rttm_export_is_well_formed(tmp_path):
    """Syntactic check matching VAV's src/lib/parsers/rttm.ts line parsing.

    parseRTTMLine requires >= 9 whitespace-separated fields and the literal
    type token 'SPEAKER'.
    """
    out_path = tmp_path / "speakers.rttm"
    export_rttm_diarization(
        [
            {"start": 1.25, "end": 3.55, "speaker_id": "SPEAKER_00"},
            {"start": 3.8, "end": 5.3, "speaker_id": "SPEAKER_01"},
        ],
        str(out_path),
        uri="video123",
    )
    content = out_path.read_text()
    lines = [line for line in content.splitlines() if line.strip()]
    assert lines, "RTTM export produced no lines"
    for line in lines:
        parts = line.split()
        assert len(parts) >= 9, (
            f"RTTM line has {len(parts)} fields, VAV requires >= 9: {line!r}"
        )
        assert parts[0] == "SPEAKER", f"unexpected RTTM type token: {parts[0]!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
