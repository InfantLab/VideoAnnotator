"""Unit tests for native format exporters (COCO, WebVTT, RTTM, TextGrid)."""

import json
from pathlib import Path

from videoannotator.exporters.native_formats import (
    ValidationResult,
    _format_timestamp,
    _seconds_to_webvtt_time,
    auto_export_annotations,
    create_coco_annotation,
    create_coco_image_entry,
    create_coco_keypoints_annotation,
    export_coco_json,
    export_rttm_diarization,
    export_textgrid_speech,
    export_webvtt,
    export_webvtt_captions,
    validate_coco_format,
    validate_coco_json,
)

# ---------------------------------------------------------------------------
# COCO helper functions
# ---------------------------------------------------------------------------


class TestCreateCOCOImageEntry:
    """Tests for create_coco_image_entry."""

    def test_required_fields(self):
        entry = create_coco_image_entry(1, 640, 480, "frame_0001.jpg")
        assert entry["id"] == 1
        assert entry["width"] == 640
        assert entry["height"] == 480
        assert entry["file_name"] == "frame_0001.jpg"

    def test_extra_kwargs(self):
        entry = create_coco_image_entry(
            1, 640, 480, "frame.jpg", video_id="v1", frame_number=0
        )
        assert entry["video_id"] == "v1"
        assert entry["frame_number"] == 0


class TestCreateCOCOAnnotation:
    """Tests for create_coco_annotation."""

    def test_required_fields(self):
        ann = create_coco_annotation(1, 1, 1, [10, 20, 100, 200])
        assert ann["id"] == 1
        assert ann["image_id"] == 1
        assert ann["category_id"] == 1
        assert ann["bbox"] == [10, 20, 100, 200]
        assert ann["iscrowd"] == 0

    def test_auto_area(self):
        ann = create_coco_annotation(1, 1, 1, [0, 0, 50, 60])
        assert ann["area"] == 3000  # 50 * 60

    def test_explicit_area(self):
        ann = create_coco_annotation(1, 1, 1, [0, 0, 50, 60], area=999)
        assert ann["area"] == 999

    def test_extra_kwargs(self):
        ann = create_coco_annotation(1, 1, 1, [0, 0, 10, 10], score=0.95, track_id=3)
        assert ann["score"] == 0.95
        assert ann["track_id"] == 3


class TestCreateCOCOKeypointsAnnotation:
    """Tests for create_coco_keypoints_annotation."""

    def test_fields(self):
        kps = [100, 200, 2, 150, 250, 2]  # 2 keypoints
        ann = create_coco_keypoints_annotation(
            1, 1, 1, keypoints=kps, bbox=[10, 20, 100, 200], num_keypoints=2
        )
        assert ann["keypoints"] == kps
        assert ann["num_keypoints"] == 2
        assert ann["area"] == 100 * 200

    def test_extra_kwargs(self):
        kps = [0, 0, 2]
        ann = create_coco_keypoints_annotation(
            1, 1, 1, kps, [0, 0, 10, 10], 1, score=0.8
        )
        assert ann["score"] == 0.8


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------


class TestTimestampFormatting:
    """Tests for WebVTT timestamp helpers."""

    def test_seconds_to_webvtt_time_zero(self):
        assert _seconds_to_webvtt_time(0) == "00:00:00.000"

    def test_seconds_to_webvtt_time_simple(self):
        assert _seconds_to_webvtt_time(65.5) == "00:01:05.500"

    def test_seconds_to_webvtt_time_hours(self):
        result = _seconds_to_webvtt_time(3661.123)
        assert result == "01:01:01.123"

    def test_format_timestamp_zero(self):
        assert _format_timestamp(0) == "00:00:00.000"

    def test_format_timestamp_fractional(self):
        assert _format_timestamp(90.75) == "00:01:30.750"


# ---------------------------------------------------------------------------
# ValidationResult NamedTuple
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for the ValidationResult NamedTuple."""

    def test_creation(self):
        r = ValidationResult(is_valid=True, warnings=["w1"], errors=[])
        assert r.is_valid is True
        assert r.warnings == ["w1"]
        assert r.errors == []


# ---------------------------------------------------------------------------
# COCO JSON export + validation (end-to-end with tmp files)
# ---------------------------------------------------------------------------


class TestExportCOCOJson:
    """Tests for export_coco_json."""

    def test_creates_valid_file(self, tmp_path):
        images = [create_coco_image_entry(1, 640, 480, "f.jpg")]
        annotations = [create_coco_annotation(1, 1, 1, [0, 0, 50, 50])]
        out = str(tmp_path / "out.json")
        coco = export_coco_json(annotations, images, out)
        assert coco is not None
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 1

    def test_default_category(self, tmp_path):
        images = [create_coco_image_entry(1, 640, 480, "f.jpg")]
        annotations = [create_coco_annotation(1, 1, 1, [0, 0, 10, 10])]
        out = str(tmp_path / "out.json")
        export_coco_json(annotations, images, out)
        data = json.loads(Path(out).read_text())
        assert data["categories"][0]["name"] == "person"

    def test_custom_categories(self, tmp_path):
        images = [create_coco_image_entry(1, 640, 480, "f.jpg")]
        annotations = [create_coco_annotation(1, 1, 2, [0, 0, 10, 10])]
        cats = [{"id": 2, "name": "face", "supercategory": "person"}]
        out = str(tmp_path / "out.json")
        export_coco_json(annotations, images, out, categories=cats)
        data = json.loads(Path(out).read_text())
        assert data["categories"][0]["name"] == "face"


class TestValidateCOCOFormat:
    """Tests for validate_coco_format convenience function."""

    def test_valid(self, tmp_path):
        data = {
            "images": [{"id": 1, "width": 640, "height": 480, "file_name": "f.jpg"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        }
        path = tmp_path / "valid.json"
        path.write_text(json.dumps(data))
        assert validate_coco_format(str(path)) is True

    def test_invalid(self, tmp_path):
        path = tmp_path / "invalid.json"
        path.write_text("not json")
        assert validate_coco_format(str(path)) is False


class TestValidateCOCOJsonFunction:
    """Tests for validate_coco_json function from native_formats."""

    def test_valid_returns_true(self, tmp_path):
        data = {
            "images": [{"id": 1, "width": 640, "height": 480, "file_name": "f.jpg"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        result = validate_coco_json(str(path), context="test")
        assert result.is_valid is True

    def test_nonexistent_file(self):
        result = validate_coco_json("/no/such/file.json", context="test")
        assert result.is_valid is False
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# WebVTT export
# ---------------------------------------------------------------------------


class TestExportWebVTTCaptions:
    """Tests for export_webvtt_captions."""

    def test_creates_file(self, tmp_path):
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 3.0, "end": 5.0, "text": "Second caption"},
        ]
        out = str(tmp_path / "out.vtt")
        export_webvtt_captions(segments, out)
        content = Path(out).read_text()
        assert "WEBVTT" in content
        assert "Hello world" in content
        assert "Second caption" in content

    def test_empty_segments(self, tmp_path):
        out = str(tmp_path / "empty.vtt")
        export_webvtt_captions([], out)
        content = Path(out).read_text()
        assert "WEBVTT" in content


class TestExportWebVTT:
    """Tests for the export_webvtt function."""

    def test_basic_export(self, tmp_path):
        segments = [
            {"start_time": 0.0, "end_time": 1.5, "text": "Hi"},
        ]
        out = str(tmp_path / "test.vtt")
        assert export_webvtt(segments, out) is True
        content = Path(out).read_text()
        assert "Hi" in content

    def test_with_speaker(self, tmp_path):
        segments = [
            {"start_time": 0, "end_time": 1, "text": "Hello", "speaker_id": "SPK1"},
        ]
        out = str(tmp_path / "spk.vtt")
        export_webvtt(segments, out)
        content = Path(out).read_text()
        assert "SPK1" in content

    def test_with_transcription_key(self, tmp_path):
        segments = [
            {"start_time": 0, "end_time": 1, "transcription": "Transcribed text"},
        ]
        out = str(tmp_path / "tr.vtt")
        export_webvtt(segments, out)
        content = Path(out).read_text()
        assert "Transcribed text" in content

    def test_with_emotions(self, tmp_path):
        segments = [
            {
                "start_time": 0,
                "end_time": 1,
                "text": "Wow",
                "emotions": {"happy": {"score": 0.9}},
            },
        ]
        out = str(tmp_path / "emo.vtt")
        export_webvtt(segments, out, include_metadata=True)
        content = Path(out).read_text()
        assert "EMOTIONS" in content
        assert "happy" in content

    def test_metadata_disabled(self, tmp_path):
        segments = [
            {
                "start_time": 0,
                "end_time": 1,
                "text": "No meta",
                "emotions": {"happy": {"score": 0.9}},
            },
        ]
        out = str(tmp_path / "nometa.vtt")
        export_webvtt(segments, out, include_metadata=False)
        content = Path(out).read_text()
        assert "EMOTIONS" not in content


# ---------------------------------------------------------------------------
# RTTM export
# ---------------------------------------------------------------------------


class TestExportRTTMDiarization:
    """Tests for export_rttm_diarization."""

    def test_creates_rttm(self, tmp_path):
        segments = [
            {"start": 0.0, "end": 2.0, "speaker_id": "SPEAKER_01"},
            {"start": 2.5, "end": 5.0, "speaker_id": "SPEAKER_02"},
        ]
        out = str(tmp_path / "out.rttm")
        export_rttm_diarization(segments, out, uri="testvideo")
        content = Path(out).read_text()
        assert "SPEAKER_01" in content
        assert "SPEAKER_02" in content
        assert "SPEAKER" in content  # RTTM format keyword


# ---------------------------------------------------------------------------
# TextGrid export
# ---------------------------------------------------------------------------


class TestExportTextGridSpeech:
    """Tests for export_textgrid_speech."""

    def test_creates_textgrid(self, tmp_path):
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello"},
            {"start": 2.0, "end": 3.5, "text": "World"},
        ]
        out = str(tmp_path / "out.TextGrid")
        export_textgrid_speech(segments, out, tier_name="speech")
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "Hello" in content
        assert "World" in content


# ---------------------------------------------------------------------------
# Auto-export
# ---------------------------------------------------------------------------


class TestAutoExportAnnotations:
    """Tests for auto_export_annotations."""

    def test_person_bbox_exports_coco(self, tmp_path):
        annotations = [
            {"type": "person_bbox", "image_id": 1, "bbox": [0, 0, 50, 50]},
        ]
        result = auto_export_annotations(annotations, str(tmp_path), base_name="test")
        assert "coco" in result
        assert Path(result["coco"]).exists()

    def test_speech_exports_webvtt_and_textgrid(self, tmp_path):
        annotations = [
            {"type": "speech", "start": 0, "end": 1, "text": "Hi"},
        ]
        result = auto_export_annotations(annotations, str(tmp_path))
        assert "webvtt" in result
        assert "textgrid" in result

    def test_diarization_exports_rttm(self, tmp_path):
        annotations = [
            {"type": "diarization", "start": 0, "end": 2, "speaker_id": "SPK1"},
        ]
        result = auto_export_annotations(annotations, str(tmp_path))
        assert "rttm" in result

    def test_empty_annotations(self, tmp_path):
        result = auto_export_annotations([], str(tmp_path))
        assert result == {}

    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "new_subdir"
        annotations = [
            {"type": "person_bbox", "image_id": 1, "bbox": [0, 0, 10, 10]},
        ]
        auto_export_annotations(annotations, str(out))
        assert out.exists()
