"""Unit tests for COCO format validation."""

import json
from pathlib import Path

import pytest

from videoannotator.validation.coco_validator import (
    COCOEvaluationResult,
    COCOValidationResult,
    COCOValidator,
)


def _make_coco_data(images=None, annotations=None, categories=None, info=None):
    """Build a minimal valid COCO dict, with overrides."""
    data = {
        "images": images
        if images is not None
        else [{"id": 1, "width": 640, "height": 480, "file_name": "frame_0001.jpg"}],
        "annotations": annotations
        if annotations is not None
        else [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 100, 200],
                "area": 20000,
                "iscrowd": 0,
            }
        ],
        "categories": categories
        if categories is not None
        else [{"id": 1, "name": "person", "supercategory": "person"}],
    }
    if info is not None:
        data["info"] = info
    return data


def _write_coco_json(data, tmp_path: Path) -> str:
    """Write COCO dict to a temp JSON file and return its path."""
    path = tmp_path / "coco_test.json"
    path.write_text(json.dumps(data))
    return str(path)


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestCOCOValidationResultModel:
    """Test the COCOValidationResult Pydantic model."""

    def test_defaults(self):
        result = COCOValidationResult(
            is_valid=True, pipeline_name="test", file_path="/tmp/test.json"
        )
        assert result.num_images == 0
        assert result.validation_errors == []
        assert result.validation_warnings == []

    def test_round_trip(self):
        result = COCOValidationResult(
            is_valid=False,
            pipeline_name="person",
            file_path="/data/out.json",
            num_images=10,
            num_annotations=50,
            num_categories=2,
            validation_errors=["Missing field: id"],
        )
        assert result.num_annotations == 50
        assert len(result.validation_errors) == 1


class TestCOCOEvaluationResultModel:
    """Test the COCOEvaluationResult Pydantic model."""

    def test_defaults(self):
        result = COCOEvaluationResult(pipeline_name="det", metric_type="bbox")
        assert result.ap_50_95 == 0.0
        assert result.stats == []


# ---------------------------------------------------------------------------
# Structure validation (no pycocotools needed)
# ---------------------------------------------------------------------------


class TestValidateCOCOStructure:
    """Tests for COCOValidator._validate_coco_structure."""

    @pytest.fixture
    def validator(self):
        return COCOValidator()

    def test_valid_structure(self, validator):
        data = _make_coco_data()
        errors = validator._validate_coco_structure(data)
        assert errors == []

    def test_missing_required_fields(self, validator):
        errors = validator._validate_coco_structure({})
        assert len(errors) == 3
        assert any("images" in e for e in errors)
        assert any("annotations" in e for e in errors)
        assert any("categories" in e for e in errors)

    def test_image_missing_required_fields(self, validator):
        data = _make_coco_data(images=[{"id": 1}])
        errors = validator._validate_coco_structure(data)
        assert any("width" in e for e in errors)
        assert any("height" in e for e in errors)
        assert any("file_name" in e for e in errors)

    def test_image_not_dict(self, validator):
        data = _make_coco_data(images=["not_a_dict"])
        errors = validator._validate_coco_structure(data)
        assert any("must be a dictionary" in e for e in errors)

    def test_annotation_missing_fields(self, validator):
        data = _make_coco_data(annotations=[{"id": 1}])
        errors = validator._validate_coco_structure(data)
        assert any("image_id" in e for e in errors)
        assert any("category_id" in e for e in errors)

    def test_annotation_not_dict(self, validator):
        data = _make_coco_data(annotations=[42])
        errors = validator._validate_coco_structure(data)
        assert any("must be a dictionary" in e for e in errors)

    def test_annotation_bad_bbox(self, validator):
        ann = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 20],  # wrong length
        }
        data = _make_coco_data(annotations=[ann])
        errors = validator._validate_coco_structure(data)
        assert any("bbox" in e for e in errors)

    def test_category_missing_name(self, validator):
        data = _make_coco_data(categories=[{"id": 1}])
        errors = validator._validate_coco_structure(data)
        assert any("name" in e for e in errors)

    def test_category_not_dict(self, validator):
        data = _make_coco_data(categories=["person"])
        errors = validator._validate_coco_structure(data)
        assert any("must be a dictionary" in e for e in errors)

    def test_info_not_dict(self, validator):
        data = _make_coco_data(info="bad")
        errors = validator._validate_coco_structure(data)
        assert any("info" in e and "dictionary" in e for e in errors)

    def test_info_dict_ok(self, validator):
        data = _make_coco_data(info={"description": "test"})
        errors = validator._validate_coco_structure(data)
        assert errors == []


# ---------------------------------------------------------------------------
# Video extension warnings
# ---------------------------------------------------------------------------


class TestValidateVideoExtensions:
    """Tests for COCOValidator._validate_video_extensions."""

    @pytest.fixture
    def validator(self):
        return COCOValidator()

    def test_no_video_fields_warns(self, validator):
        data = _make_coco_data()
        warnings = validator._validate_video_extensions(data)
        assert len(warnings) >= 1
        assert any("video fields" in w for w in warnings)

    def test_with_video_fields_no_warnings(self, validator):
        images = [
            {
                "id": 1,
                "width": 640,
                "height": 480,
                "file_name": "f.jpg",
                "video_id": "v1",
                "frame_number": 0,
                "timestamp": 0.0,
            }
        ]
        data = _make_coco_data(images=images)
        warnings = validator._validate_video_extensions(data)
        # Only the track_id warning should remain
        video_field_warnings = [w for w in warnings if "video fields" in w]
        assert len(video_field_warnings) == 0

    def test_no_track_id_warns(self, validator):
        data = _make_coco_data()
        warnings = validator._validate_video_extensions(data)
        assert any("track_id" in w for w in warnings)

    def test_track_id_present_no_warning(self, validator):
        ann = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 10, 10],
            "track_id": 1,
        }
        data = _make_coco_data(annotations=[ann])
        warnings = validator._validate_video_extensions(data)
        track_warnings = [w for w in warnings if "track_id" in w]
        assert len(track_warnings) == 0


# ---------------------------------------------------------------------------
# Full file validation (uses pycocotools COCO loader)
# ---------------------------------------------------------------------------


class TestValidateCOCOFile:
    """Tests for COCOValidator.validate_coco_file end-to-end."""

    @pytest.fixture
    def validator(self):
        return COCOValidator()

    def test_valid_file(self, validator, tmp_path):
        data = _make_coco_data()
        path = _write_coco_json(data, tmp_path)
        result = validator.validate_coco_file(path, pipeline_name="person")
        assert result.is_valid is True
        assert result.num_images == 1
        assert result.num_annotations == 1
        assert result.num_categories == 1
        assert result.pipeline_name == "person"

    def test_invalid_structure(self, validator, tmp_path):
        data = {"images": [{"id": 1}]}  # missing annotations, categories
        path = _write_coco_json(data, tmp_path)
        result = validator.validate_coco_file(path, pipeline_name="bad")
        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    def test_nonexistent_file(self, validator):
        result = validator.validate_coco_file("/nonexistent/path.json")
        assert result.is_valid is False
        assert len(result.validation_errors) > 0


# ---------------------------------------------------------------------------
# Validate all pipelines
# ---------------------------------------------------------------------------


class TestValidateAllPipelines:
    """Tests for COCOValidator.validate_all_pipelines."""

    @pytest.fixture
    def validator(self):
        return COCOValidator()

    def test_empty_directory(self, validator, tmp_path):
        results = validator.validate_all_pipelines(str(tmp_path))
        # All three pipeline patterns should be reported as not found
        assert len(results) == 3
        for r in results.values():
            assert r.is_valid is False
            assert any("No files found" in e for e in r.validation_errors)

    def test_with_valid_scene_file(self, validator, tmp_path):
        data = _make_coco_data()
        (tmp_path / "video_scenes_coco.json").write_text(json.dumps(data))
        results = validator.validate_all_pipelines(str(tmp_path))
        assert results["scene"].is_valid is True


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


class TestCreateValidationReport:
    """Tests for COCOValidator.create_validation_report."""

    @pytest.fixture
    def validator(self):
        return COCOValidator()

    def test_report_all_valid(self, validator):
        results = {
            "scene": COCOValidationResult(
                is_valid=True,
                pipeline_name="scene",
                file_path="scene.json",
                num_images=5,
                num_annotations=10,
                num_categories=1,
            ),
        }
        report = validator.create_validation_report(results)
        assert "1/1 pipelines valid" in report
        assert "SCENE" in report

    def test_report_with_errors(self, validator):
        results = {
            "person": COCOValidationResult(
                is_valid=False,
                pipeline_name="person",
                file_path="missing.json",
                validation_errors=["File not found"],
            ),
        }
        report = validator.create_validation_report(results)
        assert "0/1 pipelines valid" in report
        assert "File not found" in report
        assert "Recommendations" in report

    def test_report_with_warnings(self, validator):
        results = {
            "face": COCOValidationResult(
                is_valid=True,
                pipeline_name="face",
                file_path="face.json",
                num_images=2,
                validation_warnings=["Missing video fields"],
            ),
        }
        report = validator.create_validation_report(results)
        assert "Missing video fields" in report
