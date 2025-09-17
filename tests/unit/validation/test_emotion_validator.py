from src.validation.emotion_validator import validate_emotion_data


def test_valid_emotion_minimal():
    data = {
        "schema_version": 1,
        "source_pipeline": "face_laion_clip",
        "emotions": [
            {
                "start": 0.0,
                "end": 0.5,
                "labels": [
                    {"label": "happy", "confidence": 0.9},
                    {"label": "neutral", "confidence": 0.05},
                ],
                "confidence": 0.9,
                "source": {"modality": "video"},
            }
        ],
    }
    errors = validate_emotion_data(data)
    assert errors == []


def test_invalid_emotion_confidence_mismatch():
    data = {
        "schema_version": 1,
        "source_pipeline": "face_laion_clip",
        "emotions": [
            {
                "start": 0.0,
                "end": 0.5,
                "labels": [
                    {"label": "happy", "confidence": 0.9},
                    {"label": "neutral", "confidence": 0.05},
                ],
                "confidence": 0.8,  # mismatch
                "source": {"modality": "video"},
            }
        ],
    }
    errors = validate_emotion_data(data)
    assert any("confidence should match" in e for e in errors)


def test_invalid_duplicate_labels():
    data = {
        "schema_version": 1,
        "source_pipeline": "face_laion_clip",
        "emotions": [
            {
                "start": 0.0,
                "end": 0.5,
                "labels": [
                    {"label": "happy", "confidence": 0.9},
                    {"label": "happy", "confidence": 0.8},
                ],
                "source": {"modality": "video"},
            }
        ],
    }
    errors = validate_emotion_data(data)
    assert any("duplicate label" in e for e in errors)


def test_missing_required_top_level():
    data = {"source_pipeline": "x", "emotions": []}  # missing schema_version & empty emotions
    errors = validate_emotion_data(data)
    assert any("Missing top-level field" in e for e in errors)
    assert any("emotions must be a non-empty list" in e for e in errors)
