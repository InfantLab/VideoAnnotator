"""Unit tests for coordinate transform utility functions."""

import numpy as np
import pandas as pd
import pytest

from videoannotator.utils.transform_utils import (
    denormaliseCoordinates,
    normaliseCoordinates,
    xyxy2ltwh,
)


class TestNormaliseCoordinates:
    """Tests for normaliseCoordinates function."""

    def test_basic_normalisation(self):
        """Pixel coords are divided by frame dimensions."""
        df = pd.DataFrame(
            {
                "x": [320.0, 640.0],
                "y": [240.0, 480.0],
            }
        )
        result = normaliseCoordinates(
            df, xcols=[0], ycols=[1], frameHeight=480, frameWidth=640
        )
        assert result.iloc[0, 0] == pytest.approx(0.5)
        assert result.iloc[1, 0] == pytest.approx(1.0)
        assert result.iloc[0, 1] == pytest.approx(0.5)
        assert result.iloc[1, 1] == pytest.approx(1.0)

    def test_zero_coords_stay_zero(self):
        """Zero coordinates remain zero after normalisation."""
        df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        result = normaliseCoordinates(
            df, xcols=[0], ycols=[1], frameHeight=1080, frameWidth=1920
        )
        assert result.iloc[0, 0] == pytest.approx(0.0)
        assert result.iloc[0, 1] == pytest.approx(0.0)

    def test_multiple_columns(self):
        """Multiple x/y column indices are all normalised."""
        df = pd.DataFrame(
            {
                "x1": [640.0],
                "y1": [480.0],
                "x2": [320.0],
                "y2": [240.0],
            }
        )
        result = normaliseCoordinates(
            df, xcols=[0, 2], ycols=[1, 3], frameHeight=480, frameWidth=640
        )
        assert result.iloc[0, 0] == pytest.approx(1.0)
        assert result.iloc[0, 2] == pytest.approx(0.5)


class TestDenormaliseCoordinates:
    """Tests for denormaliseCoordinates function."""

    def test_basic_denormalisation(self):
        """Normalised coords are multiplied by frame dimensions."""
        df = pd.DataFrame({"x": [0.5, 1.0], "y": [0.5, 1.0]})
        result = denormaliseCoordinates(
            df, xcols=["x"], ycols=["y"], frameHeight=480, frameWidth=640
        )
        assert result["x"].iloc[0] == pytest.approx(320.0)
        assert result["x"].iloc[1] == pytest.approx(640.0)
        assert result["y"].iloc[0] == pytest.approx(240.0)
        assert result["y"].iloc[1] == pytest.approx(480.0)

    def test_roundtrip(self):
        """Normalise then denormalise returns original values."""
        df = pd.DataFrame({"x": [320.0], "y": [240.0]})
        width, height = 640, 480
        normalised = normaliseCoordinates(
            df.copy(), xcols=[0], ycols=[1], frameHeight=height, frameWidth=width
        )
        result = denormaliseCoordinates(
            normalised, xcols=["x"], ycols=["y"], frameHeight=height, frameWidth=width
        )
        assert result["x"].iloc[0] == pytest.approx(320.0)
        assert result["y"].iloc[0] == pytest.approx(240.0)


class TestXyxy2ltwh:
    """Tests for xyxy2ltwh bounding box conversion."""

    def test_basic_conversion(self):
        """[x1, y1, x2, y2] converts to [x, y, w, h]."""
        result = xyxy2ltwh([10, 20, 50, 80])
        assert result == [10, 20, 40, 60]

    def test_zero_size_box(self):
        """Box with zero area returns zero width/height."""
        result = xyxy2ltwh([5, 5, 5, 5])
        assert result == [5, 5, 0, 0]

    def test_numpy_input(self):
        """Accepts numpy array input and returns list."""
        bbox = np.array([10, 20, 50, 80])
        result = xyxy2ltwh(bbox)
        assert isinstance(result, list)
        assert result == [10, 20, 40, 60]

    def test_float_coordinates(self):
        """Handles float coordinates correctly."""
        result = xyxy2ltwh([10.5, 20.3, 50.7, 80.9])
        assert result[0] == pytest.approx(10.5)
        assert result[1] == pytest.approx(20.3)
        assert result[2] == pytest.approx(40.2)
        assert result[3] == pytest.approx(60.6)
