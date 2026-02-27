"""Unit tests for mathematical utility functions."""

import numpy as np
import pandas as pd
import pytest

from videoannotator.utils.math_utils import (
    avgxys,
    centreOfGravity,
    rowcogs,
    rowstds,
    stdevxys,
)


class TestAvgxys:
    """Tests for avgxys function."""

    def test_basic_average(self):
        """All points above threshold return correct mean."""
        xyc = np.array(
            [
                [10.0, 20.0, 0.9],
                [30.0, 40.0, 0.8],
            ]
        )
        avgx, avgy = avgxys(xyc, threshold=0.5)
        assert avgx == pytest.approx(20.0)
        assert avgy == pytest.approx(30.0)

    def test_threshold_filtering(self):
        """Points below threshold are excluded."""
        xyc = np.array(
            [
                [10.0, 20.0, 0.9],
                [100.0, 200.0, 0.1],  # below threshold
            ]
        )
        avgx, avgy = avgxys(xyc, threshold=0.5)
        assert avgx == pytest.approx(10.0)
        assert avgy == pytest.approx(20.0)

    def test_all_below_threshold_returns_nan(self):
        """All points below threshold returns NaN."""
        xyc = np.array(
            [
                [10.0, 20.0, 0.1],
                [30.0, 40.0, 0.2],
            ]
        )
        avgx, avgy = avgxys(xyc, threshold=0.5)
        assert np.isnan(avgx)
        assert np.isnan(avgy)

    def test_single_point(self):
        """Single point above threshold returns that point."""
        xyc = np.array([[5.0, 15.0, 0.95]])
        avgx, avgy = avgxys(xyc, threshold=0.5)
        assert avgx == pytest.approx(5.0)
        assert avgy == pytest.approx(15.0)

    def test_custom_threshold(self):
        """Custom threshold value is respected."""
        xyc = np.array(
            [
                [10.0, 20.0, 0.3],
                [30.0, 40.0, 0.7],
            ]
        )
        avgx, avgy = avgxys(xyc, threshold=0.2)
        assert avgx == pytest.approx(20.0)
        assert avgy == pytest.approx(30.0)


class TestStdevxys:
    """Tests for stdevxys function."""

    def test_basic_std(self):
        """Standard deviation computed correctly for points above threshold."""
        xyc = np.array(
            [
                [10.0, 20.0, 0.9],
                [30.0, 40.0, 0.8],
            ]
        )
        stdx, stdy = stdevxys(xyc, threshold=0.5)
        assert stdx == pytest.approx(np.std([10.0, 30.0]))
        assert stdy == pytest.approx(np.std([20.0, 40.0]))

    def test_threshold_filtering(self):
        """Low-confidence points excluded from std calculation."""
        xyc = np.array(
            [
                [10.0, 20.0, 0.9],
                [100.0, 200.0, 0.1],  # excluded
            ]
        )
        stdx, stdy = stdevxys(xyc, threshold=0.5)
        # Single point -> std is 0
        assert stdx == pytest.approx(0.0)
        assert stdy == pytest.approx(0.0)

    def test_identical_points(self):
        """Identical points yield zero standard deviation."""
        xyc = np.array(
            [
                [5.0, 5.0, 0.9],
                [5.0, 5.0, 0.9],
            ]
        )
        stdx, stdy = stdevxys(xyc, threshold=0.5)
        assert stdx == pytest.approx(0.0)
        assert stdy == pytest.approx(0.0)


class TestRowcogs:
    """Tests for rowcogs function."""

    def test_1d_keypoints(self):
        """Computes average from a 1-D keypoint series."""
        # 2 keypoints: (10, 20, 0.9), (30, 40, 0.8)
        data = pd.Series([10.0, 20.0, 0.9, 30.0, 40.0, 0.8])
        avgx, avgy = rowcogs(data, threshold=0.5)
        assert avgx == pytest.approx(20.0)
        assert avgy == pytest.approx(30.0)

    def test_filtering_low_confidence(self):
        """Low-confidence keypoints are filtered out."""
        data = pd.Series([10.0, 20.0, 0.9, 100.0, 200.0, 0.1])
        avgx, avgy = rowcogs(data, threshold=0.5)
        assert avgx == pytest.approx(10.0)
        assert avgy == pytest.approx(20.0)


class TestRowstds:
    """Tests for rowstds function."""

    def test_1d_keypoints_std(self):
        """Computes std from a 1-D keypoint series."""
        data = pd.Series([10.0, 20.0, 0.9, 30.0, 40.0, 0.8])
        stdx, stdy = rowstds(data, threshold=0.5)
        assert stdx == pytest.approx(np.std([10.0, 30.0]))
        assert stdy == pytest.approx(np.std([20.0, 40.0]))


class TestCentreOfGravity:
    """Tests for centreOfGravity function."""

    @pytest.fixture
    def keypoints_df(self):
        """Create a DataFrame mimicking the person tracking output.

        Columns 0-7 are metadata, columns 8-58 are keypoints (17 * 3 = 51 values).
        """
        cols = ["frame", "person", "x1", "y1", "x2", "y2", "conf", "cls"] + [
            f"kp{i}" for i in range(51)
        ]
        # Two frames, one person each with all keypoints at known positions
        kp_vals_person0 = [10.0, 20.0, 0.9] * 17  # avg x=10, y=20
        kp_vals_person1 = [30.0, 40.0, 0.8] * 17  # avg x=30, y=40
        rows = [
            [0, 0, 0, 0, 100, 100, 0.95, 0] + kp_vals_person0,
            [1, 0, 0, 0, 100, 100, 0.90, 0] + kp_vals_person1,
        ]
        return pd.DataFrame(rows, columns=cols)

    def test_adds_cog_columns(self, keypoints_df):
        """Adds cog.x and cog.y columns to the DataFrame."""
        result = centreOfGravity(keypoints_df)
        assert "cog.x" in result.columns
        assert "cog.y" in result.columns

    def test_correct_cog_values(self, keypoints_df):
        """Computes correct center of gravity per frame/person."""
        result = centreOfGravity(keypoints_df)
        row0 = result.iloc[0]
        assert row0["cog.x"] == pytest.approx(10.0)
        assert row0["cog.y"] == pytest.approx(20.0)

    def test_specific_frames(self, keypoints_df):
        """Only processes specified frames."""
        result = centreOfGravity(keypoints_df, frames=[0])
        row0 = result.iloc[0]
        row1 = result.iloc[1]
        assert row0["cog.x"] == pytest.approx(10.0)
        assert np.isnan(row1["cog.x"])

    def test_specific_people(self, keypoints_df):
        """Only processes specified people."""
        result = centreOfGravity(keypoints_df, people=[0])
        assert not np.isnan(result.iloc[0]["cog.x"])

    def test_non_whole_bodypart_raises(self, keypoints_df):
        """Non-'whole' bodypart raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            centreOfGravity(keypoints_df, bodypart="head")
