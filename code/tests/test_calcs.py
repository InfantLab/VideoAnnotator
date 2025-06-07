import unittest
import numpy as np
from src.calcs import avgxys

class TestAvgxys(unittest.TestCase):
    def test_avgxys(self):
        # Test case 1: all confidences above threshold
        xyc = np.array([[1, 2, 0.6], [3, 4, 0.7], [5, 6, 0.8]])
        threshold = 0.5
        avgx, avgy = avgxys(xyc, threshold)
        self.assertAlmostEqual(avgx, 3.0)
        self.assertAlmostEqual(avgy, 4.0)

        # Test case 2: no confidences above threshold
        xyc = np.array([[1, 2, 0.4], [3, 4, 0.3], [5, 6, 0.2]])
        threshold = 0.5
        avgx, avgy = avgxys(xyc, threshold)
        self.assertIsNone(avgx)
        self.assertIsNone(avgy)

        # Test case 3: some confidences above threshold
        xyc = np.array([[1, 2, 0.3], [1, 2, 0.6], [5, 6, 0.8]])
        threshold = 0.5
        avgx, avgy = avgxys(xyc, threshold)
        self.assertAlmostEqual(avgx, 3.0)
        self.assertAlmostEqual(avgy, 4.0)

if __name__ == '__main__':
    unittest.main()
    import pytest
import pandas as pd
from src.calcs import centreOfGravity

# Test data setup
@pytest.fixture
def sample_data():
    data = {
        "frame": [1, 1, 2, 2],
        "person": [1, 2, 1, 2],
        "keypoint1_x": [100, 150, 200, 250],
        "keypoint1_y": [50, 60, 70, 80],
        "keypoint1_conf": [0.9, 0.8, 0.7, 0.6],
        # Add more keypoints if needed
    }
    columns = ["frame", "person"] + [f"keypoint{i}_{axis}" for i in range(1, 2) for axis in ["x", "y", "conf"]]
    return pd.DataFrame(data, columns=columns)

@pytest.mark.parametrize("frames, people, bodypart, expected_cog_x, expected_cog_y, test_id", [
    # Happy path tests
    ([1, 2], [1, 2], "whole", [175.0, 225.0], [55.0, 75.0], "test_happy_path_all_frames_people"),
    ([1], [1], "whole", [100.0], [50.0], "test_happy_path_single_frame_person"),
    
    # Edge cases
    ([], [], "whole", [175.0, 225.0], [55.0, 75.0], "test_edge_empty_frames_people"),
    ([3], [3], "whole", [None], [None], "test_edge_nonexistent_frame_person"),
    
    # Error cases
    ([1, 2], [1, 2], "hand", None, None, "test_error_unimplemented_bodypart"),
])
def test_centreOfGravity(sample_data, frames, people, bodypart, expected_cog_x, expected_cog_y, test_id):
    # Arrange
    df = sample_data.copy()

    # Act
    if "error" in test_id:
        with pytest.raises(NotImplementedError):
            df = centreOfGravity(df, frames, people, bodypart)
    else:
        df = centreOfGravity(df, frames, people, bodypart)

    # Assert
    if "error" not in test_id:
        assert df["cog.x"].dropna().tolist() == expected_cog_x, f"Failed {test_id}: cog.x"
        assert df["cog.y"].dropna().tolist() == expected_cog_y, f"Failed {test_id}: cog.y"
