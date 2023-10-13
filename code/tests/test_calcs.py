import unittest
import numpy as np
from calcs import avgxys

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