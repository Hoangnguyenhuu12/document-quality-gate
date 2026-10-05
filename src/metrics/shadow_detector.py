"""
Shadow and Illumination Uniformity Detector.
Computes luminance disparity across document grid quadrants.
"""

import numpy as np


class ShadowDetector:
    """Detects uneven lighting, gradient shadows, and flashlight hotspots."""

    @staticmethod
    def compute_shadow_disparity(gray_array: np.ndarray, grid_size: int = 3) -> float:
        """
        Divide the page into an N x N grid and measure standard deviation of quadrant means.
        High disparity indicates harsh corner shadows, folds, or spotlight glare.
        """
        h, w = gray_array.shape[:2]
        if h < grid_size * 2 or w < grid_size * 2:
            return 0.0

        block_h = h // grid_size
        block_w = w // grid_size

        means = []
        for r in range(grid_size):
            for c in range(grid_size):
                sub = gray_array[r * block_h : (r + 1) * block_h, c * block_w : (c + 1) * block_w]
                means.append(float(np.mean(sub)))

        disparity = float(np.std(means))
        return round(disparity, 2)
