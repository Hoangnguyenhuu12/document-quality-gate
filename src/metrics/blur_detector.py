"""
Blur Detector based on Laplacian High-Frequency Energy Variance.
Measures high-frequency transition sharpness across document text.
"""

import numpy as np


class BlurDetector:
    """Detects defocus and motion blur via discrete Laplacian operator."""

    @staticmethod
    def compute_blur_score(gray_array: np.ndarray) -> float:
        """
        Compute variance of the Laplacian filter.
        High values indicate crisp, sharp edges. Low values indicate blur.
        """
        if gray_array.shape[0] < 3 or gray_array.shape[1] < 3:
            return 0.0

        # Discrete 3x3 Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        # Vectorized 2D convolution with numpy slicing
        center = gray_array[1:-1, 1:-1]
        top = gray_array[:-2, 1:-1]
        bottom = gray_array[2:, 1:-1]
        left = gray_array[1:-1, :-2]
        right = gray_array[1:-1, 2:]

        laplacian = top + bottom + left + right - 4.0 * center
        variance = float(np.var(laplacian))
        return round(variance, 2)
