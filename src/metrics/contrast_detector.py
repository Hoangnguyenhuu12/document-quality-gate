"""
Contrast and Dynamic Range Detector for Document Pages.
Evaluates root-mean-square (RMS) contrast and pixel distribution.
"""

import numpy as np


class ContrastDetector:
    """Evaluates whether document page has sufficient black-white contrast for OCR."""

    @staticmethod
    def compute_contrast_score(gray_array: np.ndarray) -> float:
        """
        Compute standard deviation of pixel intensities (RMS contrast).
        Higher values indicate strong contrast between ink and paper.
        """
        if gray_array.size == 0:
            return 0.0
        contrast = float(np.std(gray_array))
        return round(contrast, 2)
