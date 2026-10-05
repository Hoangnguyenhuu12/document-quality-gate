"""
Resolution and Dimensional DPI Estimator for Document Quality Gate.
"""

import numpy as np


class ResolutionDetector:
    """Estimates effective DPI and character stroke rendering adequacy."""

    @staticmethod
    def estimate_dpi(width: int, height: int, standard_diagonal_inches: float = 14.3) -> int:
        """
        Estimate effective DPI assuming standard A4 document diagonal (~14.3 inches).
        """
        if width <= 0 or height <= 0:
            return 0
        pixel_diagonal = (width**2 + height**2)**0.5
        dpi = int(pixel_diagonal / standard_diagonal_inches)
        return max(50, min(dpi, 600))
