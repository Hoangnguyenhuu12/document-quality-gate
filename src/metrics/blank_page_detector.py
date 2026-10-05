"""
Blank Page Detector for Scanned Document Pages.
Distinguishes intentional blank pages (e.g. duplex scans, blank backs of contracts)
from defective or blurry documents.
"""

import numpy as np


class BlankPageDetector:
    """Detects if a document page has near-zero content (blank sheet)."""

    @staticmethod
    def is_blank_page(
        gray_array: np.ndarray,
        ink_threshold: float = 230.0,
        max_ink_density: float = 0.35,
        max_variance: float = 20.0
    ) -> bool:
        """
        Check if a page is effectively blank.

        Args:
            gray_array: Grayscale image array (0 to 255).
            ink_threshold: Pixels darker than this value are considered ink/markings.
            max_ink_density: Maximum percentage of ink pixels allowed (e.g. 0.35% for minor dust/specks).
            max_variance: Maximum allowed pixel intensity standard deviation for blank paper.

        Returns:
            bool: True if page is blank, False otherwise.
        """
        if gray_array.size == 0:
            return True

        # Percentage of pixels darker than ink_threshold
        ink_pixels = np.sum(gray_array < ink_threshold)
        ink_density = (ink_pixels / gray_array.size) * 100.0

        # Standard deviation of pixel intensities across the page
        std_dev = float(np.std(gray_array))

        # Blank page condition: very low ink density AND relatively uniform paper
        if ink_density < max_ink_density and std_dev < max_variance:
            return True

        return False
