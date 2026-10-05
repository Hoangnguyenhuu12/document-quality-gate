"""
Glare and Specular Reflection Detector for Document Pages.
Detects camera flash hotspots, glossy sleeve reflections, and washed-out highlights.
"""

import numpy as np


class GlareDetector:
    """Detects overexposed specular highlights and camera flash on document surfaces."""

    @staticmethod
    def compute_glare_score(gray_array: np.ndarray, threshold: float = 250.0) -> float:
        """
        Compute percentage of severely overexposed / saturated pixels caused by specular reflections.
        Distinguishes between a uniformly clean white page (born-digital or thresholded scan)
        and an actual camera flash/glare hotspot on a photographed document.

        Returns:
            glare_percentage (0.0 to 100.0)
        """
        if gray_array.size == 0:
            return 0.0

        median_lum = float(np.median(gray_array))
        # If the overall page median is already near pure white (clean digital document or scan),
        # this is normal paper background, not camera flash reflection.
        if median_lum >= 245.0:
            return 0.0

        # In photographed contracts/documents, paper is typically 120-235.
        # Pixels saturated near 250+ in such documents indicate camera flash, hotspot, or glossy sleeve glare.
        saturated_pixels = np.sum(gray_array >= threshold)
        total_pixels = gray_array.size

        glare_ratio = (saturated_pixels / total_pixels) * 100.0
        return round(float(glare_ratio), 2)

