"""
Orientation Detector for Scanned Document Pages.
Detects coarse page rotation (0, 90, 180, 270 degrees) to orient landscape or inverted scans.
"""

from typing import Tuple
import numpy as np
from PIL import Image


class OrientationDetector:
    """Detects coarse 90-degree increments of document page orientation."""

    @staticmethod
    def detect_coarse_rotation(gray_array: np.ndarray) -> int:
        """
        Determines if text orientation is standard horizontal (0 deg) or vertical (90/270 deg).
        Returns:
            rotation_needed: 0 or 90 degrees.
        """
        h, w = gray_array.shape[:2]
        if h < 30 or w < 30:
            return 0

        # Downsample for fast projection analysis
        scale = min(1.0, 500.0 / max(h, w))
        if scale < 1.0:
            pil_img = Image.fromarray(gray_array.astype(np.uint8))
            new_w = max(10, int(w * scale))
            new_h = max(10, int(h * scale))
            small = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            sample = np.array(small, dtype=np.float32)
        else:
            sample = gray_array.astype(np.float32)

        # Gradient along vertical and horizontal
        grad_y = np.abs(sample[1:, :] - sample[:-1, :])
        grad_x = np.abs(sample[:, 1:] - sample[:, :-1])

        # Binarize edges
        edge_y = (grad_y > np.percentile(grad_y, 75)).astype(np.float32)
        edge_x = (grad_x > np.percentile(grad_x, 75)).astype(np.float32)

        # Horizontal projection profile: sum rows (detects horizontal text line gaps)
        h_proj = np.sum(edge_y, axis=1)
        h_var = float(np.var(h_proj))

        # Vertical projection profile: sum columns (detects vertical text line gaps)
        v_proj = np.sum(edge_x, axis=0)
        v_var = float(np.var(v_proj))

        # If vertical text gap variance is significantly higher than horizontal,
        # the page is rotated sideways (e.g. 90 degrees)
        if v_var > h_var * 1.6 and w > h:
            return 90

        return 0
