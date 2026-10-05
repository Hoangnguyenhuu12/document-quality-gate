"""
Skew and Orientation Detector for Scanned Document Pages.
Measures document tilt / skew angle using projection profile analysis.
"""

from typing import Tuple
import numpy as np
from PIL import Image


class SkewDetector:
    """Detects tilt/skew angle in degrees for scanned contracts and documents."""

    @staticmethod
    def estimate_skew_angle(gray_array: np.ndarray, max_angle: float = 15.0, angle_step: float = 0.5) -> float:
        """
        Estimate skew angle (in degrees) by finding the angle that maximizes
        the variance of horizontal projection profile (text line alignment).

        Returns:
            angle: Skew angle in degrees (-max_angle to +max_angle).
                   Positive = tilted clockwise, Negative = counter-clockwise.
        """
        h, w = gray_array.shape[:2]
        if h < 20 or w < 20:
            return 0.0

        # Downsample for high-speed calculation (target max dimension ~600px)
        scale = min(1.0, 600.0 / max(h, w))
        if scale < 1.0:
            pil_img = Image.fromarray(gray_array.astype(np.uint8))
            new_w = max(10, int(w * scale))
            new_h = max(10, int(h * scale))
            small_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            sample = np.array(small_img, dtype=np.float32)
        else:
            sample = gray_array.astype(np.float32)

        # Gradient / Edge magnitude to highlight text baselines
        grad_y = np.abs(sample[1:, :] - sample[:-1, :])
        thresh = np.percentile(grad_y, 80)
        edges = (grad_y > thresh).astype(np.float32)

        best_score = -1.0
        best_angle = 0.0

        pil_edges = Image.fromarray((edges * 255).astype(np.uint8))

        angles = np.arange(-max_angle, max_angle + angle_step, angle_step)
        for ang in angles:
            # Rotate edges
            rotated = pil_edges.rotate(float(ang), resample=Image.Resampling.NEAREST)
            rot_arr = np.array(rotated, dtype=np.float32)

            # Horizontal projection: sum across columns (along text lines)
            proj = np.sum(rot_arr, axis=1)

            # Variance of the projection profile is highest when text lines align horizontally
            score = float(np.var(proj))
            if score > best_score:
                best_score = score
                best_angle = float(ang)

        return round(best_angle, 2)
